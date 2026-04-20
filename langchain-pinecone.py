"""RAG pipeline using LangChain, Pinecone, and Azure Files.

Ingests documents from an Azure file share, converts files to LangChain
Documents, indexes into Pinecone, and provides an interactive Q&A loop.
"""

import os
import tempfile


from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from azure_files import DownloadedFile, connect_to_share, download_files, list_share_files
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

load_dotenv()

# Azure Storage
STORAGE_ACCOUNT_NAME = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
SHARE_NAME = os.environ["AZURE_STORAGE_SHARE_NAME"]

# Azure OpenAI
OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
OPENAI_CHAT_DEPLOYMENT = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

# RAG tuning
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

# Azure authentication
CREDENTIAL = DefaultAzureCredential()
TOKEN_PROVIDER = get_bearer_token_provider(
    CREDENTIAL, "https://cognitiveservices.azure.com/.default"
)

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

# Mapping of file extensions to LangChain document loaders and their kwargs
LOADER_MAP: dict[str, tuple] = {
    ".pdf": (PyPDFLoader, {}),
    ".docx": (Docx2txtLoader, {}),
    ".csv": (CSVLoader, {}),
    ".tsv": (CSVLoader, {"csv_args": {"delimiter": "\t"}}),
}

DEFAULT_LOADER = (TextLoader, {"encoding": "utf-8"})


def parse_downloaded_files(
    downloaded_files: list[DownloadedFile]
) -> list[Document]:
    """Parse downloaded files from an Azure file share into LangChain Documents.

    Args:
        downloaded_files: A list of DownloadedFile objects, each representing a
            file in an Azure file share, containing the path and access control
            metadata for a file.

    Returns: A list of LangChain Documents.
    """
    documents = []
   
    for info in downloaded_files:
        file_ext = os.path.splitext(info.file_name.lower())[1]
        loader_cls, kwargs = LOADER_MAP.get(file_ext, DEFAULT_LOADER)
        
        try:
            docs = loader_cls(info.local_path, **kwargs).load()
        except Exception:
            print(f"Failed to parse {info.relative_path}, skipping...")
            continue

        for doc in docs:
            doc.metadata.update({
                "azure_file_path": info.relative_path,
                "file_name": info.file_name,
            })
        documents.extend(docs)
    
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for embedding.

    Args:
        documents: A list of LangChain Documents to split.

    Returns: A list of smaller Document chunks with preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def embed_and_index(chunks: list[Document]) -> PineconeVectorStore:
    """Embed document chunks via Azure OpenAI and upsert into a Pinecone index.

    Args:
        chunks: A list of chunked LangChain Documents to embed and index.

    Returns: A PineconeVectorStore connected to the populated index.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # If it does not already exist, create the Pinecone index
    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="azure", region="eastus2"),
        )

    index = pc.Index(PINECONE_INDEX_NAME)

    # Reset index if specified via environment variable. This is useful for
    # development to ensure a clean slate on each run. In production, you would
    # typically not reset the index.
    if os.getenv("RESET_INDEX") == "true":
        index.delete(delete_all=True)

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment=OPENAI_EMBEDDING_DEPLOYMENT,
        azure_ad_token_provider=TOKEN_PROVIDER,
        dimensions=EMBEDDING_DIMENSIONS,
    )

    return PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )


def build_qa_chain(
    vector_store: PineconeVectorStore,
):
    """Build a retrieval question-answering (Q&A) chain.

    Args:
        vector_store: PineconeVectorStore to retrieve from.
    """
    llm = AzureChatOpenAI(
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment=OPENAI_CHAT_DEPLOYMENT,
        azure_ad_token_provider=TOKEN_PROVIDER,
        api_version="2024-12-01-preview",
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    prompt = PromptTemplate.from_template(
        "Answer the question based on the context below. "
        "Be specific and cite the source file name in brackets for each fact.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\nAnswer:"
    )

    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(
            f"[{d.metadata.get('azure_file_path', '')}]\n{d.page_content}"
            for d in docs
        )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def main():
    """Main execution flow."""
    share = connect_to_share(STORAGE_ACCOUNT_NAME, SHARE_NAME, CREDENTIAL)

    # 1. List files from the share
    print("Scanning file share...")
    file_references = list_share_files(share)
    if not file_references:
        print("No files found.")
        return
    print(f"Found {len(file_references)} files.\n")

    # 2. Download files (shared Azure Files logic)
    print("Downloading files onto temporary local directory...")
    with tempfile.TemporaryDirectory() as temp_directory:
        downloaded = download_files(file_references, temp_directory)
        if not downloaded:
            print("No files downloaded.")
            return
        print()

        # 3. Parse into LangChain Documents
        print("Parsing files...")
        documents = parse_downloaded_files(downloaded)
    
    if not documents:
        print("No documents parsed.")
        return
    print(f"{len(documents)} documents.\n")

    # 4. Chunk
    print("Splitting into chunks...")
    chunks = chunk_documents(documents)
    print(f"{len(documents)} docs → {len(chunks)} chunks.\n")

    # 5. Embed and index
    print("Indexing into Pinecone...")
    store = embed_and_index(chunks)
    print(f"{len(chunks)} chunks indexed.\n")

    qa_chain = build_qa_chain(store)
    print("Ready. Type 'quit' to exit.\n")

    try:
        while True:
            question = input("You: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue
            print(f"\nAnswer: {qa_chain.invoke(question)}\n")
    except KeyboardInterrupt:
        pass

    print("\nDone.")


if __name__ == "__main__":
    main()
