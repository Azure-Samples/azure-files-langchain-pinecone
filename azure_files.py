import os
import posixpath
from typing import NamedTuple

from azure.core.credentials import TokenCredential
from azure.storage.fileshare import ShareClient, ShareDirectoryClient


class DownloadedFile(NamedTuple):
    """A file downloaded from an Azure file share into a local temporary directory.

    Attributes:
        local_path: Path to the downloaded file on disk.
        file_name: The original file name.
        relative_path: The file's path relative to the specified start directory
            (default is share root).
    """
    local_path: str
    file_name: str
    relative_path: str


def connect_to_share(
    account_name: str,
    share_name: str,
    credential: TokenCredential,
) -> ShareClient:
    """Create and return a ShareClient for an Azure file share.
    
    Args:        
        account_name: The name of the Azure storage account.
        share_name: The name of the Azure file share.
        credential: A token credential for Azure authentication.
    
    Returns: A ShareClient connected to the specified share.
    """
    return ShareClient(
        account_url=f"https://{account_name}.file.core.windows.net",
        share_name=share_name,
        credential=credential,
        token_intent="backup",
    )


def list_share_files(
    share: ShareClient, 
    start_directory_path=""
) -> list[tuple[str, str, ShareDirectoryClient]]:
    """Walks the Azure file share and returns a list of
    (name, relative_path, parent_directory) tuples for each file.

    Args:
        share: ShareClient for the target file share.
        start_directory_path: Subdirectory to start from. Defaults to share root.

    Returns: A list of (name, relative_path, parent_directory) tuples.
    """
    root = share.get_directory_client(start_directory_path)

    file_references = []
    directories_to_traverse = [root]

    while directories_to_traverse:
        current = directories_to_traverse.pop()
        for item in current.list_directories_and_files():
            if item.is_directory:
                directories_to_traverse.append(current.get_subdirectory_client(item.name))
            else:
                # Azure Files paths use posix-style separators.
                relative_path = posixpath.join(current.directory_path or "", item.name)
                file_references.append((item.name, relative_path, current))
    
    return file_references


def download_files(
    file_references: list[tuple[str, str, ShareDirectoryClient]],
    destination: str,
) -> list[DownloadedFile]:
    """Downloads files from an Azure file share into a local directory.

    Args:
        file_references: A list of (name, relative_path, parent_directory) tuples.
        destination: Local directory to download files into.

    Returns: A list of DownloadedFile objects with the local path and original
        file metadata.
    """
    downloaded_files = []

    for name, relative_path, parent_directory in file_references:
        file_client = parent_directory.get_file_client(name)

        # Ensure the download path stays within the destination directory
        local_path = os.path.join(destination, relative_path)
        real_dest = os.path.realpath(destination) + os.sep
        if not os.path.realpath(local_path).startswith(real_dest):
            raise ValueError(f"Path traversal detected: {relative_path}")

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            for chunk in file_client.download_file().chunks():
                f.write(chunk)

        downloaded_files.append(DownloadedFile(
            local_path=local_path,
            file_name=name,
            relative_path=relative_path,
        ))

    return downloaded_files
