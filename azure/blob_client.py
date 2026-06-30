# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import logging

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)


class ElephantBlobClient:
    def __init__(self, account: str = "ganeshasfc2o4rujo76u", container: str = "elephant-images"):
        credential = DefaultAzureCredential()
        self._service = BlobServiceClient(
            account_url=f"https://{account}.blob.core.windows.net",
            credential=credential,
        )
        self._container = container
        self._client = self._service.get_container_client(container)

    def download(self, blob_path: str, dest: str) -> None:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        blob_client = self._client.get_blob_client(blob_path)
        with open(dest, "wb") as f:
            stream = blob_client.download_blob()
            stream.readinto(f)

    def sync_prefix(self, prefix: str, dest_dir: str) -> list[str]:
        os.makedirs(dest_dir, exist_ok=True)
        downloaded = []
        for blob in self._client.list_blobs(name_starts_with=prefix):
            blob_name = blob.name
            local_path = os.path.join(dest_dir, os.path.basename(blob_name))
            if not os.path.exists(local_path):
                self.download(blob_name, local_path)
                downloaded.append(local_path)
        return downloaded

    def list_blobs(self, prefix: str = "") -> list[str]:
        return [blob.name for blob in self._client.list_blobs(name_starts_with=prefix)]

    @classmethod
    def from_env(cls, container: str = "elephant-images") -> "ElephantBlobClient":
        account_url = os.environ.get("AZURE_BLOB_ACCOUNT_URL", "")
        if not account_url:
            logger.warning("AZURE_BLOB_ACCOUNT_URL env var is not set.")
        # Extract account name from URL: https://<account>.blob.core.windows.net
        account = account_url.split("//")[-1].split(".")[0] if account_url else "ganeshasfc2o4rujo76u"
        return cls(account=account, container=container)
