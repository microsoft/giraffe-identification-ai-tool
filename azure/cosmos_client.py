# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import logging

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.cosmos import CosmosClient, exceptions as cosmos_exceptions

logger = logging.getLogger(__name__)

# Columns intentionally excluded:
#   - GPS coordinates  → poaching-sensitive
#   - embedding vectors → unverified provenance
_INDIVIDUAL_COLUMNS = ["individual_id", "name", "seek_id", "herd", "sex", "age", "markings"]
_IMAGE_COLUMNS = ["image_id", "individual_id", "blob_path"]


class GaneshaCosmosClient:
    def __init__(self, endpoint: str, database: str = "ganesha"):
        credential = DefaultAzureCredential()
        self._db_name = database
        try:
            self._client = CosmosClient(url=endpoint, credential=credential)
            self._db = self._client.get_database_client(database)
        except Exception as exc:
            logger.warning("Could not connect to Cosmos DB at %s: %s", endpoint, exc)
            self._db = None

    @classmethod
    def from_env(cls, database: str = "ganesha") -> "GaneshaCosmosClient":
        endpoint = os.environ.get("COSMOS_ENDPOINT", "")
        if not endpoint:
            logger.warning("COSMOS_ENDPOINT env var is not set; Cosmos client will return empty DataFrames.")
        return cls(endpoint=endpoint, database=database)

    def _query_container(self, container_name: str, query: str, columns: list[str]) -> pd.DataFrame:
        if self._db is None:
            return pd.DataFrame(columns=columns)
        try:
            container = self._db.get_container_client(container_name)
            items = list(container.query_items(query=query, enable_cross_partition_query=True))
            rows = [{col: item.get(col) for col in columns} for item in items]
            return pd.DataFrame(rows, columns=columns)
        except (cosmos_exceptions.CosmosHttpResponseError, Exception) as exc:
            logger.warning("Cosmos query on '%s' failed: %s — returning empty DataFrame.", container_name, exc)
            return pd.DataFrame(columns=columns)

    def fetch_individuals(self) -> pd.DataFrame:
        query = (
            "SELECT c.individual_id, c.name, c.seek_id, c.herd, c.sex, c.age, c.markings "
            "FROM c"
        )
        return self._query_container("individuals", query, _INDIVIDUAL_COLUMNS)

    def fetch_image_inventory(self) -> pd.DataFrame:
        query = (
            "SELECT c.image_id, c.individual_id, c.blob_path "
            "FROM c"
        )
        return self._query_container("images", query, _IMAGE_COLUMNS)
