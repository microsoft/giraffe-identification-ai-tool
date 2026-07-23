# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""Data roots and artifact names for the ELPephants benchmark dataset."""

import os
from pathlib import Path

from dotenv import load_dotenv
from configs.config_artifacts import ARTIFACT_SCHEMA_VERSION

load_dotenv()

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_source_default = str(REPO_ROOT.parent / "shared" / "ELPephants")
ELPEPHANTS_SOURCE_ROOT: Path = Path(
    os.getenv("ELPEPHANTS_SOURCE_ROOT", _source_default)
)

_artifact_default = str(REPO_ROOT.parent / "shared" / "ELPephants_reid_artifacts")
ELPEPHANTS_ARTIFACT_ROOT: Path = Path(
    os.getenv("ELPEPHANTS_ARTIFACT_ROOT", _artifact_default)
)

ARTIFACT_VERSION_ROOT: Path = (
    ELPEPHANTS_ARTIFACT_ROOT / ARTIFACT_SCHEMA_VERSION
)

MANIFEST_SUBDIR: str = "manifests"
SPLITS_SUBDIR: str = "splits"
CROPS_SUBDIR: str = "crops"
EMBEDDINGS_SUBDIR: str = "embeddings"
FAISS_SUBDIR: str = "faiss_index"
CALIBRATION_SUBDIR: str = "calibration"
REPORTS_SUBDIR: str = "reports"

MANIFEST_FILENAME: str = "elpephants_image_manifest.parquet"
SPLITS_FILENAME: str = "elpephants_splits.parquet"


def canonical_individual_id(source_class_id: str) -> str:
    """Return a namespaced ID while preserving significant leading zeroes."""
    class_id = source_class_id.strip()
    if not class_id or not class_id.isdigit():
        raise ValueError(f"invalid ELPephants source class ID: {source_class_id!r}")
    return f"elpephants_{class_id}"
