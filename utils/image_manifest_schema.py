# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""Canonical source-adapter contract for normalized elephant image manifests."""

import hashlib

import pandas as pd


IMAGE_MANIFEST_COLUMNS: list[str] = [
    "image_id",
    "individual_id",
    "individual_name",
    "herd",
    "source_relative_path",
    "content_hash",
    "perceptual_hash",
    "image_id_path_component",
    "image_id_content_component",
    "session_id",
    "capture_date",
    "year",
    "session_source",
    "dataset_role",
    "include_status",
    "exclusion_reason",
    "duplicate_of",
    "review_flag",
    "review_reason",
    "body_crop_status",
    "ear_detection_status",
    "image_width",
    "image_height",
]


def fingerprint_image_manifest(manifest: pd.DataFrame) -> str:
    """Hash all canonical and source-specific metadata deterministically."""
    missing = set(IMAGE_MANIFEST_COLUMNS) - set(manifest.columns)
    if missing:
        raise ValueError(
            f"image manifest is missing canonical columns: {sorted(missing)}"
        )
    ordered = manifest.reindex(sorted(manifest.columns), axis=1).sort_values(
        "image_id",
        kind="stable",
    )
    payload = ordered.to_json(
        orient="records",
        date_format="iso",
        date_unit="us",
        force_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
