# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""Normalized artifact contracts for the shared elephant pipeline."""

import hashlib

import numpy as np
import pandas as pd

from configs.config_artifacts import ARTIFACT_SCHEMA_VERSION


CROP_MANIFEST_COLUMNS: list[str] = [
    "crop_id",
    "image_id",
    "individual_id",
    "crop_kind",
    "crop_ordinal",
    "crop_path",
    "detector_confidence",
    "detector_box",
    "detector_status",
    "review_status",
    "schema_version",
    "source_fingerprint",
    "split_fingerprint",
]

CROP_MANIFEST_DTYPES: dict[str, str] = {
    "crop_id": "string",
    "image_id": "string",
    "individual_id": "string",
    "crop_kind": "string",
    "crop_ordinal": "Int64",
    "crop_path": "string",
    "detector_confidence": "Float64",
    "detector_box": "string",
    "detector_status": "string",
    "review_status": "string",
    "schema_version": "string",
    "source_fingerprint": "string",
    "split_fingerprint": "string",
}

# Detector statuses that indicate a slot is finished and will not be retried.
# "failed" is intentionally absent — it remains retryable.
TERMINAL_CROP_STATUSES: frozenset[str] = frozenset(
    {"accepted", "none_detected", "not_applicable"}
)

# ---------------------------------------------------------------------------
# Experiment head region manifest — extends the base schema with two extra
# columns that carry source-tracking and detector-config provenance.
# These columns are NOT present in production selected-v1 crop manifests.
# ---------------------------------------------------------------------------

HEAD_EXPERIMENT_MANIFEST_COLUMNS: list[str] = CROP_MANIFEST_COLUMNS + [
    "source_used",
    "detector_fingerprint",
]

HEAD_EXPERIMENT_MANIFEST_DTYPES: dict[str, str] = {
    **CROP_MANIFEST_DTYPES,
    "source_used": "string",
    "detector_fingerprint": "string",
}

# ---------------------------------------------------------------------------
# Valid crop kinds and their allowed ordinals.
# v1 production artifacts use only "body" and "ear"; "head" is additive and
# backward-compatible — existing manifests remain valid.
# ---------------------------------------------------------------------------

_CROP_KIND_ORDINALS: dict[str, frozenset] = {
    "body": frozenset({0}),
    "ear": frozenset({0, 1}),
    "head": frozenset({0}),
}
VALID_CROP_KINDS: frozenset[str] = frozenset(_CROP_KIND_ORDINALS)

DESCRIPTOR_MAPPING_COLUMNS: list[str] = [
    "descriptor_name",
    "embedding_row",
    "faiss_row",
    "crop_id",
    "image_id",
    "individual_id",
    "crop_kind",
    "crop_ordinal",
    "crop_path",
    "schema_version",
    "source_fingerprint",
    "split_fingerprint",
    "model_preprocess_fingerprint",
]

DESCRIPTOR_MAPPING_DTYPES: dict[str, str] = {
    "descriptor_name": "string",
    "embedding_row": "Int64",
    "faiss_row": "Int64",
    "crop_id": "string",
    "image_id": "string",
    "individual_id": "string",
    "crop_kind": "string",
    "crop_ordinal": "Int64",
    "crop_path": "string",
    "schema_version": "string",
    "source_fingerprint": "string",
    "split_fingerprint": "string",
    "model_preprocess_fingerprint": "string",
}


def _require_columns(df: pd.DataFrame, columns: list[str], artifact: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{artifact} is missing required columns: {missing}")


def _assert_expected_value(
    df: pd.DataFrame, column: str, expected: str | None, artifact: str
) -> None:
    if expected is None or df.empty:
        return
    values = set(df[column].drop_duplicates().tolist())
    if values != {expected}:
        raise AssertionError(
            f"{artifact} {column} mismatch: expected {expected!r}, found {sorted(map(str, values))}"
        )


def assert_crop_manifest_integrity(
    crop_df: pd.DataFrame,
    image_manifest: pd.DataFrame,
    *,
    schema_version: str = ARTIFACT_SCHEMA_VERSION,
    expected_source_fingerprint: str | None = None,
    expected_split_fingerprint: str | None = None,
) -> None:
    """Fail loudly when a crop manifest violates the normalized contract."""
    _require_columns(crop_df, CROP_MANIFEST_COLUMNS, "crop manifest")
    _require_columns(image_manifest, ["image_id"], "image manifest")

    for column in ("image_id", "crop_id"):
        null_count = int(crop_df[column].isna().sum())
        if null_count:
            raise AssertionError(f"crop manifest has {null_count} null {column} values")

    duplicate_ids = crop_df.loc[crop_df["crop_id"].duplicated(keep=False), "crop_id"].tolist()
    if duplicate_ids:
        raise AssertionError(f"crop_id values must be unique; duplicates: {duplicate_ids}")

    known_ids = set(image_manifest["image_id"].dropna().astype(str))
    crop_image_ids = set(crop_df["image_id"].dropna().astype(str))
    unknown_ids = sorted(crop_image_ids - known_ids)
    if unknown_ids:
        raise AssertionError(f"crop manifest image_id values missing from image manifest: {unknown_ids}")

    invalid_kinds = sorted(set(crop_df["crop_kind"].dropna()) - VALID_CROP_KINDS)
    if invalid_kinds or crop_df["crop_kind"].isna().any():
        raise AssertionError(f"invalid crop_kind values: {invalid_kinds}")

    body = crop_df["crop_kind"] == "body"
    ear = crop_df["crop_kind"] == "ear"
    head = crop_df["crop_kind"] == "head"
    invalid_body_ordinals = sorted(set(crop_df.loc[body, "crop_ordinal"].dropna()) - {0})
    invalid_ear_ordinals = sorted(set(crop_df.loc[ear, "crop_ordinal"].dropna()) - {0, 1})
    invalid_head_ordinals = sorted(set(crop_df.loc[head, "crop_ordinal"].dropna()) - {0})
    null_ordinals = crop_df["crop_ordinal"].isna()
    if (
        invalid_body_ordinals
        or invalid_ear_ordinals
        or invalid_head_ordinals
        or null_ordinals.any()
    ):
        raise AssertionError(
            "invalid crop_ordinal values: "
            f"body={invalid_body_ordinals}, ear={invalid_ear_ordinals}, "
            f"head={invalid_head_ordinals}, "
            f"null_count={int(null_ordinals.sum())}"
        )

    accepted = crop_df["detector_status"] == "accepted"
    body_counts = crop_df.loc[accepted & body].groupby("image_id").size()
    excessive_bodies = body_counts[body_counts > 1].to_dict()
    if excessive_bodies:
        raise AssertionError(f"accepted body crop cardinality exceeds 1: {excessive_bodies}")

    ear_counts = crop_df.loc[accepted & ear].groupby("image_id").size()
    excessive_ears = ear_counts[ear_counts > 2].to_dict()
    if excessive_ears:
        raise AssertionError(f"accepted ear crop cardinality exceeds 2: {excessive_ears}")

    head_counts = crop_df.loc[accepted & head].groupby("image_id").size()
    excessive_heads = head_counts[head_counts > 1].to_dict()
    if excessive_heads:
        raise AssertionError(f"accepted head crop cardinality exceeds 1: {excessive_heads}")

    # Every accepted crop must carry a non-empty individual_id that matches
    # the image manifest entry for the same image_id.
    accepted_crops = crop_df.loc[accepted].copy()
    if not accepted_crops.empty:
        empty_ids = accepted_crops["individual_id"].isna() | accepted_crops["individual_id"].astype(str).str.strip().eq("")
        if empty_ids.any():
            bad = accepted_crops.loc[empty_ids, "crop_id"].tolist()
            raise AssertionError(
                f"accepted crops have empty individual_id: {bad}"
            )
        if "individual_id" in image_manifest.columns:
            manifest_id_map = (
                image_manifest.set_index(
                    image_manifest["image_id"].astype(str)
                )["individual_id"]
                .astype(str)
                .to_dict()
            )
            mismatches = []
            for _, row in accepted_crops.iterrows():
                manifest_identity = manifest_id_map.get(str(row["image_id"]))
                if manifest_identity is not None and str(row["individual_id"]) != manifest_identity:
                    mismatches.append(
                        (str(row["crop_id"]), str(row["individual_id"]), manifest_identity)
                    )
            if mismatches:
                raise AssertionError(
                    f"crop individual_id disagrees with image manifest for: {mismatches}"
                )

    _assert_expected_value(crop_df, "schema_version", schema_version, "crop manifest")
    _assert_expected_value(
        crop_df,
        "source_fingerprint",
        expected_source_fingerprint,
        "crop manifest",
    )
    _assert_expected_value(
        crop_df,
        "split_fingerprint",
        expected_split_fingerprint,
        "crop manifest",
    )


def assert_descriptor_mapping_integrity(
    mapping_df: pd.DataFrame,
    embedding_matrix: np.ndarray,
    faiss_index,
    *,
    is_reference: bool,
    schema_version: str = ARTIFACT_SCHEMA_VERSION,
    expected_source_fingerprint: str | None = None,
    expected_split_fingerprint: str | None = None,
    expected_model_fingerprint: str | None = None,
) -> None:
    """Validate one descriptor's mapping, matrix, and optional FAISS index."""
    _require_columns(mapping_df, DESCRIPTOR_MAPPING_COLUMNS, "descriptor mapping")
    if embedding_matrix.ndim != 2:
        raise ValueError(
            f"embedding_matrix must be 2-dimensional, got shape {embedding_matrix.shape}"
        )

    null_embedding_rows = int(mapping_df["embedding_row"].isna().sum())
    if null_embedding_rows:
        raise AssertionError(
            f"descriptor mapping has {null_embedding_rows} null embedding_row values"
        )

    n_rows = len(mapping_df)
    if n_rows != len(embedding_matrix):
        raise AssertionError(
            f"mapping/matrix row count mismatch: mapping={n_rows}, matrix={len(embedding_matrix)}"
        )
    if is_reference and faiss_index is not None and n_rows != int(faiss_index.ntotal):
        raise AssertionError(
            f"mapping/FAISS row count mismatch: mapping={n_rows}, faiss.ntotal={faiss_index.ntotal}"
        )

    rows = mapping_df["embedding_row"].astype(int).tolist()
    expected_rows = list(range(n_rows))
    if rows != expected_rows or len(set(rows)) != n_rows:
        raise AssertionError(
            f"embedding_row must be unique and contiguous in mapping order; "
            f"expected {expected_rows}, found {rows}"
        )

    if is_reference:
        if mapping_df["faiss_row"].isna().any():
            raise AssertionError("reference descriptor mapping contains null faiss_row values")
        faiss_rows = mapping_df["faiss_row"].astype(int).tolist()
        if faiss_rows != rows:
            raise AssertionError(
                f"faiss_row must equal embedding_row; embedding={rows}, faiss={faiss_rows}"
            )

    if not np.isfinite(embedding_matrix).all():
        bad_rows = np.where(~np.isfinite(embedding_matrix).all(axis=1))[0].tolist()
        raise AssertionError(f"embedding matrix contains non-finite vectors at rows {bad_rows}")
    norms = np.linalg.norm(embedding_matrix, axis=1)
    zero_rows = np.where(norms == 0)[0].tolist()
    if zero_rows:
        raise AssertionError(f"embedding matrix contains zero-norm vectors at rows {zero_rows}")
    non_normalized = np.where(~np.isclose(norms, 1.0, atol=1e-4, rtol=0.0))[0].tolist()
    if non_normalized:
        values = norms[non_normalized].tolist()
        raise AssertionError(
            f"embedding vectors are not L2-normalized at rows {non_normalized}: norms={values}"
        )

    duplicate_pairs = mapping_df.duplicated(["descriptor_name", "crop_id"], keep=False)
    if duplicate_pairs.any():
        pairs = mapping_df.loc[duplicate_pairs, ["descriptor_name", "crop_id"]].values.tolist()
        raise AssertionError(f"duplicate (descriptor_name, crop_id) pairs: {pairs}")

    identity_counts = mapping_df.groupby("crop_id", dropna=False)["individual_id"].nunique(
        dropna=False
    )
    disagreements = identity_counts[identity_counts > 1].to_dict()
    if disagreements:
        raise AssertionError(
            f"individual_id disagreement for crop_id values: {disagreements}"
        )

    _assert_expected_value(mapping_df, "schema_version", schema_version, "descriptor mapping")
    _assert_expected_value(
        mapping_df,
        "source_fingerprint",
        expected_source_fingerprint,
        "descriptor mapping",
    )
    _assert_expected_value(
        mapping_df,
        "split_fingerprint",
        expected_split_fingerprint,
        "descriptor mapping",
    )
    _assert_expected_value(
        mapping_df,
        "model_preprocess_fingerprint",
        expected_model_fingerprint,
        "descriptor mapping",
    )


def assert_no_cross_artifact_contamination(*mapping_dfs: pd.DataFrame) -> None:
    """Assert every descriptor mapping owns an independent contiguous row space."""
    for index, mapping_df in enumerate(mapping_dfs):
        _require_columns(mapping_df, ["embedding_row"], f"descriptor mapping {index}")
        rows = mapping_df["embedding_row"].tolist()
        expected = list(range(len(mapping_df)))
        if rows != expected:
            raise AssertionError(
                f"descriptor mapping {index} embedding_row must equal {expected}; found {rows}"
            )


def fingerprint_dataframe(df: pd.DataFrame, id_column: str) -> str:
    """Return a deterministic SHA-256 fingerprint of a DataFrame ID column."""
    if id_column not in df.columns:
        raise ValueError(f"fingerprint id column {id_column!r} is missing")
    if df[id_column].isna().any():
        raise ValueError(f"fingerprint id column {id_column!r} contains null values")
    payload = "\n".join(sorted(df[id_column].astype(str).tolist())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fingerprint_dataframe_columns(
    df: pd.DataFrame,
    columns: list[str],
    *,
    sort_by: list[str] | None = None,
) -> str:
    """Return a deterministic SHA-256 fingerprint over selected row values."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"fingerprint columns are missing: {missing}")
    ordered = df[columns].copy()
    ordered = ordered.sort_values(sort_by or columns, kind="mergesort")
    payload = ordered.fillna("<NULL>").astype(str).to_csv(
        index=False, header=False, lineterminator="\n"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_crop_id(image_id: str, crop_kind: str, crop_ordinal: int) -> str:
    """Construct ``{image_id}__{crop_kind}_{crop_ordinal}``."""
    if not isinstance(image_id, str) or not image_id:
        raise ValueError(f"image_id must be a non-empty string, got {image_id!r}")
    if crop_kind not in _CROP_KIND_ORDINALS:
        raise ValueError(
            f"crop_kind must be one of {sorted(_CROP_KIND_ORDINALS)}, got {crop_kind!r}"
        )
    allowed_ordinals = _CROP_KIND_ORDINALS[crop_kind]
    if crop_ordinal not in allowed_ordinals:
        raise ValueError(
            f"crop_ordinal for {crop_kind!r} must be one of {sorted(allowed_ordinals)}, "
            f"got {crop_ordinal!r}"
        )
    return f"{image_id}__{crop_kind}_{crop_ordinal}"


def assert_head_experiment_manifest_integrity(
    head_df: pd.DataFrame,
    image_manifest: pd.DataFrame,
    *,
    schema_version: str = ARTIFACT_SCHEMA_VERSION,
    expected_detector_fingerprint: str | None = None,
    expected_source_fingerprint: str | None = None,
    expected_split_fingerprint: str | None = None,
) -> None:
    """Validate an experiment head region manifest.

    Calls the base ``assert_crop_manifest_integrity`` and then adds
    head-specific rules:

    * Every row must have ``crop_kind == 'head'``.
    * ``source_used`` and ``detector_fingerprint`` columns must be present.
    * All accepted rows must agree on ``detector_fingerprint`` when an
      expected value is supplied.
    * Does not mutate or validate production selected-v1 artifacts.
    """
    # Verify head-specific columns exist before running the base check so the
    # error message is actionable.
    for extra_col in ("source_used", "detector_fingerprint"):
        if extra_col not in head_df.columns:
            raise AssertionError(
                f"head experiment manifest is missing required column {extra_col!r}"
            )

    # Base schema validation (validates crop_kind, ordinals, cardinality,
    # individual_id, schema_version, etc.).
    assert_crop_manifest_integrity(
        head_df,
        image_manifest,
        schema_version=schema_version,
    )

    # Every row must be crop_kind='head'.
    non_head = head_df[head_df["crop_kind"] != "head"]
    if not non_head.empty:
        raise AssertionError(
            f"head experiment manifest contains non-head rows: "
            f"{non_head['crop_id'].tolist()}"
        )

    # source_fingerprint / split_fingerprint agreement.
    if expected_source_fingerprint is not None and not head_df.empty:
        _assert_expected_value(
            head_df,
            "source_fingerprint",
            expected_source_fingerprint,
            "head experiment manifest",
        )
    if expected_split_fingerprint is not None and not head_df.empty:
        _assert_expected_value(
            head_df,
            "split_fingerprint",
            expected_split_fingerprint,
            "head experiment manifest",
        )

    # detector_fingerprint: accepted rows must agree.
    if expected_detector_fingerprint is not None and not head_df.empty:
        accepted = head_df[head_df["detector_status"] == "accepted"]
        if not accepted.empty:
            fps = accepted["detector_fingerprint"].dropna().unique().tolist()
            if len(fps) != 1 or fps[0] != expected_detector_fingerprint:
                raise AssertionError(
                    f"head experiment manifest detector_fingerprint mismatch: "
                    f"expected {expected_detector_fingerprint!r}, found {fps}"
                )
