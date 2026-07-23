#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
BTEH crop-quality pilot: stratified sampling, contact-sheet reports, and
machine-readable crop-quality metrics.

Two sub-commands:
  sample  – select a stratified pilot sample from the canonical manifests
            and write a pilot image manifest parquet + JSON sidecar.
  report  – consume the pilot manifest + normalized crop manifest and
            produce contact sheets plus JSON/CSV/Markdown metric reports.

Usage
-----
    python pipeline/bteh_crop_pilot.py sample \\
        [--manifest PATH]   \\
        [--splits   PATH]   \\
        [--output-dir PATH] \\
        [--n-pilot  N]      \\   # default 120
        [--n-review N]      \\   # default 12 unresolved/review audit images
        [--seed     N]          # default 42

    python pipeline/bteh_crop_pilot.py report \\
        [--pilot-manifest  PATH]       \\
        [--crop-manifest   PATH]       \\
        [--source-root     PATH]       \\
        [--output-dir      PATH]       \\
        [--review-csv      PATH]       \\   # optional human-review CSV
        [--page-size       N]              # images per contact-sheet page (default 12)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_bteh import (
    ARTIFACT_SCHEMA_VERSION,
    BTEH_ARTIFACT_ROOT,
    BTEH_SOURCE_ROOT,
    CONTACT_SHEETS_SUBDIR,
    MANIFEST_FILENAME,
    MANIFEST_SUBDIR,
    REPORTS_SUBDIR,
    SPLITS_FILENAME,
    SPLITS_SUBDIR,
)
from utils.artifact_schema import (
    CROP_MANIFEST_COLUMNS,
    TERMINAL_CROP_STATUSES,
    fingerprint_dataframe,
    fingerprint_dataframe_columns,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PILOT_N: int = 120
DEFAULT_REVIEW_N: int = 12
DEFAULT_SEED: int = 42
DEFAULT_PAGE_SIZE: int = 12

PILOT_MANIFEST_FILENAME: str = "bteh_pilot_manifest.parquet"
PILOT_SIDECAR_FILENAME: str = "bteh_pilot_manifest.json"
PILOT_CROP_REPORT_JSON: str = "bteh_pilot_crop_report.json"
PILOT_CROP_REPORT_CSV: str = "bteh_pilot_crop_report.csv"
PILOT_CROP_REPORT_MD: str = "bteh_pilot_crop_report.md"

# Strata definitions
_SIZE_BUCKETS = [
    ("small", lambda w, h: (w * h) < 500_000),
    ("medium", lambda w, h: 500_000 <= (w * h) < 2_000_000),
    ("large", lambda w, h: (w * h) >= 2_000_000),
]

_ASPECT_BUCKETS = [
    ("landscape", lambda w, h: w > h * 1.1),
    ("portrait", lambda w, h: h > w * 1.1),
    ("square", lambda w, h: True),  # catch-all
]


# ---------------------------------------------------------------------------
# Strata helpers
# ---------------------------------------------------------------------------

def _size_bucket(width: Any, height: Any) -> str:
    try:
        w, h = int(width), int(height)
        for name, fn in _SIZE_BUCKETS:
            if fn(w, h):
                return name
    except (TypeError, ValueError):
        pass
    return "unknown"


def _aspect_bucket(width: Any, height: Any) -> str:
    try:
        w, h = int(width), int(height)
        for name, fn in _ASPECT_BUCKETS:
            if fn(w, h):
                return name
    except (TypeError, ValueError):
        pass
    return "unknown"


def _origin_bucket(dataset_role: Any) -> str:
    if str(dataset_role).lower() == "ref":
        return "ref"
    return "regular"


def _year_bucket(year: Any) -> str:
    if pd.isna(year) or not str(year).strip():
        return "unknown"
    return str(year).strip()


def _session_source_bucket(session_source: Any) -> str:
    if pd.isna(session_source) or not str(session_source).strip():
        return "unknown"
    return str(session_source).strip()


def _split_bucket(split: Any) -> str:
    if pd.isna(split) or not str(split).strip():
        return "no_split"
    return str(split).strip()


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------

def _fingerprint_manifest(df: pd.DataFrame) -> str:
    """SHA-256 over sorted image_ids."""
    return fingerprint_dataframe(df, "image_id")


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def _assign_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Add composite stratum column to a copy of *df*."""
    df = df.copy()
    df["_size_bucket"] = df.apply(
        lambda r: _size_bucket(r.get("image_width"), r.get("image_height")), axis=1
    )
    df["_aspect_bucket"] = df.apply(
        lambda r: _aspect_bucket(r.get("image_width"), r.get("image_height")), axis=1
    )
    df["_origin"] = df["dataset_role"].apply(_origin_bucket)
    df["_year"] = df["year"].apply(_year_bucket)
    df["_session_src"] = df["session_source"].apply(_session_source_bucket)
    split_col = "split" if "split" in df.columns else None
    if split_col:
        df["_split"] = df[split_col].apply(_split_bucket)
    else:
        df["_split"] = "no_split"

    # Composite stratum: individual + split + year + session_source + size + aspect + origin
    df["_stratum"] = (
        df["individual_id"].fillna("unknown").astype(str)
        + "|"
        + df["_split"].astype(str)
        + "|"
        + df["_year"].astype(str)
        + "|"
        + df["_session_src"].astype(str)
        + "|"
        + df["_size_bucket"].astype(str)
        + "|"
        + df["_aspect_bucket"].astype(str)
        + "|"
        + df["_origin"].astype(str)
    )
    return df


def _stratified_sample(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Stratified sample of *n* rows from *df* using column ``_stratum``.

    Within each stratum, one representative is selected first; remaining
    quota is filled by sampling from strata with more candidates, always
    preferring the least-represented stratum so small strata are included.
    """
    if df.empty or n <= 0:
        return df.iloc[:0].copy()

    strata = df.groupby("_stratum", sort=False)
    strata_keys = list(strata.groups.keys())
    # shuffle strata order deterministically
    strata_keys = [strata_keys[i] for i in rng.permutation(len(strata_keys))]

    selected_idx: list[int] = []
    strata_remaining: dict[str, list[int]] = {
        k: rng.permutation(strata.groups[k]).tolist() for k in strata_keys
    }

    # First pass: one from each stratum
    for key in strata_keys:
        if len(selected_idx) >= n:
            break
        candidates = strata_remaining[key]
        if candidates:
            selected_idx.append(candidates.pop(0))

    # Second pass: fill quota round-robin, sorted by stratum size (smallest first)
    remaining_strata = sorted(
        [k for k in strata_keys if strata_remaining[k]],
        key=lambda k: len(strata_remaining[k]),
    )
    round_robin = list(remaining_strata)
    while len(selected_idx) < n and round_robin:
        next_round = []
        for key in round_robin:
            if len(selected_idx) >= n:
                break
            candidates = strata_remaining[key]
            if candidates:
                selected_idx.append(candidates.pop(0))
                if candidates:
                    next_round.append(key)
        round_robin = next_round

    return df.loc[selected_idx].copy()


def select_pilot_sample(
    manifest: pd.DataFrame,
    splits: pd.DataFrame | None = None,
    *,
    n_pilot: int = DEFAULT_PILOT_N,
    n_review: int = DEFAULT_REVIEW_N,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Select a stratified pilot sample from the canonical manifest.

    Parameters
    ----------
    manifest:  canonical image manifest (from bteh_manifest.py)
    splits:    optional splits DataFrame (from bteh_splits.py)
    n_pilot:   target size of named-identity crop-quality sample
    n_review:  target size of unresolved/review audit sample
    seed:      random seed for deterministic selection

    Returns
    -------
    (pilot_df, review_df, strata_report)
      pilot_df      – pilot sample parquet rows, eligible named rows only
      review_df     – small audit sample of unresolved/review_required rows
      strata_report – dict of strata counts and other metadata
    """
    rng = np.random.default_rng(seed)

    # --- Eligible named rows: included + duplicate_primary, non-unresolved ---
    eligible_mask = (
        manifest["include_status"].isin({"included", "duplicate_primary"})
        & manifest["individual_id"].notna()
        & (manifest["individual_id"].astype(str) != "unresolved")
    )
    eligible = manifest[eligible_mask].copy()

    if eligible.empty:
        raise ValueError(
            "No eligible named rows in manifest "
            "(include_status ∈ {included, duplicate_primary} and individual_id != unresolved)."
        )

    # Attach splits if provided
    if splits is not None:
        split_cols = ["image_id", "split", "split_protocol", "evaluable", "fold"]
        available = [c for c in split_cols if c in splits.columns]
        splits_sub = splits[available].rename(
            columns={c: c for c in available}
        )
        # Use split from splits df, not from manifest (may be stale)
        for col in ("split", "split_protocol", "evaluable", "fold"):
            if col in eligible.columns:
                eligible = eligible.drop(columns=[col])
        eligible = eligible.merge(
            splits_sub, on="image_id", how="left"
        )

    eligible = _assign_strata(eligible)

    # --- Named pilot sample ---
    pilot_df = _stratified_sample(eligible, n_pilot, rng)

    # --- Unresolved / review_required audit sample ---
    review_mask = (
        manifest["include_status"].isin({"review_required"})
        | (
            manifest["include_status"].isin({"included", "duplicate_primary"})
            & (manifest.get("review_flag", pd.Series(False, index=manifest.index)) == True)  # noqa: E712
        )
    )
    review_pool = manifest[review_mask].copy()
    if not review_pool.empty and n_review > 0:
        n_take = min(n_review, len(review_pool))
        review_idx = rng.choice(len(review_pool), size=n_take, replace=False)
        review_df = review_pool.iloc[review_idx].copy()
    else:
        review_df = review_pool.iloc[:0].copy()

    # --- Strata report ---
    strata_cols = ["_stratum", "_split", "_year", "_session_src", "_size_bucket",
                   "_aspect_bucket", "_origin", "individual_id"]
    strata_report = {
        "n_pilot_requested": n_pilot,
        "n_pilot_selected": len(pilot_df),
        "n_review_requested": n_review,
        "n_review_selected": len(review_df),
        "n_eligible": len(eligible),
        "n_identities": int(pilot_df["individual_id"].nunique()),
        "strata_counts": (
            pilot_df["_stratum"].value_counts().to_dict()
            if not pilot_df.empty else {}
        ),
        "split_counts": (
            pilot_df["_split"].value_counts().to_dict()
            if "_split" in pilot_df.columns and not pilot_df.empty else {}
        ),
        "year_counts": (
            pilot_df["_year"].value_counts().to_dict()
            if "_year" in pilot_df.columns and not pilot_df.empty else {}
        ),
        "origin_counts": (
            pilot_df["_origin"].value_counts().to_dict()
            if "_origin" in pilot_df.columns and not pilot_df.empty else {}
        ),
        "size_bucket_counts": (
            pilot_df["_size_bucket"].value_counts().to_dict()
            if "_size_bucket" in pilot_df.columns and not pilot_df.empty else {}
        ),
        "aspect_bucket_counts": (
            pilot_df["_aspect_bucket"].value_counts().to_dict()
            if "_aspect_bucket" in pilot_df.columns and not pilot_df.empty else {}
        ),
        "selected_image_ids": sorted(pilot_df["image_id"].tolist()),
    }
    return pilot_df, review_df, strata_report


# ---------------------------------------------------------------------------
# Pilot manifest I/O
# ---------------------------------------------------------------------------

_PILOT_INTERNAL_COLS = [
    "_stratum", "_split", "_year", "_session_src",
    "_size_bucket", "_aspect_bucket", "_origin",
]


def write_pilot_manifest(
    pilot_df: pd.DataFrame,
    review_df: pd.DataFrame,
    strata_report: dict,
    output_dir: Path,
    *,
    source_fingerprint: str | None,
    split_fingerprint: str | None,
    seed: int,
) -> tuple[Path, Path]:
    """
    Write pilot_manifest.parquet and pilot_manifest.json sidecar.

    The parquet is compatible with run_bteh_detection (retains all
    canonical manifest columns; internal stratum columns are dropped).

    Returns (parquet_path, sidecar_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Drop internal-only columns before writing
    keep_pilot = pilot_df.drop(
        columns=[c for c in _PILOT_INTERNAL_COLS if c in pilot_df.columns],
        errors="ignore",
    )
    # Mark review rows so downstream can distinguish them
    keep_review = review_df.copy()
    keep_review["_pilot_role"] = "review"
    keep_pilot["_pilot_role"] = "pilot"

    combined = pd.concat([keep_pilot, keep_review], ignore_index=True)

    parquet_path = output_dir / PILOT_MANIFEST_FILENAME
    combined.to_parquet(parquet_path, index=False)
    logger.info("Pilot manifest written: %s (%d rows)", parquet_path, len(combined))

    pilot_fingerprint = fingerprint_dataframe(keep_pilot, "image_id") if not keep_pilot.empty else ""

    sidecar: dict = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_fingerprint": source_fingerprint,
        "split_fingerprint": split_fingerprint,
        "pilot_fingerprint": pilot_fingerprint,
        "seed": seed,
        "n_pilot": len(keep_pilot),
        "n_review": len(keep_review),
        "strata_counts": strata_report.get("strata_counts", {}),
        "split_counts": strata_report.get("split_counts", {}),
        "year_counts": strata_report.get("year_counts", {}),
        "origin_counts": strata_report.get("origin_counts", {}),
        "size_bucket_counts": strata_report.get("size_bucket_counts", {}),
        "aspect_bucket_counts": strata_report.get("aspect_bucket_counts", {}),
        "n_identities": strata_report.get("n_identities", 0),
        "selected_image_ids": sorted(keep_pilot["image_id"].tolist()),
    }
    sidecar_path = output_dir / PILOT_SIDECAR_FILENAME
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    logger.info("Pilot sidecar written: %s", sidecar_path)

    return parquet_path, sidecar_path


# ---------------------------------------------------------------------------
# Schema / fingerprint validation
# ---------------------------------------------------------------------------

def _fail_loud_schema(crop_df: pd.DataFrame) -> None:
    """Raise ValueError when crop manifest is missing required columns."""
    missing = [c for c in CROP_MANIFEST_COLUMNS if c not in crop_df.columns]
    if missing:
        raise ValueError(
            f"Crop manifest is missing required columns: {missing}"
        )


def _fail_loud_fingerprint(
    pilot_df: pd.DataFrame,
    sidecar: dict,
) -> None:
    """Raise ValueError when the pilot manifest does not match its sidecar fingerprint."""
    pilot_only = pilot_df[pilot_df.get("_pilot_role", pd.Series("pilot", index=pilot_df.index)) == "pilot"]
    if pilot_only.empty and not sidecar.get("selected_image_ids"):
        return
    expected_fp = sidecar.get("pilot_fingerprint", "")
    if not expected_fp:
        return  # sidecar was built before fingerprinting was added; skip
    computed_fp = fingerprint_dataframe(pilot_only.rename(columns={}) if "image_id" in pilot_only.columns else pilot_df, "image_id")
    if computed_fp != expected_fp:
        raise ValueError(
            f"Pilot manifest fingerprint mismatch: "
            f"sidecar={expected_fp!r}, computed={computed_fp!r}. "
            "Re-run 'sample' to regenerate the pilot manifest."
        )


def _fail_loud_crop_fingerprints(crop_df: pd.DataFrame, sidecar: dict) -> None:
    """Require crop artifacts to match the pilot's source and split manifests."""
    for column, sidecar_key in (
        ("source_fingerprint", "source_fingerprint"),
        ("split_fingerprint", "split_fingerprint"),
    ):
        expected = sidecar.get(sidecar_key)
        if not expected:
            continue
        actual = set(crop_df[column].dropna().astype(str).unique())
        if actual != {str(expected)}:
            raise ValueError(
                f"Crop manifest {column} mismatch: expected {expected!r}, "
                f"found {sorted(actual)!r}."
            )


def _fail_loud_joins(
    pilot_df: pd.DataFrame,
    crop_df: pd.DataFrame,
) -> None:
    """
    Raise ValueError for any missing image/crop joins.

    Every pilot image_id must appear in the crop manifest.
    Accepted crops must have an existing file (crop_path must be non-null and
    the file must exist when running in a full-data context).
    """
    pilot_ids = set(pilot_df["image_id"].dropna().astype(str))
    crop_ids_in_manifest = set(crop_df["image_id"].dropna().astype(str))
    missing = sorted(pilot_ids - crop_ids_in_manifest)
    if missing:
        raise ValueError(
            f"{len(missing)} pilot image_id(s) have no entry in the crop manifest: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    # Accepted crops must have a crop_path
    accepted = crop_df[
        (crop_df["image_id"].astype(str).isin(pilot_ids))
        & (crop_df["detector_status"] == "accepted")
    ]
    null_paths = accepted[accepted["crop_path"].isna() | accepted["crop_path"].astype(str).str.strip().eq("")]
    if not null_paths.empty:
        bad_ids = null_paths["crop_id"].tolist()
        raise ValueError(
            f"{len(bad_ids)} accepted crop(s) have null/empty crop_path: {bad_ids[:10]}"
        )


def _fail_loud_accepted_files_exist(
    crop_df: pd.DataFrame,
    pilot_ids: set[str],
) -> None:
    """Raise ValueError when accepted crop files are missing from disk."""
    accepted = crop_df[
        (crop_df["image_id"].astype(str).isin(pilot_ids))
        & (crop_df["detector_status"] == "accepted")
        & crop_df["crop_path"].notna()
    ]
    missing_files = []
    for _, row in accepted.iterrows():
        p = Path(str(row["crop_path"]))
        if not p.exists():
            missing_files.append(str(row["crop_id"]))
    if missing_files:
        raise ValueError(
            f"{len(missing_files)} accepted crop file(s) are missing from disk: "
            f"{missing_files[:10]}{'...' if len(missing_files) > 10 else ''}"
        )


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_crop_metrics(
    pilot_df: pd.DataFrame,
    crop_df: pd.DataFrame,
    *,
    review_csv_path: Path | None = None,
) -> dict:
    """
    Compute machine-readable crop quality metrics for the pilot sample.

    Separate detector coverage from human acceptance:
    - ``detector_*``   : based solely on detector_status values
    - ``human_*``      : based on human review CSV (if provided)

    Parameters
    ----------
    pilot_df       : pilot image manifest (output of select_pilot_sample)
    crop_df        : normalized crop manifest (from run_bteh_detection)
    review_csv_path: optional path to a human-review CSV with columns
                     ``crop_id`` and ``status`` (accepted/rejected/uncertain)

    Returns
    -------
    Nested dict of metrics.
    """
    pilot_ids = set(pilot_df["image_id"].dropna().astype(str))
    pilot_crops = crop_df[crop_df["image_id"].astype(str).isin(pilot_ids)].copy()

    body = pilot_crops[pilot_crops["crop_kind"] == "body"]
    ear = pilot_crops[pilot_crops["crop_kind"] == "ear"]

    # Per-image metrics
    n_images = len(pilot_ids)

    body_accepted_ids = set(
        body.loc[body["detector_status"] == "accepted", "image_id"].astype(str)
    )
    ear_accepted = ear[ear["detector_status"] == "accepted"]
    ear_accepted_per_image = ear_accepted.groupby("image_id")["crop_ordinal"].count()
    ear1_ids = set(ear_accepted_per_image[ear_accepted_per_image >= 1].index.astype(str))
    ear2_ids = set(ear_accepted_per_image[ear_accepted_per_image >= 2].index.astype(str))

    # Terminal failure counts
    def _terminal_counts(sub: pd.DataFrame) -> dict:
        counts = sub["detector_status"].value_counts().to_dict()
        return {k: int(v) for k, v in counts.items()}

    status_breakdown = {
        "body": _terminal_counts(body),
        "ear": _terminal_counts(ear),
    }

    detector_metrics = {
        "n_images": n_images,
        "body_accepted_coverage": round(len(body_accepted_ids) / n_images, 4) if n_images else 0.0,
        "image_ge1_ear_coverage": round(len(ear1_ids) / n_images, 4) if n_images else 0.0,
        "image_2ear_coverage": round(len(ear2_ids) / n_images, 4) if n_images else 0.0,
        "accepted_ear_count": int(len(ear_accepted)),
        "status_breakdown": status_breakdown,
    }

    # Breakdown by strata
    strata_metrics = _compute_strata_metrics(pilot_df, pilot_crops)

    # Human review metrics
    human_metrics = _compute_human_metrics(pilot_crops, review_csv_path)

    return {
        "detector": detector_metrics,
        "human": human_metrics,
        "by_stratum": strata_metrics,
    }


def _compute_strata_metrics(
    pilot_df: pd.DataFrame,
    pilot_crops: pd.DataFrame,
) -> dict:
    """Breakdown by year, session_source, dataset_role (origin)."""
    result: dict = {}
    pilot_with_meta = pilot_df.copy()
    pilot_with_meta["image_id_str"] = pilot_with_meta["image_id"].astype(str)

    for dim, col in [
        ("year", "year"),
        ("session_source", "session_source"),
        ("origin", "dataset_role"),
    ]:
        if col not in pilot_with_meta.columns:
            continue
        dim_counts: dict = {}
        for val, grp in pilot_with_meta.groupby(
            pilot_with_meta[col].fillna("unknown").astype(str)
        ):
            ids = set(grp["image_id_str"].tolist())
            crops_sub = pilot_crops[pilot_crops["image_id"].astype(str).isin(ids)]
            body_sub = crops_sub[crops_sub["crop_kind"] == "body"]
            ear_sub = crops_sub[crops_sub["crop_kind"] == "ear"]
            body_acc = int((body_sub["detector_status"] == "accepted").sum())
            ear_acc = int((ear_sub["detector_status"] == "accepted").sum())
            dim_counts[str(val)] = {
                "n_images": len(ids),
                "body_accepted": body_acc,
                "ear_accepted": ear_acc,
            }
        result[dim] = dim_counts

    return result


def _compute_human_metrics(
    pilot_crops: pd.DataFrame,
    review_csv_path: Path | None,
) -> dict:
    """Parse optional human review CSV and compute accepted-crop precision."""
    if review_csv_path is None:
        return {
            "precision": None,
            "note": "No human review CSV provided. Precision unavailable.",
        }

    review_path = Path(review_csv_path)
    if not review_path.exists():
        raise FileNotFoundError(f"Human review CSV not found: {review_path}")

    review_df = pd.read_csv(review_path)
    required_cols = {"crop_id", "status"}
    missing = required_cols - set(review_df.columns)
    if missing:
        raise ValueError(
            f"Human review CSV is missing required columns: {sorted(missing)}"
        )

    valid_statuses = {"accepted", "rejected", "uncertain"}
    bad_statuses = set(review_df["status"].dropna().unique()) - valid_statuses
    if bad_statuses:
        raise ValueError(
            f"Human review CSV contains invalid status values: {bad_statuses}. "
            f"Must be one of: {sorted(valid_statuses)}"
        )

    crop_ids_in_pilot = set(
        pilot_crops.loc[
            pilot_crops["detector_status"] == "accepted", "crop_id"
        ].astype(str)
    )
    review_for_pilot = review_df[review_df["crop_id"].astype(str).isin(crop_ids_in_pilot)]

    accepted = int((review_for_pilot["status"] == "accepted").sum())
    rejected = int((review_for_pilot["status"] == "rejected").sum())
    uncertain = int((review_for_pilot["status"] == "uncertain").sum())
    total_decisive = accepted + rejected

    precision = round(accepted / total_decisive, 4) if total_decisive > 0 else None

    # Rejection reasons: from a 'reason' column if present, otherwise N/A
    rejection_reasons: dict = {}
    if "reason" in review_for_pilot.columns:
        rejected_rows = review_for_pilot[review_for_pilot["status"] == "rejected"]
        if not rejected_rows.empty:
            rejection_reasons = (
                rejected_rows["reason"]
                .fillna("unspecified")
                .value_counts()
                .to_dict()
            )
            rejection_reasons = {k: int(v) for k, v in rejection_reasons.items()}

    return {
        "n_reviewed": len(review_for_pilot),
        "n_accepted": accepted,
        "n_rejected": rejected,
        "n_uncertain": uncertain,
        "precision": precision,
        "rejection_reasons": rejection_reasons,
        "note": (
            "Precision = accepted / (accepted + rejected); uncertain crops excluded."
            if precision is not None
            else "No decisive reviews; precision unavailable."
        ),
    }


# ---------------------------------------------------------------------------
# Report writers: JSON / CSV / Markdown
# ---------------------------------------------------------------------------

def write_metric_reports(
    metrics: dict,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write JSON, CSV, and Markdown metric reports. Returns (json, csv, md) paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / PILOT_CROP_REPORT_JSON
    json_path.write_text(json.dumps(metrics, indent=2))
    logger.info("JSON report: %s", json_path)

    # CSV — flatten detector metrics into per-row format
    csv_path = output_dir / PILOT_CROP_REPORT_CSV
    det = metrics.get("detector", {})
    hum = metrics.get("human", {})
    rows_csv = [
        {
            "metric": "n_images",
            "value": det.get("n_images", ""),
            "note": "",
        },
        {
            "metric": "body_accepted_coverage",
            "value": det.get("body_accepted_coverage", ""),
            "note": "detector only",
        },
        {
            "metric": "image_ge1_ear_coverage",
            "value": det.get("image_ge1_ear_coverage", ""),
            "note": "detector only",
        },
        {
            "metric": "image_2ear_coverage",
            "value": det.get("image_2ear_coverage", ""),
            "note": "detector only",
        },
        {
            "metric": "accepted_ear_count",
            "value": det.get("accepted_ear_count", ""),
            "note": "detector only",
        },
        {
            "metric": "human_precision",
            "value": hum.get("precision", "N/A"),
            "note": hum.get("note", ""),
        },
    ]
    # Status breakdown rows
    for kind, counts in det.get("status_breakdown", {}).items():
        for status, count in counts.items():
            rows_csv.append({
                "metric": f"{kind}_status_{status}",
                "value": count,
                "note": "detector only",
            })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows_csv)
    logger.info("CSV report: %s", csv_path)

    # Markdown
    md_path = output_dir / PILOT_CROP_REPORT_MD
    md_lines = [
        "# BTEH Crop Pilot Quality Report",
        "",
        "## Detector Metrics",
        "",
        f"- Images in pilot: **{det.get('n_images', 'N/A')}**",
        f"- Body accepted coverage: **{det.get('body_accepted_coverage', 'N/A')}** (detector)",
        f"- ≥1 ear coverage: **{det.get('image_ge1_ear_coverage', 'N/A')}** (detector)",
        f"- 2-ear coverage: **{det.get('image_2ear_coverage', 'N/A')}** (detector)",
        f"- Accepted ear crops: **{det.get('accepted_ear_count', 'N/A')}** (detector)",
        "",
        "### Detector Status Breakdown",
        "",
        "| Kind | Status | Count |",
        "| ---- | ------ | ----- |",
    ]
    for kind, counts in det.get("status_breakdown", {}).items():
        for status, count in sorted(counts.items()):
            md_lines.append(f"| {kind} | {status} | {count} |")

    md_lines += [
        "",
        "## Human Review",
        "",
        f"- Reviewed: {hum.get('n_reviewed', 'N/A')}",
        f"- Accepted: {hum.get('n_accepted', 'N/A')}",
        f"- Rejected: {hum.get('n_rejected', 'N/A')}",
        f"- Uncertain: {hum.get('n_uncertain', 'N/A')}",
        f"- Precision: **{hum.get('precision', 'N/A')}**",
        f"- Note: _{hum.get('note', '')}_",
        "",
        "## Breakdown by Stratum",
        "",
    ]
    for dim, dim_data in metrics.get("by_stratum", {}).items():
        md_lines.append(f"### {dim.title()}")
        md_lines.append("")
        md_lines.append("| Value | Images | Body Accepted | Ear Accepted |")
        md_lines.append("| ----- | ------ | ------------- | ------------ |")
        for val, stats in sorted(dim_data.items()):
            md_lines.append(
                f"| {val} | {stats.get('n_images', 0)} | "
                f"{stats.get('body_accepted', 0)} | {stats.get('ear_accepted', 0)} |"
            )
        md_lines.append("")

    md_path.write_text("\n".join(md_lines))
    logger.info("Markdown report: %s", md_path)

    return json_path, csv_path, md_path


# ---------------------------------------------------------------------------
# Contact-sheet generation
# ---------------------------------------------------------------------------

_THUMB_SIZE = (192, 192)
_LABEL_HEIGHT = 28
_MARGIN = 8
_PAGE_COLS = 4  # images per row on a contact sheet page

_PLACEHOLDER_COLORS = {
    "none_detected": (200, 200, 200),
    "not_applicable": (180, 180, 220),
    "failed": (255, 180, 180),
    "pending": (255, 240, 180),
}


def _load_thumb(path: str | Path | None) -> Image.Image | None:
    """Load an image thumbnail or return None."""
    if path is None or str(path).strip() == "":
        return None
    try:
        img = Image.open(str(path)).convert("RGB")
        img.thumbnail(_THUMB_SIZE, Image.LANCZOS)
        return img
    except Exception:
        return None


def _placeholder_image(status: str, label: str = "") -> Image.Image:
    """Return a small grey placeholder image with a status label."""
    color = _PLACEHOLDER_COLORS.get(str(status).lower(), (220, 220, 220))
    img = Image.new("RGB", _THUMB_SIZE, color=color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    text = f"{status}\n{label}" if label else str(status)
    draw.multiline_text((8, _THUMB_SIZE[1] // 2 - 16), text, fill=(60, 60, 60), font=font)
    return img


def _strip_absolute_prefix(path_str: str, source_root: str | None) -> str:
    """Return a source-relative path, stripping absolute prefix."""
    if not path_str:
        return path_str
    if source_root:
        try:
            rel = Path(path_str).relative_to(source_root)
            return str(rel)
        except ValueError:
            pass
    # Fallback: strip common user-home prefixes.
    import re
    path_str = re.sub(r"^/" r"home/[^/]+/[^/]+/", ".../", path_str)
    return path_str


def _make_cell(
    source_path: str | None,
    body_crop_row: pd.Series | None,
    ear_rows: list[pd.Series],
    image_id: str,
    source_root: str | None,
) -> Image.Image:
    """
    Build a single contact-sheet cell: source image | body | ear_0 | ear_1.
    """
    thumb_w, thumb_h = _THUMB_SIZE
    n_slots = 4  # source + body + ear_0 + ear_1
    cell_w = n_slots * (thumb_w + _MARGIN) + _MARGIN
    cell_h = thumb_h + _LABEL_HEIGHT * 2 + _MARGIN * 2

    cell = Image.new("RGB", (cell_w, cell_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(cell)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    def _paste_at(img: Image.Image, slot: int, label: str) -> None:
        x = _MARGIN + slot * (thumb_w + _MARGIN)
        y = _MARGIN + _LABEL_HEIGHT
        # Resize to exactly _THUMB_SIZE to keep grid uniform
        img_resized = img.resize(_THUMB_SIZE, Image.LANCZOS)
        cell.paste(img_resized, (x, y))
        short_lbl = label[:28] + "…" if len(label) > 28 else label
        draw.text((x, _MARGIN), short_lbl, fill=(40, 40, 40), font=font)

    # Slot 0: source image (use source_relative_path for label, not absolute)
    src_label = _strip_absolute_prefix(str(source_path or ""), source_root)
    src_label = f"src:{Path(src_label).name}" if src_label else f"id:{image_id[:12]}"
    src_img = _load_thumb(source_path)
    if src_img is None:
        src_img = _placeholder_image("no_source", src_label)
    _paste_at(src_img, 0, src_label)

    # Slot 1: body crop
    if body_crop_row is not None:
        status = str(body_crop_row.get("detector_status", ""))
        score_raw = body_crop_row.get("detector_confidence")
        try:
            score_str = f"{float(score_raw):.2f}"
        except (TypeError, ValueError):
            score_str = "N/A"
        if status == "accepted":
            body_img = _load_thumb(body_crop_row.get("crop_path"))
            if body_img is None:
                body_img = _placeholder_image("missing_file")
        else:
            body_img = _placeholder_image(status)
        _paste_at(body_img, 1, f"body s={score_str}")
    else:
        _paste_at(_placeholder_image("no_record"), 1, "body N/A")

    # Slots 2–3: ear crops
    for slot_idx, ear_ordinal in enumerate([0, 1]):
        matching = [r for r in ear_rows if int(r.get("crop_ordinal", -1)) == ear_ordinal]
        ear_row = matching[0] if matching else None
        if ear_row is not None:
            e_status = str(ear_row.get("detector_status", ""))
            e_score_raw = ear_row.get("detector_confidence")
            try:
                e_score_str = f"{float(e_score_raw):.2f}"
            except (TypeError, ValueError):
                e_score_str = "N/A"
            if e_status == "accepted":
                ear_img = _load_thumb(ear_row.get("crop_path"))
                if ear_img is None:
                    ear_img = _placeholder_image("missing_file")
            else:
                ear_img = _placeholder_image(e_status)
            _paste_at(ear_img, 2 + slot_idx, f"ear_{ear_ordinal} s={e_score_str}")
        else:
            _paste_at(_placeholder_image("no_record"), 2 + slot_idx, f"ear_{ear_ordinal} N/A")

    # Bottom label row: image_id
    draw.text((_MARGIN, cell_h - _LABEL_HEIGHT), f"id:{image_id[:20]}", fill=(80, 80, 80), font=font)

    return cell


def _build_contact_sheet_page(
    cells: list[Image.Image],
    page_num: int,
) -> Image.Image:
    """Arrange cells in a grid and return a page image."""
    n = len(cells)
    if n == 0:
        return Image.new("RGB", (100, 100), color=(255, 255, 255))

    cols = min(_PAGE_COLS, n)
    rows = (n + cols - 1) // cols

    cell_w, cell_h = cells[0].size
    page_w = cols * cell_w + (cols + 1) * _MARGIN
    page_h = rows * cell_h + (rows + 1) * _MARGIN + 40  # 40 for page header

    page = Image.new("RGB", (page_w, page_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(page)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((_MARGIN, _MARGIN), f"BTEH Crop Pilot — Page {page_num}", fill=(40, 40, 40), font=font)

    for idx, cell in enumerate(cells):
        col = idx % cols
        row = idx // cols
        x = _MARGIN + col * (cell_w + _MARGIN)
        y = 40 + _MARGIN + row * (cell_h + _MARGIN)
        page.paste(cell, (x, y))

    return page


def generate_contact_sheets(
    pilot_df: pd.DataFrame,
    crop_df: pd.DataFrame,
    output_dir: Path,
    *,
    source_root: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> list[Path]:
    """
    Generate paginated contact sheets for all pilot images.

    Each image becomes one cell: source | body | ear_0 | ear_1.
    Uses source-relative paths/image IDs in labels — no absolute paths.

    Returns list of page image paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Index crop manifest by image_id for fast lookup
    crop_indexed = crop_df.copy()
    crop_indexed["image_id"] = crop_indexed["image_id"].astype(str)
    crop_indexed = crop_indexed.set_index("image_id")

    cells: list[Image.Image] = []
    page_num = 1

    for _, img_row in pilot_df.iterrows():
        image_id = str(img_row["image_id"])

        # Resolve source path
        source_path: str | None = None
        for col in ("source_path", "absolute_path"):
            v = img_row.get(col)
            if pd.notna(v) and str(v).strip():
                source_path = str(v)
                break
        if source_path is None:
            rel = img_row.get("source_relative_path")
            if pd.notna(rel) and str(rel).strip():
                if source_root:
                    source_path = str(Path(source_root) / str(rel))
                else:
                    source_path = str(rel)

        # Get crop rows for this image
        try:
            img_crops = crop_indexed.loc[[image_id]].reset_index()
        except KeyError:
            raise ValueError(
                f"Pilot image_id {image_id!r} is missing from the crop manifest"
            ) from None
        body_rows = img_crops[img_crops["crop_kind"] == "body"]
        ear_rows_df = img_crops[img_crops["crop_kind"] == "ear"]

        body_row = body_rows.iloc[0] if not body_rows.empty else None
        ear_rows = [ear_rows_df.iloc[i] for i in range(len(ear_rows_df))]

        cell = _make_cell(source_path, body_row, ear_rows, image_id, source_root)
        cells.append(cell)

        if len(cells) >= page_size:
            page_img = _build_contact_sheet_page(cells, page_num)
            page_path = output_dir / f"contact_sheet_page_{page_num:03d}.jpg"
            page_img.save(str(page_path), "JPEG", quality=85)
            logger.info("Contact sheet page %d written: %s", page_num, page_path)
            written.append(page_path)
            cells = []
            page_num += 1

    if cells:
        page_img = _build_contact_sheet_page(cells, page_num)
        page_path = output_dir / f"contact_sheet_page_{page_num:03d}.jpg"
        page_img.save(str(page_path), "JPEG", quality=85)
        logger.info("Contact sheet page %d written: %s", page_num, page_path)
        written.append(page_path)

    return written


# ---------------------------------------------------------------------------
# Sub-command: sample
# ---------------------------------------------------------------------------

def cmd_sample(args: argparse.Namespace) -> int:
    artifact_v_root = BTEH_ARTIFACT_ROOT / ARTIFACT_SCHEMA_VERSION

    manifest_path = Path(args.manifest) if args.manifest else (
        artifact_v_root / MANIFEST_SUBDIR / MANIFEST_FILENAME
    )
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    splits_path = Path(args.splits) if args.splits else (
        artifact_v_root / SPLITS_SUBDIR / SPLITS_FILENAME
    )

    manifest = pd.read_parquet(manifest_path)
    logger.info("Loaded manifest: %d rows", len(manifest))

    source_fingerprint = fingerprint_dataframe(manifest, "image_id")

    splits: pd.DataFrame | None = None
    split_fingerprint: str | None = None
    if splits_path.exists():
        splits = pd.read_parquet(splits_path)
        split_fingerprint = fingerprint_dataframe_columns(
            splits,
            [
                "image_id",
                "split",
                "split_protocol",
                "evaluable",
                "fold",
                "session_id",
            ],
            sort_by=["image_id"],
        )
        logger.info("Loaded splits: %d rows", len(splits))
    else:
        logger.warning("Splits file not found at %s; proceeding without split labels.", splits_path)

    pilot_df, review_df, strata_report = select_pilot_sample(
        manifest,
        splits=splits,
        n_pilot=args.n_pilot,
        n_review=args.n_review,
        seed=args.seed,
    )

    logger.info(
        "Pilot selected: %d named, %d review/unresolved", len(pilot_df), len(review_df)
    )

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else artifact_v_root / "pilot"
    )
    write_pilot_manifest(
        pilot_df,
        review_df,
        strata_report,
        output_dir,
        source_fingerprint=source_fingerprint,
        split_fingerprint=split_fingerprint,
        seed=args.seed,
    )

    print(f"Pilot manifest: {output_dir / PILOT_MANIFEST_FILENAME}")
    print(f"Sidecar: {output_dir / PILOT_SIDECAR_FILENAME}")
    print(
        f"Selected {len(pilot_df)} named images across "
        f"{strata_report['n_identities']} identities, "
        f"{len(review_df)} review/unresolved."
    )
    return 0


# ---------------------------------------------------------------------------
# Sub-command: report
# ---------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> int:
    artifact_v_root = BTEH_ARTIFACT_ROOT / ARTIFACT_SCHEMA_VERSION

    pilot_manifest_path = Path(args.pilot_manifest) if args.pilot_manifest else (
        artifact_v_root / "pilot" / PILOT_MANIFEST_FILENAME
    )
    if not pilot_manifest_path.exists():
        logger.error("Pilot manifest not found: %s. Run 'sample' first.", pilot_manifest_path)
        return 1

    sidecar_path = pilot_manifest_path.with_suffix(".json")
    if sidecar_path.exists():
        with open(sidecar_path) as f:
            sidecar = json.load(f)
    else:
        logger.warning("Sidecar not found: %s; skipping fingerprint check.", sidecar_path)
        sidecar = {}

    crop_manifest_path = Path(args.crop_manifest) if args.crop_manifest else (
        artifact_v_root / "crops" / "crop_manifest.parquet"
    )
    if not crop_manifest_path.exists():
        logger.error("Crop manifest not found: %s. Run run_bteh_detection first.", crop_manifest_path)
        return 1

    pilot_df_full = pd.read_parquet(pilot_manifest_path)
    pilot_role_col = "_pilot_role" if "_pilot_role" in pilot_df_full.columns else None
    if pilot_role_col:
        pilot_df = pilot_df_full[pilot_df_full[pilot_role_col] == "pilot"].copy()
    else:
        pilot_df = pilot_df_full.copy()

    crop_df = pd.read_parquet(crop_manifest_path)

    # --- Fail-loud checks ---
    _fail_loud_schema(crop_df)

    if sidecar:
        _fail_loud_fingerprint(pilot_df, sidecar)
        pilot_ids = set(pilot_df["image_id"].astype(str))
        pilot_crops = crop_df[crop_df["image_id"].astype(str).isin(pilot_ids)]
        _fail_loud_crop_fingerprints(pilot_crops, sidecar)

    _fail_loud_joins(pilot_df, crop_df)

    if args.check_files:
        pilot_ids = set(pilot_df["image_id"].astype(str))
        _fail_loud_accepted_files_exist(crop_df, pilot_ids)

    source_root = args.source_root or str(BTEH_SOURCE_ROOT)

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else artifact_v_root / REPORTS_SUBDIR
    )
    cs_dir = (
        Path(args.output_dir) / "contact_sheets" if args.output_dir
        else artifact_v_root / CONTACT_SHEETS_SUBDIR
    )

    review_csv = Path(args.review_csv) if args.review_csv else None

    # Metrics
    metrics = compute_crop_metrics(
        pilot_df, crop_df, review_csv_path=review_csv
    )
    write_metric_reports(metrics, output_dir)

    # Contact sheets
    page_paths = generate_contact_sheets(
        pilot_df,
        crop_df,
        cs_dir,
        source_root=source_root,
        page_size=args.page_size,
    )

    print(f"Reports written to: {output_dir}")
    print(f"Contact sheets ({len(page_paths)} pages): {cs_dir}")
    det = metrics.get("detector", {})
    print(
        f"Detector summary — body coverage: {det.get('body_accepted_coverage', 'N/A')}, "
        f"≥1 ear: {det.get('image_ge1_ear_coverage', 'N/A')}, "
        f"2 ear: {det.get('image_2ear_coverage', 'N/A')}"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bteh_crop_pilot",
        description=dedent("""\
            BTEH crop-quality pilot tooling.

            Sub-commands:
              sample   Select stratified pilot sample and write manifest + sidecar.
              report   Generate contact sheets and quality metrics from crop manifest.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- sample --
    sp = sub.add_parser("sample", help="Select pilot sample")
    sp.add_argument("--manifest", default=None, help="Path to bteh_image_manifest.parquet")
    sp.add_argument("--splits", default=None, help="Path to bteh_splits.parquet")
    sp.add_argument("--output-dir", default=None, help="Output directory (default: <artifact-root>/v1/pilot)")
    sp.add_argument("--n-pilot", type=int, default=DEFAULT_PILOT_N,
                    help=f"Target pilot sample size (default: {DEFAULT_PILOT_N})")
    sp.add_argument("--n-review", type=int, default=DEFAULT_REVIEW_N,
                    help=f"Target review/unresolved audit size (default: {DEFAULT_REVIEW_N})")
    sp.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"Random seed (default: {DEFAULT_SEED})")
    sp.add_argument("--verbose", "-v", action="store_true")

    # -- report --
    rp = sub.add_parser("report", help="Generate contact sheets and quality metrics")
    rp.add_argument("--pilot-manifest", default=None, help="Path to bteh_pilot_manifest.parquet")
    rp.add_argument("--crop-manifest", default=None, help="Path to crop_manifest.parquet")
    rp.add_argument("--source-root", default=None, help="BTEH source image root for loading thumbnails")
    rp.add_argument("--output-dir", default=None, help="Output directory for reports")
    rp.add_argument("--review-csv", default=None, help="Optional human-review CSV (crop_id, status columns)")
    rp.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE,
                    help=f"Images per contact-sheet page (default: {DEFAULT_PAGE_SIZE})")
    rp.add_argument("--check-files", action="store_true",
                    help="Fail if accepted crop files are missing from disk")
    rp.add_argument("--verbose", "-v", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.command == "sample":
        return cmd_sample(args)
    elif args.command == "report":
        return cmd_report(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
