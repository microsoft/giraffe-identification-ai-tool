#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
BTEH head-crop quality pilot: contact-sheet reports, machine-readable metrics,
visual review enforcement, and detector fingerprint management.

Sub-commands
------------
  sample       Select (or reuse) the 120-image pilot sample from the existing
               body-pilot manifest; write a head-specific pilot sidecar.
  report       Consume the pilot + head experiment manifest; produce head
               contact sheets and JSON/CSV/Markdown metric reports.
  fingerprint  Write a content-addressed detector-config fingerprint file;
               fail on any source/split/detector mismatch or missing accepted
               head crop files.

Design constraints
------------------
* This module never reads or writes the production selected-v1 body/ear
  crop manifests or the body pilot report.
* No real model inference is performed here; head experiment manifests are
  produced by pipeline/step_1_run_head_detection.py.
* Contact-sheet labels use relative paths and crop_ids — no absolute paths.
* Review CSV must cover every accepted head crop exactly once; non-accepted
  crop_ids (none_detected / not_applicable / failed) must not appear in the
  review CSV.
* Precision gate: decisive precision (accepted / (accepted + rejected)) must
  meet the configurable threshold (default 0.95); gate failure exits non-zero.

Usage
-----
    python pipeline/bteh_head_crop_pilot.py sample \\
        [--pilot-manifest PATH]  \\   # default: <artifact-root>/v1/pilot/bteh_pilot_manifest.parquet
        [--output-dir    PATH]       # default: <artifact-root>/v1/pilot/head

    python pipeline/bteh_head_crop_pilot.py report \\
        [--pilot-manifest  PATH]  \\
        [--head-manifest   PATH]  \\
        [--output-dir      PATH]  \\
        [--body-manifest   PATH]  \\  # optional; enables body-crop column in contact sheets
        [--source-root     PATH]  \\
        [--review-csv      PATH]  \\
        [--precision-gate  FLOAT] \\  # default 0.95
        [--page-size       N]     \\
        [--check-files]

    python pipeline/bteh_head_crop_pilot.py fingerprint \\
        [--pilot-manifest  PATH]  \\
        [--head-manifest   PATH]  \\
        [--output-dir      PATH]  \\
        [--check-files]           \\
        --conf-threshold   FLOAT  \\
        --min-area-frac    FLOAT  \\
        --max-area-frac    FLOAT  \\
        --min-aspect       FLOAT  \\
        --max-aspect       FLOAT  \\
        --iou-threshold    FLOAT  \\
        --pad-frac         FLOAT
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
    EXPERIMENT_ROOT,
    HEAD_MANIFEST_FILENAME,
    REPORTS_SUBDIR,
)
from utils.artifact_schema import (
    HEAD_EXPERIMENT_MANIFEST_COLUMNS,
    TERMINAL_CROP_STATUSES,
    fingerprint_dataframe,
    make_crop_id,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PILOT_N: int = 120
DEFAULT_SEED: int = 42
DEFAULT_PAGE_SIZE: int = 12
DEFAULT_PRECISION_GATE: float = 0.95

HEAD_PILOT_SIDECAR_FILENAME: str = "bteh_head_pilot_manifest.json"
HEAD_PILOT_REPORT_JSON: str = "bteh_head_pilot_report.json"
HEAD_PILOT_REPORT_CSV: str = "bteh_head_pilot_report.csv"
HEAD_PILOT_REPORT_MD: str = "bteh_head_pilot_report.md"
HEAD_PILOT_FINGERPRINT_FILENAME: str = "bteh_head_pilot_detector_fingerprint.json"

# Pilot manifest that must already exist (body-pilot selected-v1)
BODY_PILOT_MANIFEST_FILENAME: str = "bteh_pilot_manifest.parquet"

_THUMB_SIZE = (192, 192)
_LABEL_HEIGHT = 28
_MARGIN = 8
_PAGE_COLS = 4

_PLACEHOLDER_COLORS: dict[str, tuple[int, int, int]] = {
    "none_detected": (200, 200, 200),
    "not_applicable": (180, 180, 220),
    "failed": (255, 180, 180),
    "pending": (255, 240, 180),
    "no_source": (230, 230, 180),
    "no_record": (210, 210, 230),
    "missing_file": (255, 200, 200),
}


# ---------------------------------------------------------------------------
# Utility: strip absolute path prefix for labels
# ---------------------------------------------------------------------------

def _strip_absolute_prefix(path_str: str, source_root: str | None) -> str:
    """Return a source-relative display path, stripping any absolute prefix."""
    if not path_str:
        return path_str
    if source_root:
        try:
            rel = Path(path_str).relative_to(source_root)
            return str(rel)
        except ValueError:
            pass
    import re
    path_str = re.sub(r"^/" r"home/[^/]+/[^/]+/", ".../", path_str)
    return path_str


# ---------------------------------------------------------------------------
# Schema / fingerprint validation
# ---------------------------------------------------------------------------

def _fail_loud_head_schema(head_df: pd.DataFrame) -> None:
    """Raise ValueError when head manifest is missing required columns."""
    missing = [c for c in HEAD_EXPERIMENT_MANIFEST_COLUMNS if c not in head_df.columns]
    if missing:
        raise ValueError(
            f"Head manifest is missing required columns: {missing}"
        )


def _fail_loud_head_crop_kind(head_df: pd.DataFrame) -> None:
    """Raise ValueError if any rows have crop_kind != 'head'."""
    non_head = head_df[head_df["crop_kind"] != "head"]
    if not non_head.empty:
        ids = non_head["crop_id"].tolist()[:10]
        raise ValueError(
            f"Head manifest contains non-head rows: {ids}"
        )


def _fail_loud_head_joins(
    pilot_df: pd.DataFrame,
    head_df: pd.DataFrame,
) -> None:
    """
    Every pilot image_id must appear in the head manifest.
    Accepted head crops must have a non-null crop_path.
    """
    pilot_ids = set(pilot_df["image_id"].dropna().astype(str))
    head_ids = set(head_df["image_id"].dropna().astype(str))
    missing = sorted(pilot_ids - head_ids)
    if missing:
        raise ValueError(
            f"{len(missing)} pilot image_id(s) have no entry in the head manifest: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    accepted = head_df[
        head_df["image_id"].astype(str).isin(pilot_ids)
        & (head_df["detector_status"] == "accepted")
    ]
    null_paths = accepted[
        accepted["crop_path"].isna() | accepted["crop_path"].astype(str).str.strip().eq("")
    ]
    if not null_paths.empty:
        bad = null_paths["crop_id"].tolist()[:10]
        raise ValueError(
            f"{len(null_paths)} accepted head crop(s) have null/empty crop_path: {bad}"
        )


def _fail_loud_head_accepted_files_exist(
    head_df: pd.DataFrame,
    pilot_ids: set[str],
) -> None:
    """Raise ValueError when accepted head crop files are missing from disk."""
    accepted = head_df[
        head_df["image_id"].astype(str).isin(pilot_ids)
        & (head_df["detector_status"] == "accepted")
        & head_df["crop_path"].notna()
    ]
    missing_files: list[str] = []
    for _, row in accepted.iterrows():
        p = Path(str(row["crop_path"]))
        if not p.exists():
            missing_files.append(str(row["crop_id"]))
    if missing_files:
        raise ValueError(
            f"{len(missing_files)} accepted head crop file(s) are missing from disk: "
            f"{missing_files[:10]}{'...' if len(missing_files) > 10 else ''}"
        )


def _fail_loud_head_detector_fingerprint(
    head_df: pd.DataFrame,
    expected_fingerprint: str,
) -> None:
    """Raise ValueError when the head manifest does not match the expected detector fingerprint."""
    if head_df.empty:
        return
    accepted = head_df[head_df["detector_status"] == "accepted"]
    if accepted.empty:
        return
    actual_fps = accepted["detector_fingerprint"].dropna().unique().tolist()
    if len(actual_fps) != 1 or actual_fps[0] != expected_fingerprint:
        raise ValueError(
            f"Head manifest detector_fingerprint mismatch: "
            f"expected {expected_fingerprint!r}, found {actual_fps!r}. "
            "Re-run head detection with the matching detector configuration."
        )


def _fail_loud_head_source_split_fingerprints(
    head_df: pd.DataFrame,
    fingerprint_record: dict,
) -> None:
    """Raise ValueError when source or split fingerprints disagree with the fingerprint record."""
    for col, key in (
        ("source_fingerprint", "source_fingerprint"),
        ("split_fingerprint", "split_fingerprint"),
    ):
        expected = fingerprint_record.get(key)
        if not expected:
            continue
        actual = set(head_df[col].dropna().astype(str).unique())
        if actual != {str(expected)}:
            raise ValueError(
                f"Head manifest {col} mismatch: "
                f"expected {expected!r}, found {sorted(actual)!r}."
            )


# ---------------------------------------------------------------------------
# Pilot sample selection (reuses the existing body-pilot manifest)
# ---------------------------------------------------------------------------

def load_body_pilot_manifest(
    pilot_manifest_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Load the existing fixed-120 body-pilot manifest and its sidecar.

    Returns (pilot_df, sidecar) where pilot_df contains only the 'pilot'
    role rows (excluding 'review' rows).
    """
    if not pilot_manifest_path.exists():
        raise FileNotFoundError(
            f"Body-pilot manifest not found: {pilot_manifest_path}. "
            "Run 'bteh_crop_pilot.py sample' first."
        )
    full_df = pd.read_parquet(pilot_manifest_path)
    if "_pilot_role" in full_df.columns:
        pilot_df = full_df[full_df["_pilot_role"] == "pilot"].copy()
    else:
        pilot_df = full_df.copy()

    sidecar_path = pilot_manifest_path.with_suffix(".json")
    sidecar: dict = {}
    if sidecar_path.exists():
        with open(sidecar_path) as f:
            sidecar = json.load(f)

    return pilot_df, sidecar


def write_head_pilot_sidecar(
    pilot_df: pd.DataFrame,
    sidecar: dict,
    output_dir: Path,
) -> Path:
    """
    Write a head-specific sidecar JSON that records which pilot image_ids are
    being used for the head pilot, plus inherited source/split fingerprints.

    This is a lightweight provenance record; the actual pilot image_ids are
    inherited from the body-pilot manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    head_sidecar = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "body_pilot_source_fingerprint": sidecar.get("source_fingerprint"),
        "body_pilot_split_fingerprint": sidecar.get("split_fingerprint"),
        "body_pilot_pilot_fingerprint": sidecar.get("pilot_fingerprint"),
        "n_pilot": len(pilot_df),
        "pilot_image_ids": sorted(pilot_df["image_id"].astype(str).tolist()),
        "note": (
            "Head-crop pilot reuses the fixed-120 named body-pilot sample. "
            "Do not modify this file manually."
        ),
    }
    sidecar_path = output_dir / HEAD_PILOT_SIDECAR_FILENAME
    sidecar_path.write_text(json.dumps(head_sidecar, indent=2))
    logger.info("Head pilot sidecar written: %s", sidecar_path)
    return sidecar_path


# ---------------------------------------------------------------------------
# Detector fingerprint record
# ---------------------------------------------------------------------------

def compute_detector_config_fingerprint(
    conf_threshold: float,
    min_area_frac: float,
    max_area_frac: float,
    min_aspect: float,
    max_aspect: float,
    iou_threshold: float,
    pad_frac: float,
    prompt: str = "elephant head.",
) -> str:
    """Return a short (16-char) SHA-256 of the head detector hyperparameters."""
    payload = json.dumps(
        {
            "prompt": prompt,
            "conf_threshold": conf_threshold,
            "min_area_frac": min_area_frac,
            "max_area_frac": max_area_frac,
            "min_aspect": min_aspect,
            "max_aspect": max_aspect,
            "iou_threshold": iou_threshold,
            "pad_frac": pad_frac,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def write_detector_fingerprint_record(
    fingerprint: str,
    config: dict,
    pilot_df: pd.DataFrame,
    body_pilot_sidecar: dict,
    output_dir: Path,
) -> Path:
    """
    Write a content-addressed detector fingerprint JSON file.

    The record captures:
    - detector_fingerprint: short hash of detector hyperparameters
    - detector_config: the hyperparameters themselves
    - source_fingerprint / split_fingerprint: inherited from body-pilot sidecar
    - pilot_image_ids: sorted list of image_ids being piloted
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "detector_fingerprint": fingerprint,
        "detector_config": config,
        "source_fingerprint": body_pilot_sidecar.get("source_fingerprint"),
        "split_fingerprint": body_pilot_sidecar.get("split_fingerprint"),
        "n_pilot": len(pilot_df),
        "pilot_image_ids": sorted(pilot_df["image_id"].astype(str).tolist()),
    }
    fp_path = output_dir / HEAD_PILOT_FINGERPRINT_FILENAME
    fp_path.write_text(json.dumps(record, indent=2))
    logger.info("Head pilot detector fingerprint written: %s", fp_path)
    return fp_path


def load_detector_fingerprint_record(output_dir: Path) -> dict | None:
    """Load existing fingerprint record, or return None if absent."""
    fp_path = output_dir / HEAD_PILOT_FINGERPRINT_FILENAME
    if not fp_path.exists():
        return None
    with open(fp_path) as f:
        return json.load(f)


def _fail_loud_fingerprint_record_match(
    existing: dict,
    computed_fingerprint: str,
    source_fingerprint: str | None,
    split_fingerprint: str | None,
) -> None:
    """Raise ValueError if any fingerprint in the existing record disagrees."""
    if existing.get("detector_fingerprint") != computed_fingerprint:
        raise ValueError(
            f"Detector fingerprint mismatch: "
            f"existing={existing['detector_fingerprint']!r}, "
            f"computed={computed_fingerprint!r}. "
            "Detector config has changed; delete the fingerprint record and re-run."
        )
    if source_fingerprint and existing.get("source_fingerprint") not in (None, source_fingerprint):
        raise ValueError(
            f"Source fingerprint mismatch: "
            f"existing={existing['source_fingerprint']!r}, "
            f"current={source_fingerprint!r}."
        )
    if split_fingerprint and existing.get("split_fingerprint") not in (None, split_fingerprint):
        raise ValueError(
            f"Split fingerprint mismatch: "
            f"existing={existing['split_fingerprint']!r}, "
            f"current={split_fingerprint!r}."
        )


# ---------------------------------------------------------------------------
# Head metrics computation
# ---------------------------------------------------------------------------

def compute_head_metrics(
    pilot_df: pd.DataFrame,
    head_df: pd.DataFrame,
    body_df: pd.DataFrame | None = None,
    *,
    review_csv_path: Path | None = None,
) -> dict:
    """
    Compute machine-readable head crop quality metrics.

    Separates detector coverage from human visual correctness:
    - ``detector_*`` : based solely on detector_status values
    - ``human_*``    : based on human review CSV (if provided)

    Parameters
    ----------
    pilot_df       : pilot image manifest (image_id, individual_id, year, ...)
    head_df        : head experiment manifest (from step_1_run_head_detection.py)
    body_df        : optional body crop manifest for body-input availability
    review_csv_path: optional human-review CSV (crop_id, status, reason columns)

    Returns
    -------
    Nested dict of metrics.
    """
    pilot_ids = set(pilot_df["image_id"].dropna().astype(str))
    head_pilot = head_df[head_df["image_id"].astype(str).isin(pilot_ids)].copy()

    n_images = len(pilot_ids)

    # --- Body input availability ---
    body_available_ids: set[str] = set()
    if body_df is not None:
        accepted_body = body_df[body_df["detector_status"] == "accepted"]
        body_available_ids = set(
            accepted_body["image_id"].astype(str).isin(pilot_ids).pipe(
                lambda m: accepted_body.loc[m, "image_id"].astype(str)
            )
        )
        # More direct:
        body_available_ids = set(
            accepted_body[
                accepted_body["image_id"].astype(str).isin(pilot_ids)
            ]["image_id"].astype(str)
        )

    # --- Head detector coverage ---
    head_accepted = head_pilot[head_pilot["detector_status"] == "accepted"]
    head_accepted_ids = set(head_accepted["image_id"].astype(str))

    # source_used breakdown
    source_used_counts: dict[str, int] = {}
    if "source_used" in head_accepted.columns:
        source_used_counts = {
            k: int(v)
            for k, v in head_accepted["source_used"]
            .fillna("unknown")
            .value_counts()
            .to_dict()
            .items()
        }

    # body_crop fallback rate: accepted heads that used 'original' instead of 'body_crop'
    n_from_body_crop = source_used_counts.get("body_crop", 0)
    n_from_original = source_used_counts.get("original", 0)
    n_head_accepted = int(len(head_accepted))

    # Status breakdown
    status_counts: dict[str, int] = {}
    if not head_pilot.empty:
        status_counts = {
            k: int(v)
            for k, v in head_pilot["detector_status"]
            .fillna("unknown")
            .value_counts()
            .to_dict()
            .items()
        }

    # --- Detector confidence / area / aspect distributions ---
    distributions = _compute_distributions(head_accepted)

    # --- Strata metrics ---
    strata_metrics = _compute_head_strata_metrics(pilot_df, head_pilot)

    # --- Human review metrics ---
    human_metrics = _compute_head_human_metrics(
        head_pilot, review_csv_path
    )

    detector_metrics = {
        "n_images": n_images,
        "n_body_input_available": len(body_available_ids) if body_df is not None else None,
        "body_input_coverage": (
            round(len(body_available_ids) / n_images, 4)
            if n_images and body_df is not None
            else None
        ),
        "head_accepted_coverage": round(len(head_accepted_ids) / n_images, 4) if n_images else 0.0,
        "n_head_accepted": n_head_accepted,
        "n_from_body_crop": n_from_body_crop,
        "n_from_original_fallback": n_from_original,
        "body_crop_rate": round(n_from_body_crop / n_head_accepted, 4) if n_head_accepted else None,
        "original_fallback_rate": round(n_from_original / n_head_accepted, 4) if n_head_accepted else None,
        "source_used_counts": source_used_counts,
        "status_counts": status_counts,
        "distributions": distributions,
    }

    return {
        "detector": detector_metrics,
        "human": human_metrics,
        "by_stratum": strata_metrics,
    }


def _compute_distributions(head_accepted: pd.DataFrame) -> dict:
    """Compute confidence, area, and aspect ratio distributions for accepted heads."""
    result: dict = {}

    if head_accepted.empty:
        return {"confidence": {}, "area_frac": {}, "aspect_ratio": {}}

    # Confidence distribution
    if "detector_confidence" in head_accepted.columns:
        conf = head_accepted["detector_confidence"].dropna().astype(float)
        result["confidence"] = _histogram_stats(conf, "confidence")
    else:
        result["confidence"] = {}

    # Box-derived area and aspect ratio
    areas: list[float] = []
    aspects: list[float] = []
    if "detector_box" in head_accepted.columns:
        for box_str in head_accepted["detector_box"].dropna():
            try:
                box = json.loads(str(box_str))
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    w = abs(x2 - x1)
                    h = abs(y2 - y1)
                    area = w * h
                    areas.append(float(area))
                    if h > 0:
                        aspects.append(float(w) / float(h))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

    result["area_px"] = _histogram_stats(pd.Series(areas), "area_px") if areas else {}
    result["aspect_ratio"] = _histogram_stats(pd.Series(aspects), "aspect_ratio") if aspects else {}
    return result


def _histogram_stats(series: pd.Series, name: str) -> dict:
    """Return basic descriptive stats and a simple histogram for a numeric Series."""
    if series.empty:
        return {}
    arr = series.dropna()
    if arr.empty:
        return {}
    return {
        "count": int(len(arr)),
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "p25": round(float(arr.quantile(0.25)), 4),
        "median": round(float(arr.median()), 4),
        "p75": round(float(arr.quantile(0.75)), 4),
        "max": round(float(arr.max()), 4),
    }


def _compute_head_strata_metrics(
    pilot_df: pd.DataFrame,
    head_pilot: pd.DataFrame,
) -> dict:
    """Breakdown by year, session_source, dataset_role, source_used."""
    result: dict = {}
    meta = pilot_df.copy()
    meta["_img_id"] = meta["image_id"].astype(str)

    dims = [
        ("year", "year"),
        ("session_source", "session_source"),
        ("origin", "dataset_role"),
    ]
    for dim, col in dims:
        if col not in meta.columns:
            continue
        dim_counts: dict = {}
        for val, grp in meta.groupby(meta[col].fillna("unknown").astype(str)):
            ids = set(grp["_img_id"].tolist())
            sub = head_pilot[head_pilot["image_id"].astype(str).isin(ids)]
            accepted = int((sub["detector_status"] == "accepted").sum())
            none_detected = int((sub["detector_status"] == "none_detected").sum())
            not_applicable = int((sub["detector_status"] == "not_applicable").sum())
            from_body = 0
            from_orig = 0
            if "source_used" in sub.columns:
                from_body = int((sub["source_used"] == "body_crop").sum())
                from_orig = int((sub["source_used"] == "original").sum())
            dim_counts[str(val)] = {
                "n_images": len(ids),
                "head_accepted": accepted,
                "none_detected": none_detected,
                "not_applicable": not_applicable,
                "from_body_crop": from_body,
                "from_original": from_orig,
            }
        result[dim] = dim_counts

    return result


def _compute_head_human_metrics(
    head_pilot: pd.DataFrame,
    review_csv_path: Path | None,
) -> dict:
    """
    Parse optional human review CSV and compute head-crop precision.

    Rules:
    - Review must cover each accepted head crop exactly once.
    - Non-accepted crop_ids (none_detected / not_applicable / failed) must
      not appear in the review CSV.
    - Precision = accepted / (accepted + rejected); uncertain excluded.
    - Returns coverage_complete=True only when every accepted crop is reviewed.
    """
    accepted_crop_ids = set(
        head_pilot.loc[
            head_pilot["detector_status"] == "accepted", "crop_id"
        ].astype(str)
    )
    terminal_crop_ids = set(
        head_pilot.loc[
            head_pilot["detector_status"].isin(TERMINAL_CROP_STATUSES - {"accepted"}),
            "crop_id",
        ].astype(str)
    )

    if review_csv_path is None:
        return {
            "precision": None,
            "coverage_complete": False,
            "n_accepted_crops": len(accepted_crop_ids),
            "n_reviewed": 0,
            "n_accepted": 0,
            "n_rejected": 0,
            "n_uncertain": 0,
            "terminal_ids_in_review": [],
            "missing_from_review": sorted(accepted_crop_ids),
            "note": "No human review CSV provided. Precision unavailable.",
        }

    review_path = Path(review_csv_path)
    if not review_path.exists():
        raise FileNotFoundError(f"Human review CSV not found: {review_path}")

    review_df = pd.read_csv(review_path)
    required_cols = {"crop_id", "status"}
    missing_cols = required_cols - set(review_df.columns)
    if missing_cols:
        raise ValueError(
            f"Human review CSV is missing required columns: {sorted(missing_cols)}"
        )

    valid_statuses = {"accepted", "rejected", "uncertain"}
    bad_statuses = set(review_df["status"].dropna().unique()) - valid_statuses
    if bad_statuses:
        raise ValueError(
            f"Human review CSV contains invalid status values: {bad_statuses}. "
            f"Must be one of: {sorted(valid_statuses)}"
        )

    review_crop_ids = set(review_df["crop_id"].astype(str))

    # Terminal IDs that should not be in the review CSV
    terminal_ids_in_review = sorted(review_crop_ids & terminal_crop_ids)

    # Accepted crops not in review
    missing_from_review = sorted(accepted_crop_ids - review_crop_ids)

    # Restrict to accepted-crop reviews only (filter out any terminal IDs)
    review_for_accepted = review_df[
        review_df["crop_id"].astype(str).isin(accepted_crop_ids)
    ]

    # Check for duplicate reviews
    dup_ids = review_for_accepted[
        review_for_accepted["crop_id"].duplicated(keep=False)
    ]["crop_id"].tolist()
    if dup_ids:
        raise ValueError(
            f"Human review CSV contains duplicate entries for crop_id(s): "
            f"{sorted(set(dup_ids))[:10]}. "
            "Each accepted head crop must be reviewed exactly once."
        )

    n_accepted = int((review_for_accepted["status"] == "accepted").sum())
    n_rejected = int((review_for_accepted["status"] == "rejected").sum())
    n_uncertain = int((review_for_accepted["status"] == "uncertain").sum())
    total_decisive = n_accepted + n_rejected

    precision = round(n_accepted / total_decisive, 4) if total_decisive > 0 else None

    coverage_complete = (
        len(missing_from_review) == 0
        and len(terminal_ids_in_review) == 0
    )

    rejection_reasons: dict = {}
    if "reason" in review_for_accepted.columns:
        rejected_rows = review_for_accepted[review_for_accepted["status"] == "rejected"]
        if not rejected_rows.empty:
            rejection_reasons = {
                k: int(v)
                for k, v in rejected_rows["reason"]
                .fillna("unspecified")
                .value_counts()
                .to_dict()
                .items()
            }

    return {
        "n_accepted_crops": len(accepted_crop_ids),
        "n_reviewed": len(review_for_accepted),
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "n_uncertain": n_uncertain,
        "precision": precision,
        "coverage_complete": coverage_complete,
        "missing_from_review": missing_from_review,
        "terminal_ids_in_review": terminal_ids_in_review,
        "rejection_reasons": rejection_reasons,
        "note": (
            "Precision = accepted / (accepted + rejected); uncertain crops excluded. "
            "Visual correctness is independent of detector confidence."
            if precision is not None
            else "No decisive reviews; precision unavailable."
        ),
    }


def check_precision_gate(
    metrics: dict,
    gate: float = DEFAULT_PRECISION_GATE,
) -> bool:
    """
    Return True when precision meets the gate threshold.
    Returns True unconditionally when no review CSV was provided (precision is None).
    """
    precision = metrics.get("human", {}).get("precision")
    if precision is None:
        return True
    return float(precision) >= gate


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_head_metric_reports(
    metrics: dict,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write JSON, CSV, and Markdown head-crop metric reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / HEAD_PILOT_REPORT_JSON
    json_path.write_text(json.dumps(metrics, indent=2))
    logger.info("JSON report: %s", json_path)

    csv_path = output_dir / HEAD_PILOT_REPORT_CSV
    det = metrics.get("detector", {})
    hum = metrics.get("human", {})
    rows_csv = [
        {"metric": "n_images", "value": det.get("n_images", ""), "note": ""},
        {
            "metric": "n_body_input_available",
            "value": det.get("n_body_input_available", "N/A"),
            "note": "accepted body crops in pilot",
        },
        {
            "metric": "body_input_coverage",
            "value": det.get("body_input_coverage", "N/A"),
            "note": "detector only",
        },
        {
            "metric": "head_accepted_coverage",
            "value": det.get("head_accepted_coverage", ""),
            "note": "detector only",
        },
        {
            "metric": "n_head_accepted",
            "value": det.get("n_head_accepted", ""),
            "note": "detector only",
        },
        {
            "metric": "n_from_body_crop",
            "value": det.get("n_from_body_crop", ""),
            "note": "source_used=body_crop",
        },
        {
            "metric": "n_from_original_fallback",
            "value": det.get("n_from_original_fallback", ""),
            "note": "source_used=original",
        },
        {
            "metric": "body_crop_rate",
            "value": det.get("body_crop_rate", "N/A"),
            "note": "fraction of accepted heads from body_crop source",
        },
        {
            "metric": "original_fallback_rate",
            "value": det.get("original_fallback_rate", "N/A"),
            "note": "fraction of accepted heads from original fallback",
        },
        {
            "metric": "human_precision",
            "value": hum.get("precision", "N/A"),
            "note": hum.get("note", ""),
        },
        {
            "metric": "review_coverage_complete",
            "value": str(hum.get("coverage_complete", False)),
            "note": "all accepted crops reviewed exactly once",
        },
    ]
    for status, count in det.get("status_counts", {}).items():
        rows_csv.append({
            "metric": f"head_status_{status}",
            "value": count,
            "note": "detector only",
        })
    for src, count in det.get("source_used_counts", {}).items():
        rows_csv.append({
            "metric": f"source_used_{src}",
            "value": count,
            "note": "accepted head crops",
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows_csv)
    logger.info("CSV report: %s", csv_path)

    md_path = output_dir / HEAD_PILOT_REPORT_MD
    md_lines = [
        "# BTEH Head-Crop Pilot Quality Report",
        "",
        "## Detector Metrics",
        "",
        f"- Images in pilot: **{det.get('n_images', 'N/A')}**",
        f"- Body input available: **{det.get('n_body_input_available', 'N/A')}** "
        f"({det.get('body_input_coverage', 'N/A')} coverage)",
        f"- Head accepted coverage: **{det.get('head_accepted_coverage', 'N/A')}** (detector)",
        f"- Accepted head crops: **{det.get('n_head_accepted', 'N/A')}**",
        f"- From body_crop source: **{det.get('n_from_body_crop', 'N/A')}** "
        f"(rate: {det.get('body_crop_rate', 'N/A')})",
        f"- From original fallback: **{det.get('n_from_original_fallback', 'N/A')}** "
        f"(rate: {det.get('original_fallback_rate', 'N/A')})",
        "",
        "### Head Detector Status Counts",
        "",
        "| Status | Count |",
        "| ------ | ----- |",
    ]
    for status, count in sorted(det.get("status_counts", {}).items()):
        md_lines.append(f"| {status} | {count} |")

    md_lines += [
        "",
        "### Detector Score / Area / Aspect Distributions",
        "",
    ]
    for dist_name, dist_stats in det.get("distributions", {}).items():
        if not dist_stats:
            continue
        md_lines.append(f"**{dist_name}**")
        md_lines.append("")
        md_lines.append("| Stat | Value |")
        md_lines.append("| ---- | ----- |")
        for k, v in dist_stats.items():
            md_lines.append(f"| {k} | {v} |")
        md_lines.append("")

    md_lines += [
        "## Visual Review (Human)",
        "",
        "> Visual correctness is independent of detector confidence.",
        "> Precision = accepted / (accepted + rejected); uncertain crops excluded.",
        "",
        f"- Accepted head crops: {hum.get('n_accepted_crops', 'N/A')}",
        f"- Reviewed: {hum.get('n_reviewed', 'N/A')}",
        f"- Accepted: {hum.get('n_accepted', 'N/A')}",
        f"- Rejected: {hum.get('n_rejected', 'N/A')}",
        f"- Uncertain: {hum.get('n_uncertain', 'N/A')}",
        f"- **Precision: {hum.get('precision', 'N/A')}**",
        f"- Coverage complete: {hum.get('coverage_complete', False)}",
        f"- Note: _{hum.get('note', '')}_",
        "",
    ]
    terminal_in_review = hum.get("terminal_ids_in_review", [])
    if terminal_in_review:
        md_lines.append(
            f"> ⚠️ WARNING: {len(terminal_in_review)} terminal-status crop_id(s) "
            "found in review CSV — these should be removed."
        )
        md_lines.append("")
    missing = hum.get("missing_from_review", [])
    if missing:
        md_lines.append(
            f"> ⚠️ WARNING: {len(missing)} accepted crop(s) not yet reviewed."
        )
        md_lines.append("")

    md_lines += [
        "## Breakdown by Stratum",
        "",
    ]
    for dim, dim_data in metrics.get("by_stratum", {}).items():
        md_lines.append(f"### {dim.title()}")
        md_lines.append("")
        md_lines.append(
            "| Value | Images | Head Accepted | None Detected | Not Applicable | From Body | From Original |"
        )
        md_lines.append(
            "| ----- | ------ | ------------- | ------------- | -------------- | --------- | ------------- |"
        )
        for val, stats in sorted(dim_data.items()):
            md_lines.append(
                f"| {val} | {stats.get('n_images', 0)} | "
                f"{stats.get('head_accepted', 0)} | "
                f"{stats.get('none_detected', 0)} | "
                f"{stats.get('not_applicable', 0)} | "
                f"{stats.get('from_body_crop', 0)} | "
                f"{stats.get('from_original', 0)} |"
            )
        md_lines.append("")

    md_path.write_text("\n".join(md_lines))
    logger.info("Markdown report: %s", md_path)

    return json_path, csv_path, md_path


# ---------------------------------------------------------------------------
# Contact-sheet generation
# ---------------------------------------------------------------------------

def _load_thumb(path: str | Path | None) -> Image.Image | None:
    """Load an image thumbnail or return None on any error."""
    if path is None or str(path).strip() == "":
        return None
    try:
        img = Image.open(str(path)).convert("RGB")
        img.thumbnail(_THUMB_SIZE, Image.LANCZOS)
        return img
    except Exception:
        return None


def _placeholder_image(status: str, label: str = "") -> Image.Image:
    """Return a small placeholder image with a status label."""
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


def _make_head_cell(
    source_path: str | None,
    body_crop_row: pd.Series | None,
    head_crop_row: pd.Series | None,
    image_id: str,
    source_root: str | None,
) -> Image.Image:
    """
    Build one head contact-sheet cell:
      slot 0 — source image
      slot 1 — accepted body crop (or placeholder)
      slot 2 — detected head crop (or terminal-status placeholder)
      slot 3 — info card: status / score / box / source_used / crop_id

    Labels use relative paths and crop_ids; no absolute paths.
    """
    thumb_w, thumb_h = _THUMB_SIZE
    n_slots = 4
    cell_w = n_slots * (thumb_w + _MARGIN) + _MARGIN
    cell_h = thumb_h + _LABEL_HEIGHT * 2 + _MARGIN * 2

    cell = Image.new("RGB", (cell_w, cell_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(cell)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    def _paste(img: Image.Image, slot: int, label: str) -> None:
        x = _MARGIN + slot * (thumb_w + _MARGIN)
        y = _MARGIN + _LABEL_HEIGHT
        img_resized = img.resize(_THUMB_SIZE, Image.LANCZOS)
        cell.paste(img_resized, (x, y))
        short = label[:28] + "…" if len(label) > 28 else label
        draw.text((x, _MARGIN), short, fill=(40, 40, 40), font=font)

    # --- Slot 0: source image ---
    src_label = _strip_absolute_prefix(str(source_path or ""), source_root)
    src_label = f"src:{Path(src_label).name}" if src_label else f"id:{image_id[:12]}"
    src_img = _load_thumb(source_path)
    if src_img is None:
        src_img = _placeholder_image("no_source", src_label)
    _paste(src_img, 0, src_label)

    # --- Slot 1: body crop ---
    if body_crop_row is not None:
        b_status = str(body_crop_row.get("detector_status", ""))
        b_score_raw = body_crop_row.get("detector_confidence")
        try:
            b_score = f"{float(b_score_raw):.2f}"
        except (TypeError, ValueError):
            b_score = "N/A"
        if b_status == "accepted":
            body_img = _load_thumb(body_crop_row.get("crop_path"))
            if body_img is None:
                body_img = _placeholder_image("missing_file")
        else:
            body_img = _placeholder_image(b_status)
        _paste(body_img, 1, f"body s={b_score}")
    else:
        _paste(_placeholder_image("no_record"), 1, "body N/A")

    # --- Slot 2: head crop ---
    if head_crop_row is not None:
        h_status = str(head_crop_row.get("detector_status", ""))
        h_score_raw = head_crop_row.get("detector_confidence")
        try:
            h_score = f"{float(h_score_raw):.2f}"
        except (TypeError, ValueError):
            h_score = "N/A"
        if h_status == "accepted":
            head_img = _load_thumb(head_crop_row.get("crop_path"))
            if head_img is None:
                head_img = _placeholder_image("missing_file")
        else:
            head_img = _placeholder_image(h_status)
        _paste(head_img, 2, f"head s={h_score}")
    else:
        _paste(_placeholder_image("no_record"), 2, "head N/A")

    # --- Slot 3: info card ---
    info_img = Image.new("RGB", _THUMB_SIZE, color=(248, 248, 248))
    info_draw = ImageDraw.Draw(info_img)
    try:
        info_font = ImageFont.load_default()
    except Exception:
        info_font = None

    h_status_lbl = "N/A"
    h_score_lbl = "N/A"
    h_box_lbl = "N/A"
    h_src_lbl = "N/A"
    h_crop_id_lbl = "N/A"

    if head_crop_row is not None:
        h_status_lbl = str(head_crop_row.get("detector_status", ""))[:16]
        raw_conf = head_crop_row.get("detector_confidence")
        try:
            h_score_lbl = f"{float(raw_conf):.3f}"
        except (TypeError, ValueError):
            h_score_lbl = "N/A"
        raw_box = head_crop_row.get("detector_box")
        if pd.notna(raw_box) and str(raw_box).strip() not in ("", "None", "nan"):
            try:
                box_list = json.loads(str(raw_box))
                h_box_lbl = f"[{','.join(str(int(v)) for v in box_list)}]"
            except Exception:
                h_box_lbl = str(raw_box)[:20]
        if "source_used" in head_crop_row.index:
            h_src_lbl = str(head_crop_row.get("source_used", ""))[:12]
        raw_cid = head_crop_row.get("crop_id")
        if pd.notna(raw_cid) and str(raw_cid):
            cid_str = str(raw_cid)
            # show last segment only to avoid absolute paths in labels
            h_crop_id_lbl = cid_str.split("__")[-1] if "__" in cid_str else cid_str[-20:]

    info_lines = [
        f"status:{h_status_lbl}",
        f"score:{h_score_lbl}",
        f"box:{h_box_lbl}",
        f"src:{h_src_lbl}",
        f"cid:{h_crop_id_lbl}",
    ]
    y_off = 6
    for line in info_lines:
        info_draw.text((4, y_off), line[:28], fill=(60, 60, 60), font=info_font)
        y_off += 16

    _paste(info_img, 3, "info")

    # Bottom label: image_id (not an absolute path)
    draw.text((_MARGIN, cell_h - _LABEL_HEIGHT), f"id:{image_id[:20]}", fill=(80, 80, 80), font=font)

    return cell


def _build_head_contact_sheet_page(
    cells: list[Image.Image],
    page_num: int,
) -> Image.Image:
    """Arrange head-crop cells into a page image."""
    n = len(cells)
    if n == 0:
        return Image.new("RGB", (100, 100), color=(255, 255, 255))

    cols = min(_PAGE_COLS, n)
    rows = (n + cols - 1) // cols

    cell_w, cell_h = cells[0].size
    page_w = cols * cell_w + (cols + 1) * _MARGIN
    page_h = rows * cell_h + (rows + 1) * _MARGIN + 40

    page = Image.new("RGB", (page_w, page_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(page)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((_MARGIN, _MARGIN), f"BTEH Head-Crop Pilot — Page {page_num}", fill=(40, 40, 40), font=font)

    for idx, cell in enumerate(cells):
        col = idx % cols
        row = idx // cols
        x = _MARGIN + col * (cell_w + _MARGIN)
        y = 40 + _MARGIN + row * (cell_h + _MARGIN)
        page.paste(cell, (x, y))

    return page


def generate_head_contact_sheets(
    pilot_df: pd.DataFrame,
    head_df: pd.DataFrame,
    output_dir: Path,
    *,
    body_df: pd.DataFrame | None = None,
    source_root: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> list[Path]:
    """
    Generate paginated head-crop contact sheets.

    Each image row produces one cell:
      source image | body crop | head crop | info card (status/score/box/source_used/crop_id)

    Labels use relative paths and crop_ids — no absolute paths.

    Returns list of written page paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Index manifests by image_id
    head_idx = head_df.copy()
    head_idx["image_id"] = head_idx["image_id"].astype(str)
    head_idx = head_idx.set_index("image_id")

    body_idx: pd.DataFrame | None = None
    if body_df is not None:
        body_idx = body_df[body_df["crop_kind"] == "body"].copy()
        body_idx["image_id"] = body_idx["image_id"].astype(str)
        body_idx = body_idx.set_index("image_id")

    cells: list[Image.Image] = []
    page_num = 1

    for _, img_row in pilot_df.iterrows():
        image_id = str(img_row["image_id"])

        # Resolve source path (relative label, no absolute paths in cell)
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

        # Get head crop row
        try:
            head_rows = head_idx.loc[[image_id]].reset_index()
            head_row = head_rows.iloc[0] if not head_rows.empty else None
        except KeyError:
            raise ValueError(
                f"Pilot image_id {image_id!r} is missing from the head manifest"
            ) from None

        # Get body crop row (optional)
        body_row = None
        if body_idx is not None:
            try:
                b_rows = body_idx.loc[[image_id]].reset_index()
                accepted_b = b_rows[b_rows["detector_status"] == "accepted"]
                body_row = accepted_b.iloc[0] if not accepted_b.empty else None
            except KeyError:
                pass

        cell = _make_head_cell(source_path, body_row, head_row, image_id, source_root)
        cells.append(cell)

        if len(cells) >= page_size:
            page_img = _build_head_contact_sheet_page(cells, page_num)
            page_path = output_dir / f"head_contact_sheet_page_{page_num:03d}.jpg"
            page_img.save(str(page_path), "JPEG", quality=85)
            logger.info("Head contact sheet page %d written: %s", page_num, page_path)
            written.append(page_path)
            cells = []
            page_num += 1

    if cells:
        page_img = _build_head_contact_sheet_page(cells, page_num)
        page_path = output_dir / f"head_contact_sheet_page_{page_num:03d}.jpg"
        page_img.save(str(page_path), "JPEG", quality=85)
        logger.info("Head contact sheet page %d written: %s", page_num, page_path)
        written.append(page_path)

    return written


# ---------------------------------------------------------------------------
# Sub-command: sample
# ---------------------------------------------------------------------------

def cmd_sample(args: argparse.Namespace) -> int:
    artifact_v_root = BTEH_ARTIFACT_ROOT / ARTIFACT_SCHEMA_VERSION

    pilot_manifest_path = Path(args.pilot_manifest) if args.pilot_manifest else (
        artifact_v_root / "pilot" / BODY_PILOT_MANIFEST_FILENAME
    )

    try:
        pilot_df, sidecar = load_body_pilot_manifest(pilot_manifest_path)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else artifact_v_root / "pilot" / "head"
    )

    sidecar_path = write_head_pilot_sidecar(pilot_df, sidecar, output_dir)

    print(f"Head pilot sidecar written: {sidecar_path}")
    print(f"Reusing {len(pilot_df)} pilot images from body-pilot manifest.")
    return 0


# ---------------------------------------------------------------------------
# Sub-command: report
# ---------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> int:
    artifact_v_root = BTEH_ARTIFACT_ROOT / ARTIFACT_SCHEMA_VERSION

    pilot_manifest_path = Path(args.pilot_manifest) if args.pilot_manifest else (
        artifact_v_root / "pilot" / BODY_PILOT_MANIFEST_FILENAME
    )
    head_manifest_path = Path(args.head_manifest) if args.head_manifest else (
        EXPERIMENT_ROOT / HEAD_MANIFEST_FILENAME
    )

    if not pilot_manifest_path.exists():
        logger.error("Pilot manifest not found: %s. Run 'sample' first.", pilot_manifest_path)
        return 1
    if not head_manifest_path.exists():
        logger.error(
            "Head manifest not found: %s. Run step_1_run_head_detection.py first.",
            head_manifest_path,
        )
        return 1

    try:
        pilot_df, _ = load_body_pilot_manifest(pilot_manifest_path)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    head_df = pd.read_parquet(head_manifest_path)

    # --- Fail-loud checks ---
    _fail_loud_head_schema(head_df)
    _fail_loud_head_crop_kind(head_df)
    _fail_loud_head_joins(pilot_df, head_df)

    if args.check_files:
        _fail_loud_head_accepted_files_exist(head_df, set(pilot_df["image_id"].astype(str)))

    # Optional body manifest
    body_df: pd.DataFrame | None = None
    if args.body_manifest:
        body_manifest_path = Path(args.body_manifest)
        if body_manifest_path.exists():
            body_df = pd.read_parquet(body_manifest_path)
        else:
            logger.warning("Body manifest not found at %s; body-input metrics skipped.", body_manifest_path)

    source_root = args.source_root or str(BTEH_SOURCE_ROOT)
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else artifact_v_root / REPORTS_SUBDIR / "head"
    )
    cs_dir = output_dir / "head_contact_sheets"
    review_csv = Path(args.review_csv) if args.review_csv else None
    gate = float(args.precision_gate)

    metrics = compute_head_metrics(
        pilot_df, head_df, body_df, review_csv_path=review_csv
    )
    write_head_metric_reports(metrics, output_dir)

    page_paths = generate_head_contact_sheets(
        pilot_df, head_df, cs_dir,
        body_df=body_df,
        source_root=source_root,
        page_size=args.page_size,
    )

    print(f"Reports written to: {output_dir}")
    print(f"Head contact sheets ({len(page_paths)} pages): {cs_dir}")

    det = metrics.get("detector", {})
    print(
        f"Detector summary — head coverage: {det.get('head_accepted_coverage', 'N/A')}, "
        f"from body_crop: {det.get('n_from_body_crop', 'N/A')}, "
        f"from original: {det.get('n_from_original_fallback', 'N/A')}"
    )

    # Precision gate
    if review_csv:
        hum = metrics.get("human", {})
        precision = hum.get("precision")
        if not check_precision_gate(metrics, gate):
            print(
                f"ERROR: Precision gate FAILED: {precision} < {gate}. "
                "Head crops do not meet the required visual correctness threshold."
            )
            return 2
        if precision is not None:
            print(f"Precision gate PASSED: {precision} >= {gate}")
        if not hum.get("coverage_complete"):
            missing = hum.get("missing_from_review", [])
            terminal = hum.get("terminal_ids_in_review", [])
            if missing:
                print(
                    f"WARNING: {len(missing)} accepted head crop(s) not yet reviewed."
                )
            if terminal:
                print(
                    f"WARNING: {len(terminal)} terminal-status crop_id(s) in review CSV "
                    "(should be removed)."
                )

    return 0


# ---------------------------------------------------------------------------
# Sub-command: fingerprint
# ---------------------------------------------------------------------------

def cmd_fingerprint(args: argparse.Namespace) -> int:
    artifact_v_root = BTEH_ARTIFACT_ROOT / ARTIFACT_SCHEMA_VERSION

    pilot_manifest_path = Path(args.pilot_manifest) if args.pilot_manifest else (
        artifact_v_root / "pilot" / BODY_PILOT_MANIFEST_FILENAME
    )
    head_manifest_path = Path(args.head_manifest) if args.head_manifest else None

    if not pilot_manifest_path.exists():
        logger.error("Pilot manifest not found: %s", pilot_manifest_path)
        return 1

    try:
        pilot_df, body_sidecar = load_body_pilot_manifest(pilot_manifest_path)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    detector_config = {
        "prompt": args.prompt,
        "conf_threshold": args.conf_threshold,
        "min_area_frac": args.min_area_frac,
        "max_area_frac": args.max_area_frac,
        "min_aspect": args.min_aspect,
        "max_aspect": args.max_aspect,
        "iou_threshold": args.iou_threshold,
        "pad_frac": args.pad_frac,
    }

    fingerprint = compute_detector_config_fingerprint(**detector_config)

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else artifact_v_root / "pilot" / "head"
    )

    existing = load_detector_fingerprint_record(output_dir)
    if existing is not None:
        try:
            _fail_loud_fingerprint_record_match(
                existing,
                fingerprint,
                body_sidecar.get("source_fingerprint"),
                body_sidecar.get("split_fingerprint"),
            )
            print(f"Fingerprint record verified (no change): {fingerprint!r}")
        except ValueError as e:
            logger.error("Fingerprint mismatch: %s", e)
            return 1
    else:
        fp_path = write_detector_fingerprint_record(
            fingerprint, detector_config, pilot_df, body_sidecar, output_dir
        )
        print(f"Detector fingerprint record written: {fp_path}")
        print(f"Fingerprint: {fingerprint!r}")

    # Optionally validate head manifest against fingerprint
    if head_manifest_path and head_manifest_path.exists():
        head_df = pd.read_parquet(head_manifest_path)
        _fail_loud_head_schema(head_df)
        try:
            _fail_loud_head_detector_fingerprint(head_df, fingerprint)
            print("Head manifest detector_fingerprint: OK")
        except ValueError as e:
            logger.error("%s", e)
            return 1

        try:
            _fail_loud_head_source_split_fingerprints(
                head_df,
                {
                    "source_fingerprint": body_sidecar.get("source_fingerprint"),
                    "split_fingerprint": body_sidecar.get("split_fingerprint"),
                },
            )
            print("Head manifest source/split fingerprints: OK")
        except ValueError as e:
            logger.error("%s", e)
            return 1

        _fail_loud_head_joins(pilot_df, head_df)

        if args.check_files:
            pilot_ids = set(pilot_df["image_id"].astype(str))
            try:
                _fail_loud_head_accepted_files_exist(head_df, pilot_ids)
                print("All accepted head crop files present: OK")
            except ValueError as e:
                logger.error("%s", e)
                return 1

    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bteh_head_crop_pilot",
        description=dedent("""\
            BTEH head-crop quality pilot tooling.

            Sub-commands:
              sample      Validate / record the fixed-120 pilot selection sidecar.
              report      Generate head contact sheets and quality metric reports.
              fingerprint Write or verify the content-addressed detector config fingerprint.

            This tool never alters the production selected-v1 body/ear crop
            manifests or the body pilot report.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- sample ---
    sp = sub.add_parser(
        "sample",
        help="Record head-pilot sidecar (reuses fixed-120 body-pilot manifest).",
        description=dedent("""\
            Loads the existing fixed-120 named body-pilot manifest and writes a
            head-specific sidecar JSON to --output-dir.  Does not alter the body
            pilot manifest or run any model inference.
        """),
    )
    sp.add_argument(
        "--pilot-manifest", default=None,
        help="Path to bteh_pilot_manifest.parquet (default: <artifact-root>/v1/pilot/)",
    )
    sp.add_argument(
        "--output-dir", default=None,
        help="Output directory for head pilot sidecar (default: <artifact-root>/v1/pilot/head/)",
    )
    sp.add_argument("--verbose", "-v", action="store_true")

    # --- report ---
    rp = sub.add_parser(
        "report",
        help="Generate head contact sheets and quality metric reports.",
        description=dedent("""\
            Consumes the body-pilot manifest and a head experiment manifest to produce:
              * Head contact sheets (source | body | head | info card per image)
              * JSON / CSV / Markdown metric reports
              * Optional visual review precision gate enforcement (default >=0.95)

            Contact-sheet labels use relative paths and crop_ids; no absolute paths
            are embedded.  Visual correctness (human review) is kept separate from
            detector confidence metrics.
        """),
    )
    rp.add_argument(
        "--pilot-manifest", default=None,
        help="Path to bteh_pilot_manifest.parquet",
    )
    rp.add_argument(
        "--head-manifest", default=None,
        help="Path to head_manifest.parquet (experiment output)",
    )
    rp.add_argument(
        "--body-manifest", default=None,
        help="Optional path to body crop manifest (enables body-input coverage metrics)",
    )
    rp.add_argument(
        "--source-root", default=None,
        help="BTEH source image root for loading thumbnails",
    )
    rp.add_argument(
        "--output-dir", default=None,
        help="Output directory for reports and contact sheets",
    )
    rp.add_argument(
        "--review-csv", default=None,
        help=(
            "Optional human-review CSV with columns: crop_id, status "
            "(accepted/rejected/uncertain), reason (optional). "
            "Must cover each accepted head crop exactly once; terminal-status "
            "crop_ids must not appear."
        ),
    )
    rp.add_argument(
        "--precision-gate", type=float, default=DEFAULT_PRECISION_GATE,
        help=(
            f"Minimum required decisive precision (default: {DEFAULT_PRECISION_GATE}). "
            "Exit code 2 if gate fails."
        ),
    )
    rp.add_argument(
        "--page-size", type=int, default=DEFAULT_PAGE_SIZE,
        help=f"Images per contact-sheet page (default: {DEFAULT_PAGE_SIZE})",
    )
    rp.add_argument(
        "--check-files", action="store_true",
        help="Fail if accepted head crop files are missing from disk",
    )
    rp.add_argument("--verbose", "-v", action="store_true")

    # --- fingerprint ---
    fp = sub.add_parser(
        "fingerprint",
        help="Write or verify content-addressed detector config fingerprint.",
        description=dedent("""\
            Computes a short SHA-256 fingerprint of the head detector hyperparameters
            and writes a JSON record to --output-dir.

            On subsequent runs, verifies that the detector config, source manifest,
            and split manifest all match the saved record; fails if any mismatch is
            detected.

            Optionally validates an existing head manifest against the fingerprint.
        """),
    )
    fp.add_argument(
        "--pilot-manifest", default=None,
        help="Path to bteh_pilot_manifest.parquet",
    )
    fp.add_argument(
        "--head-manifest", default=None,
        help="Optional path to head_manifest.parquet for validation",
    )
    fp.add_argument(
        "--output-dir", default=None,
        help="Output directory for fingerprint record (default: <artifact-root>/v1/pilot/head/)",
    )
    fp.add_argument(
        "--check-files", action="store_true",
        help="Also verify accepted head crop files exist on disk",
    )
    fp.add_argument("--conf-threshold", type=float, default=0.30,
                    help="Detector confidence threshold (default: 0.30)")
    fp.add_argument("--min-area-frac", type=float, default=0.02,
                    help="Minimum head area fraction (default: 0.02)")
    fp.add_argument("--max-area-frac", type=float, default=0.70,
                    help="Maximum head area fraction (default: 0.70)")
    fp.add_argument("--min-aspect", type=float, default=0.4,
                    help="Minimum head aspect ratio (default: 0.4)")
    fp.add_argument("--max-aspect", type=float, default=2.5,
                    help="Maximum head aspect ratio (default: 2.5)")
    fp.add_argument("--iou-threshold", type=float, default=0.50,
                    help="NMS IoU threshold (default: 0.50)")
    fp.add_argument("--pad-frac", type=float, default=0.10,
                    help="Head crop padding fraction (default: 0.10)")
    fp.add_argument("--prompt", default="elephant head.",
                    help="GroundingDINO text prompt (default: elephant head.)")
    fp.add_argument("--verbose", "-v", action="store_true")

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
    elif args.command == "fingerprint":
        return cmd_fingerprint(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
