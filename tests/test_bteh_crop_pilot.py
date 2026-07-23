# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Synthetic tests for pipeline/bteh_crop_pilot.py.

Covers:
  - Determinism: same seed → same pilot selection
  - Seed variation: different seeds → different selections
  - Strata inclusion: all required strata represented
  - Unresolved separation: unresolved/review_required never in named pilot
  - Contact-sheet generation for 0, 1, and 2 ears
  - Terminal placeholder rows (no crop file needed)
  - Missing accepted files error (fail-loud)
  - Fingerprint mismatch error
  - Human-review precision computation
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from itertools import count as _count

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("BTEH_SOURCE_ROOT", "/nonexistent/BTEH")
os.environ.setdefault("BTEH_ARTIFACT_ROOT", "/nonexistent/artifacts")
os.environ.setdefault("data_root_abs_path", "/tmp/test_data")
os.environ.setdefault("container_name", "test_container")

from configs.config_bteh import ARTIFACT_SCHEMA_VERSION
from pipeline.bteh_crop_pilot import (
    DEFAULT_PILOT_N,
    DEFAULT_REVIEW_N,
    DEFAULT_SEED,
    _fail_loud_accepted_files_exist,
    _fail_loud_crop_fingerprints,
    _fail_loud_fingerprint,
    _fail_loud_joins,
    _fail_loud_schema,
    _stratified_sample,
    _assign_strata,
    compute_crop_metrics,
    generate_contact_sheets,
    select_pilot_sample,
    write_pilot_manifest,
    write_metric_reports,
)
from utils.artifact_schema import CROP_MANIFEST_COLUMNS, make_crop_id


# ---------------------------------------------------------------------------
# Synthetic manifest builders
# ---------------------------------------------------------------------------

_COLOR_COUNTER = _count(50)


def _unique_image(path: Path, size=(24, 24)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = next(_COLOR_COUNTER)
    color = (c % 256, (c * 5) % 256, (c * 11) % 256)
    Image.new("RGB", size, color=color).save(str(path), "JPEG")


def _make_manifest(
    n_named: int = 8,
    sessions_per: int = 2,
    with_unresolved: int = 3,
    with_review_required: int = 2,
    include_ref: int = 2,
) -> pd.DataFrame:
    """
    Build a synthetic canonical manifest with a mix of:
      - named identities (included)
      - unresolved rows
      - review_required rows
      - ref-origin rows
    """
    rows: list[dict] = []
    img_counter = _count(1)

    years = ["2021", "2022", "2023", None]
    session_sources = ["exif", "folder", "year_folder"]
    widths = [320, 1280, 3840]
    heights = [240, 720, 2160]

    for i in range(n_named):
        ind_id = f"bteh_ident_{i}"
        for s in range(sessions_per):
            idx = next(img_counter)
            w = widths[idx % len(widths)]
            h = heights[idx % len(heights)]
            rows.append({
                "image_id": f"img_{idx:04d}",
                "individual_id": ind_id,
                "individual_name": f"Ident {i}",
                "herd": None,
                "source_relative_path": f"Ident_{i}/sess_{s}/img_{idx}.jpg",
                "content_hash": f"hash_{idx:04d}",
                "perceptual_hash": None,
                "image_id_path_component": f"p{idx:04d}",
                "image_id_content_component": f"c{idx:04d}",
                "session_id": f"{ind_id}_sess_{s}",
                "capture_date": None,
                "year": years[idx % len(years)],
                "session_source": session_sources[idx % len(session_sources)],
                "dataset_role": "source",
                "include_status": "included",
                "exclusion_reason": None,
                "duplicate_of": None,
                "review_flag": False,
                "review_reason": None,
                "body_crop_status": "pending",
                "ear_detection_status": "pending",
                "image_width": w,
                "image_height": h,
            })

    for i in range(include_ref):
        idx = next(img_counter)
        rows.append({
            "image_id": f"img_{idx:04d}",
            "individual_id": f"bteh_ident_{i}",
            "individual_name": f"Ident {i}",
            "herd": None,
            "source_relative_path": f"ref/Ident_{i}/ref_{idx}.jpg",
            "content_hash": f"hash_ref_{idx:04d}",
            "perceptual_hash": None,
            "image_id_path_component": f"pr{idx:04d}",
            "image_id_content_component": f"cr{idx:04d}",
            "session_id": f"bteh_ident_{i}_ref",
            "capture_date": None,
            "year": "2022",
            "session_source": "folder",
            "dataset_role": "ref",
            "include_status": "included",
            "exclusion_reason": None,
            "duplicate_of": None,
            "review_flag": False,
            "review_reason": None,
            "body_crop_status": "pending",
            "ear_detection_status": "pending",
            "image_width": 640,
            "image_height": 480,
        })

    for i in range(with_unresolved):
        idx = next(img_counter)
        rows.append({
            "image_id": f"img_{idx:04d}",
            "individual_id": "unresolved",
            "individual_name": None,
            "herd": None,
            "source_relative_path": f"uuid-dir-{i}/img_{idx}.jpg",
            "content_hash": f"hash_uuid_{idx:04d}",
            "perceptual_hash": None,
            "image_id_path_component": f"pu{idx:04d}",
            "image_id_content_component": f"cu{idx:04d}",
            "session_id": f"unresolved_sess_{i}",
            "capture_date": None,
            "year": None,
            "session_source": "folder",
            "dataset_role": "source",
            "include_status": "review_required",
            "exclusion_reason": None,
            "duplicate_of": None,
            "review_flag": True,
            "review_reason": "uuid_dir_unresolved",
            "body_crop_status": "pending",
            "ear_detection_status": "pending",
            "image_width": 640,
            "image_height": 480,
        })

    for i in range(with_review_required):
        idx = next(img_counter)
        rows.append({
            "image_id": f"img_{idx:04d}",
            "individual_id": f"bteh_ident_{i}",
            "individual_name": f"Ident {i}",
            "herd": None,
            "source_relative_path": f"Ident_{i}/review/img_{idx}.jpg",
            "content_hash": f"hash_review_{idx:04d}",
            "perceptual_hash": None,
            "image_id_path_component": f"pv{idx:04d}",
            "image_id_content_component": f"cv{idx:04d}",
            "session_id": f"bteh_ident_{i}_review",
            "capture_date": None,
            "year": "2023",
            "session_source": "folder",
            "dataset_role": "source",
            "include_status": "included",
            "exclusion_reason": None,
            "duplicate_of": None,
            "review_flag": True,
            "review_reason": "ambiguous_identity;",
            "body_crop_status": "pending",
            "ear_detection_status": "pending",
            "image_width": 640,
            "image_height": 480,
        })

    return pd.DataFrame(rows)


def _make_crop_manifest(
    image_ids: list[str],
    individual_id: str = "bteh_ident_0",
    *,
    n_ears: int = 2,
    body_status: str = "accepted",
    ear_status: str = "accepted",
    crop_root: str | None = None,
) -> pd.DataFrame:
    """Build a synthetic crop manifest for the given image_ids."""
    rows: list[dict] = []

    def _path(image_id: str, kind: str, ordinal: int) -> str:
        if crop_root:
            return str(Path(crop_root) / f"{make_crop_id(image_id, kind, ordinal)}.jpg")
        return f"/fake/crops/{make_crop_id(image_id, kind, ordinal)}.jpg"

    for image_id in image_ids:
        rows.append({
            "crop_id": make_crop_id(image_id, "body", 0),
            "image_id": image_id,
            "individual_id": individual_id,
            "crop_kind": "body",
            "crop_ordinal": 0,
            "crop_path": _path(image_id, "body", 0),
            "detector_confidence": 0.9,
            "detector_box": "[0,0,100,100]",
            "detector_status": body_status,
            "review_status": "pending",
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "source_fingerprint": None,
            "split_fingerprint": None,
        })

        for ear_ordinal in range(min(n_ears, 2)):
            rows.append({
                "crop_id": make_crop_id(image_id, "ear", ear_ordinal),
                "image_id": image_id,
                "individual_id": individual_id,
                "crop_kind": "ear",
                "crop_ordinal": ear_ordinal,
                "crop_path": _path(image_id, "ear", ear_ordinal),
                "detector_confidence": 0.85 - ear_ordinal * 0.05,
                "detector_box": "[0,0,50,50]",
                "detector_status": ear_status,
                "review_status": "pending",
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_fingerprint": None,
                "split_fingerprint": None,
            })

        # Fill in terminal placeholders for missing ears
        found = set(range(min(n_ears, 2)))
        for ordinal in [0, 1]:
            if ordinal not in found:
                status = "not_applicable" if ordinal == 1 and 0 not in found else "none_detected"
                rows.append({
                    "crop_id": make_crop_id(image_id, "ear", ordinal),
                    "image_id": image_id,
                    "individual_id": individual_id,
                    "crop_kind": "ear",
                    "crop_ordinal": ordinal,
                    "crop_path": None,
                    "detector_confidence": None,
                    "detector_box": None,
                    "detector_status": status,
                    "review_status": "pending",
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "source_fingerprint": None,
                    "split_fingerprint": None,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Determinism: same seed → same selection
# ---------------------------------------------------------------------------

def test_determinism():
    manifest = _make_manifest(n_named=10, sessions_per=3)
    pilot1, _, _ = select_pilot_sample(manifest, seed=42, n_pilot=12, n_review=3)
    pilot2, _, _ = select_pilot_sample(manifest, seed=42, n_pilot=12, n_review=3)
    assert list(pilot1["image_id"]) == list(pilot2["image_id"]), (
        "Same seed must produce identical pilot selection"
    )


# ---------------------------------------------------------------------------
# 2. Seed variation: different seeds → different selections
# ---------------------------------------------------------------------------

def test_seed_variation():
    manifest = _make_manifest(n_named=10, sessions_per=4)
    pilot_42, _, _ = select_pilot_sample(manifest, seed=42, n_pilot=12, n_review=3)
    pilot_99, _, _ = select_pilot_sample(manifest, seed=99, n_pilot=12, n_review=3)
    ids_42 = set(pilot_42["image_id"])
    ids_99 = set(pilot_99["image_id"])
    assert ids_42 != ids_99, "Different seeds must produce different selections"


# ---------------------------------------------------------------------------
# 3. Strata inclusion: multiple identities, years, origins present
# ---------------------------------------------------------------------------

def test_strata_inclusion():
    manifest = _make_manifest(n_named=6, sessions_per=3, include_ref=3)
    pilot, _, report = select_pilot_sample(manifest, seed=42, n_pilot=20, n_review=0)

    # Multiple identities should appear
    assert pilot["individual_id"].nunique() > 1, "Pilot must include multiple identities"

    # Both ref and regular origins should appear (if enough ref rows)
    origins = set(pilot["dataset_role"].fillna("source").unique())
    assert "ref" in origins or len(pilot) < 6, (
        "Ref-origin rows should appear in pilot when available"
    )

    # Report should carry strata metadata
    assert "strata_counts" in report
    assert report["n_identities"] >= 1


# ---------------------------------------------------------------------------
# 4. Unresolved / review_required rows never enter named pilot sample
# ---------------------------------------------------------------------------

def test_unresolved_not_in_named_pilot():
    manifest = _make_manifest(n_named=6, sessions_per=2, with_unresolved=4,
                               with_review_required=0)
    pilot, review, _ = select_pilot_sample(manifest, seed=42, n_pilot=10, n_review=5)

    unresolved_in_pilot = pilot[
        pilot["individual_id"].astype(str) == "unresolved"
    ]
    assert len(unresolved_in_pilot) == 0, (
        "Unresolved rows must never appear in the named pilot sample"
    )

    # Review sample should draw from review_required / flagged rows
    assert len(review) >= 0  # may be 0 if no flagged rows match
    if len(review) > 0:
        # All review rows must be from unresolved/review_required pool
        # (none may be named-eligible rows in this fixture since with_review_required=0)
        review_unresolved = review[review["individual_id"].astype(str) == "unresolved"]
        assert len(review_unresolved) == len(review), (
            "Review rows should be unresolved when with_review_required=0"
        )


def test_review_required_not_assigned_identity():
    manifest = _make_manifest(n_named=4, sessions_per=2, with_unresolved=3)
    pilot, review, _ = select_pilot_sample(manifest, seed=42, n_pilot=8, n_review=3)

    # Unresolved rows must not have an identity assigned in the pilot
    assert "unresolved" not in pilot["individual_id"].tolist()


# ---------------------------------------------------------------------------
# 5. Contact-sheet: 0 ears
# ---------------------------------------------------------------------------

def test_contact_sheet_zero_ears(tmp_path):
    manifest = _make_manifest(n_named=2, sessions_per=1)
    pilot, _, _ = select_pilot_sample(manifest, seed=42, n_pilot=2, n_review=0)
    image_ids = pilot["image_id"].tolist()

    # Build crop manifest with 0 ears (terminal placeholders)
    crop_df = _make_crop_manifest(image_ids, n_ears=0, body_status="accepted", ear_status="none_detected")

    cs_dir = tmp_path / "cs_zero_ears"
    pages = generate_contact_sheets(
        pilot, crop_df, cs_dir, source_root=None, page_size=4
    )
    assert len(pages) >= 1
    assert all(p.exists() for p in pages)


# ---------------------------------------------------------------------------
# 6. Contact-sheet: 1 ear
# ---------------------------------------------------------------------------

def test_contact_sheet_one_ear(tmp_path):
    manifest = _make_manifest(n_named=2, sessions_per=1)
    pilot, _, _ = select_pilot_sample(manifest, seed=42, n_pilot=2, n_review=0)
    image_ids = pilot["image_id"].tolist()

    # Create real crop files for the accepted body crops
    crop_root = str(tmp_path / "crops")
    crop_df = _make_crop_manifest(
        image_ids, n_ears=1, body_status="accepted", ear_status="accepted",
        crop_root=crop_root,
    )
    Path(crop_root).mkdir(parents=True, exist_ok=True)
    for _, row in crop_df[crop_df["detector_status"] == "accepted"].iterrows():
        p = Path(str(row["crop_path"]))
        p.parent.mkdir(parents=True, exist_ok=True)
        _unique_image(p)

    cs_dir = tmp_path / "cs_one_ear"
    pages = generate_contact_sheets(
        pilot, crop_df, cs_dir, source_root=None, page_size=4
    )
    assert len(pages) >= 1


# ---------------------------------------------------------------------------
# 7. Contact-sheet: 2 ears
# ---------------------------------------------------------------------------

def test_contact_sheet_two_ears(tmp_path):
    manifest = _make_manifest(n_named=2, sessions_per=1)
    pilot, _, _ = select_pilot_sample(manifest, seed=42, n_pilot=2, n_review=0)
    image_ids = pilot["image_id"].tolist()

    crop_root = str(tmp_path / "crops")
    crop_df = _make_crop_manifest(
        image_ids, n_ears=2, body_status="accepted", ear_status="accepted",
        crop_root=crop_root,
    )
    Path(crop_root).mkdir(parents=True, exist_ok=True)
    for _, row in crop_df[crop_df["detector_status"] == "accepted"].iterrows():
        p = Path(str(row["crop_path"]))
        p.parent.mkdir(parents=True, exist_ok=True)
        _unique_image(p)

    cs_dir = tmp_path / "cs_two_ears"
    pages = generate_contact_sheets(
        pilot, crop_df, cs_dir, source_root=None, page_size=4
    )
    assert len(pages) >= 1


# ---------------------------------------------------------------------------
# 8. Terminal placeholder rows: none_detected / not_applicable
# ---------------------------------------------------------------------------

def test_terminal_placeholder_rows(tmp_path):
    manifest = _make_manifest(n_named=2, sessions_per=1)
    pilot, _, _ = select_pilot_sample(manifest, seed=42, n_pilot=2, n_review=0)
    image_ids = pilot["image_id"].tolist()

    # 0 ears: ear_0=none_detected, ear_1=not_applicable; body=none_detected
    crop_df = _make_crop_manifest(
        image_ids, n_ears=0, body_status="none_detected", ear_status="none_detected"
    )

    # none_detected rows must have no crop_path requirement
    placeholder_rows = crop_df[crop_df["detector_status"].isin({"none_detected", "not_applicable"})]
    assert len(placeholder_rows) > 0, "Expected terminal placeholder rows"

    # Fail-loud join check must pass (all images covered)
    _fail_loud_joins(pilot, crop_df)

    # accepted_files_exist must pass (no accepted rows → nothing to check)
    _fail_loud_accepted_files_exist(crop_df, set(image_ids))


# ---------------------------------------------------------------------------
# 9. Missing accepted files → error
# ---------------------------------------------------------------------------

def test_missing_accepted_files_raises(tmp_path):
    manifest = _make_manifest(n_named=2, sessions_per=1)
    pilot, _, _ = select_pilot_sample(manifest, seed=42, n_pilot=2, n_review=0)
    image_ids = pilot["image_id"].tolist()

    # Accepted crops with non-existent file paths
    crop_root = str(tmp_path / "crops_missing")
    crop_df = _make_crop_manifest(
        image_ids, n_ears=2, body_status="accepted", ear_status="accepted",
        crop_root=crop_root,
    )
    # Do NOT create the files → should raise

    with pytest.raises(ValueError, match="accepted crop file"):
        _fail_loud_accepted_files_exist(crop_df, set(image_ids))


# ---------------------------------------------------------------------------
# 10. Fingerprint mismatch → error
# ---------------------------------------------------------------------------

def test_fingerprint_mismatch_raises():
    manifest = _make_manifest(n_named=4, sessions_per=2)
    pilot, _, _ = select_pilot_sample(manifest, seed=42, n_pilot=5)

    # Tamper with the sidecar fingerprint
    tampered_sidecar = {
        "pilot_fingerprint": "deadbeef" * 8,  # clearly wrong
    }
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _fail_loud_fingerprint(pilot, tampered_sidecar)


def test_fingerprint_match_passes():
    from pipeline.bteh_crop_pilot import fingerprint_dataframe
    manifest = _make_manifest(n_named=4, sessions_per=2)
    pilot, _, _ = select_pilot_sample(manifest, seed=42, n_pilot=5)

    fp = fingerprint_dataframe(pilot, "image_id")
    sidecar = {"pilot_fingerprint": fp}
    # Should not raise
    _fail_loud_fingerprint(pilot, sidecar)


def test_crop_fingerprint_mismatch_raises():
    crop_df = _make_crop_manifest(["img_1"], n_ears=0)
    crop_df["source_fingerprint"] = "source-old"
    crop_df["split_fingerprint"] = "split-current"
    with pytest.raises(ValueError, match="source_fingerprint mismatch"):
        _fail_loud_crop_fingerprints(
            crop_df,
            {
                "source_fingerprint": "source-current",
                "split_fingerprint": "split-current",
            },
        )


# ---------------------------------------------------------------------------
# 11. Missing join in crop manifest → error
# ---------------------------------------------------------------------------

def test_missing_join_raises():
    manifest = _make_manifest(n_named=2, sessions_per=1)
    pilot, _, _ = select_pilot_sample(manifest, seed=42, n_pilot=2, n_review=0)

    # Empty crop manifest → all pilot image_ids are missing
    empty_crop = pd.DataFrame(columns=CROP_MANIFEST_COLUMNS)

    with pytest.raises(ValueError, match="no entry in the crop manifest"):
        _fail_loud_joins(pilot, empty_crop)


# ---------------------------------------------------------------------------
# 12. Human-review precision
# ---------------------------------------------------------------------------

def test_human_review_precision(tmp_path):
    manifest = _make_manifest(n_named=4, sessions_per=2)
    pilot, _, _ = select_pilot_sample(manifest, seed=42, n_pilot=6)
    image_ids = pilot["image_id"].tolist()
    crop_df = _make_crop_manifest(image_ids, n_ears=2)

    # Write a fake review CSV: mark half as accepted, half as rejected
    crop_ids = crop_df["crop_id"].tolist()
    n = len(crop_ids)
    statuses = ["accepted"] * (n // 2) + ["rejected"] * (n - n // 2)
    review_path = tmp_path / "review.csv"
    with open(review_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_id", "status"])
        writer.writeheader()
        for crop_id, status in zip(crop_ids, statuses):
            writer.writerow({"crop_id": crop_id, "status": status})

    metrics = compute_crop_metrics(pilot, crop_df, review_csv_path=review_path)
    hum = metrics["human"]
    assert hum["precision"] is not None
    assert 0.0 <= hum["precision"] <= 1.0
    assert hum["n_reviewed"] > 0
    assert hum["n_accepted"] + hum["n_rejected"] + hum["n_uncertain"] == hum["n_reviewed"]
    assert set(hum["precision_by_kind"]) == {"body", "ear"}


def test_human_review_accepts_matching_crop_kind_column(tmp_path):
    crop_df = _make_crop_manifest(["img_1"], n_ears=1)
    accepted = crop_df[crop_df["detector_status"] == "accepted"]
    review_path = tmp_path / "review_with_kind.csv"
    accepted[["crop_id", "crop_kind"]].assign(status="accepted").to_csv(
        review_path,
        index=False,
    )
    metrics = compute_crop_metrics(
        pd.DataFrame({"image_id": ["img_1"]}),
        crop_df,
        review_csv_path=review_path,
    )
    assert set(metrics["human"]["precision_by_kind"]) == {"body", "ear"}


def test_human_review_excludes_terminal_placeholders(tmp_path):
    crop_df = _make_crop_manifest(["img_1"], n_ears=0)
    body_id = crop_df.loc[crop_df["detector_status"] == "accepted", "crop_id"].iloc[0]
    terminal_id = crop_df.loc[
        crop_df["detector_status"] == "none_detected", "crop_id"
    ].iloc[0]
    review_path = tmp_path / "review.csv"
    pd.DataFrame(
        [
            {"crop_id": body_id, "status": "accepted"},
            {"crop_id": terminal_id, "status": "rejected"},
        ]
    ).to_csv(review_path, index=False)

    metrics = compute_crop_metrics(
        pd.DataFrame({"image_id": ["img_1"]}),
        crop_df,
        review_csv_path=review_path,
    )
    assert metrics["human"]["n_reviewed"] == 1
    assert metrics["human"]["precision"] == 1.0


def test_human_review_no_csv():
    manifest = _make_manifest(n_named=2, sessions_per=1)
    pilot, _, _ = select_pilot_sample(manifest, seed=42, n_pilot=2)
    image_ids = pilot["image_id"].tolist()
    crop_df = _make_crop_manifest(image_ids, n_ears=1)

    metrics = compute_crop_metrics(pilot, crop_df, review_csv_path=None)
    assert metrics["human"]["precision"] is None
    assert "No human review CSV" in metrics["human"]["note"]


def test_human_review_invalid_status(tmp_path):
    manifest = _make_manifest(n_named=2, sessions_per=1)
    pilot, _, _ = select_pilot_sample(manifest, seed=42, n_pilot=2)
    image_ids = pilot["image_id"].tolist()
    crop_df = _make_crop_manifest(image_ids)

    review_path = tmp_path / "bad_review.csv"
    with open(review_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_id", "status"])
        writer.writeheader()
        writer.writerow({"crop_id": crop_df["crop_id"].iloc[0], "status": "INVALID"})

    with pytest.raises(ValueError, match="invalid status values"):
        compute_crop_metrics(pilot, crop_df, review_csv_path=review_path)


# ---------------------------------------------------------------------------
# 13. Schema validation: missing column → error
# ---------------------------------------------------------------------------

def test_schema_missing_column_raises():
    bad_df = pd.DataFrame({"crop_id": ["x"], "image_id": ["y"]})  # missing many cols
    with pytest.raises(ValueError, match="missing required columns"):
        _fail_loud_schema(bad_df)


def test_schema_full_columns_passes():
    # Build a minimal crop df with all required columns
    row = {col: None for col in CROP_MANIFEST_COLUMNS}
    row["crop_id"] = "c0"
    row["image_id"] = "i0"
    _fail_loud_schema(pd.DataFrame([row]))


# ---------------------------------------------------------------------------
# 14. write_pilot_manifest round-trip
# ---------------------------------------------------------------------------

def test_write_pilot_manifest_roundtrip(tmp_path):
    manifest = _make_manifest(n_named=4, sessions_per=2)
    pilot, review, report = select_pilot_sample(manifest, seed=42, n_pilot=6, n_review=2)

    pq_path, sc_path = write_pilot_manifest(
        pilot, review, report, tmp_path,
        source_fingerprint="sfp", split_fingerprint=None, seed=42,
    )
    assert pq_path.exists()
    assert sc_path.exists()

    reloaded = pd.read_parquet(pq_path)
    assert "image_id" in reloaded.columns
    assert "_pilot_role" in reloaded.columns
    # Internal strata columns should be stripped
    internal_cols = {"_stratum", "_split", "_year", "_session_src", "_size_bucket",
                     "_aspect_bucket", "_origin", "_viewpoint"}
    assert internal_cols.isdisjoint(set(reloaded.columns)), (
        "Internal stratum columns must not appear in the written parquet"
    )

    sidecar = json.loads(sc_path.read_text())
    assert sidecar["seed"] == 42
    assert "selected_image_ids" in sidecar
    assert len(sidecar["selected_image_ids"]) == len(pilot)


# ---------------------------------------------------------------------------
# 15. Metric reports written to disk
# ---------------------------------------------------------------------------

def test_metric_reports_written(tmp_path):
    manifest = _make_manifest(n_named=4, sessions_per=2)
    pilot, _, _ = select_pilot_sample(manifest, seed=42, n_pilot=6)
    image_ids = pilot["image_id"].tolist()
    crop_df = _make_crop_manifest(image_ids, n_ears=2)

    metrics = compute_crop_metrics(pilot, crop_df)
    json_path, csv_path, md_path = write_metric_reports(metrics, tmp_path)

    assert json_path.exists()
    assert csv_path.exists()
    assert md_path.exists()

    # JSON must be parseable
    data = json.loads(json_path.read_text())
    assert "detector" in data
    assert "human" in data

    # Markdown must contain key headings
    md_text = md_path.read_text()
    assert "BTEH Crop Pilot" in md_text
    assert "Detector Metrics" in md_text


# ---------------------------------------------------------------------------
# 16. Empty pilot → no crash
# ---------------------------------------------------------------------------

def test_select_sample_no_eligible():
    # All rows are unresolved/review_required, no named eligible rows
    manifest = _make_manifest(
        n_named=0, sessions_per=0,
        with_unresolved=2, with_review_required=0, include_ref=0,
    )
    with pytest.raises((ValueError, Exception)):
        select_pilot_sample(manifest, seed=42, n_pilot=10)


# ---------------------------------------------------------------------------
# 17. _stratified_sample: n=0 returns empty
# ---------------------------------------------------------------------------

def test_stratified_sample_zero():
    manifest = _make_manifest(n_named=3, sessions_per=2)
    df = _assign_strata(manifest[manifest["include_status"] == "included"].copy())
    result = _stratified_sample(df, 0, np.random.default_rng(0))
    assert len(result) == 0


# ---------------------------------------------------------------------------
# 18. n_pilot > eligible → select all eligible without crash
# ---------------------------------------------------------------------------

def test_select_more_than_eligible():
    manifest = _make_manifest(n_named=2, sessions_per=2)
    pilot, _, report = select_pilot_sample(manifest, seed=42, n_pilot=1000, n_review=0)
    n_eligible = len(
        manifest[
            manifest["include_status"].isin({"included", "duplicate_primary"})
            & (manifest["individual_id"].astype(str) != "unresolved")
        ]
    )
    assert len(pilot) <= n_eligible, "Cannot select more than eligible rows"
    assert len(pilot) > 0


# ---------------------------------------------------------------------------
# 19. Contact sheet: source path not exposed in label
# ---------------------------------------------------------------------------

def test_contact_sheet_no_absolute_path_in_label(tmp_path):
    """
    Verify contact-sheet page files don't embed participant-specific paths
    in their PIL-rendered text by inspecting the PIL text calls indirectly
    — we check that source_root stripping is applied.
    """
    from pipeline.bteh_crop_pilot import _strip_absolute_prefix
    abs_path = "/workspace/someuser/projects/BTEH/Ident_0/sess_0/img_001.jpg"
    source_root = "/workspace/someuser/projects/BTEH"
    result = _strip_absolute_prefix(abs_path, source_root)
    assert not result.startswith("/workspace/"), (
        f"Absolute path not stripped: {result!r}"
    )
    assert "img_001.jpg" in result


# ---------------------------------------------------------------------------
# 20. Duplicate_primary rows enter the pilot but review_required rows don't
# ---------------------------------------------------------------------------

def test_duplicate_primary_eligible():
    manifest = _make_manifest(n_named=4, sessions_per=2)
    # Mark first eligible row as duplicate_primary
    eligible_idx = manifest[manifest["include_status"] == "included"].index[0]
    manifest.loc[eligible_idx, "include_status"] = "duplicate_primary"

    pilot, _, _ = select_pilot_sample(manifest, seed=42, n_pilot=20)
    # The duplicate_primary row should be in the pool (not excluded)
    eligible_ids = set(
        manifest[
            manifest["include_status"].isin({"included", "duplicate_primary"})
            & (manifest["individual_id"].astype(str) != "unresolved")
        ]["image_id"]
    )
    for img_id in pilot["image_id"]:
        assert img_id in eligible_ids, f"Non-eligible image {img_id} in pilot"
