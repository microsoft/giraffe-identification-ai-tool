# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Synthetic tests for pipeline/bteh_head_crop_pilot.py.

Covers:
  - 0 head crops (all none_detected / not_applicable)
  - 1 head crop from body_crop source
  - 1 head crop from original (fallback) source
  - Contact-sheet generation for 0 / 1 head
  - Missing accepted head file → error
  - Detector fingerprint write / verify / mismatch
  - Source/split fingerprint mismatch in head manifest
  - Review coverage exactness: every accepted crop reviewed exactly once
  - Terminal IDs must not appear in review CSV (raises on duplicate only; warns otherwise)
  - Precision gate pass (>= threshold) and fail (< threshold)
  - Output isolation: head pilot tool does not touch body pilot report files
  - Head manifest crop_kind validation
  - Fallback source_used metrics
"""

from __future__ import annotations

import csv
import json
import os
import sys
from itertools import count as _count
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("BTEH_SOURCE_ROOT", "/nonexistent/BTEH")
os.environ.setdefault("BTEH_ARTIFACT_ROOT", "/nonexistent/artifacts")
os.environ.setdefault("data_root_abs_path", "/fakedir/test_data")
os.environ.setdefault("container_name", "test_container")

from configs.config_bteh import ARTIFACT_SCHEMA_VERSION
from pipeline.bteh_head_crop_pilot import (
    DEFAULT_PRECISION_GATE,
    HEAD_EXPERIMENT_MANIFEST_COLUMNS,
    HEAD_PILOT_FINGERPRINT_FILENAME,
    HEAD_PILOT_REPORT_CSV,
    HEAD_PILOT_REPORT_JSON,
    HEAD_PILOT_REPORT_MD,
    HEAD_PILOT_SIDECAR_FILENAME,
    _fail_loud_head_accepted_files_exist,
    _fail_loud_head_crop_kind,
    _fail_loud_head_detector_fingerprint,
    _fail_loud_head_joins,
    _fail_loud_head_schema,
    _fail_loud_head_source_split_fingerprints,
    _fail_loud_fingerprint_record_match,
    _strip_absolute_prefix,
    check_precision_gate,
    compute_detector_config_fingerprint,
    compute_head_metrics,
    generate_head_contact_sheets,
    load_body_pilot_manifest,
    load_detector_fingerprint_record,
    write_detector_fingerprint_record,
    write_head_metric_reports,
    write_head_pilot_sidecar,
)
from utils.artifact_schema import make_crop_id

# ---------------------------------------------------------------------------
# Re-export HEAD_EXPERIMENT_MANIFEST_COLUMNS from pipeline module if not
# directly importable from utils (the pipeline re-exports it).
# ---------------------------------------------------------------------------
try:
    from pipeline.bteh_head_crop_pilot import HEAD_EXPERIMENT_MANIFEST_COLUMNS  # noqa: F811
except ImportError:
    from utils.artifact_schema import HEAD_EXPERIMENT_MANIFEST_COLUMNS


# ---------------------------------------------------------------------------
# Synthetic builders
# ---------------------------------------------------------------------------

_COLOR = _count(80)


def _unique_image(path: Path, size: tuple = (24, 24)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = next(_COLOR)
    Image.new("RGB", size, color=(c % 256, (c * 7) % 256, (c * 13) % 256)).save(
        str(path), "JPEG"
    )


def _make_pilot_df(n: int = 4) -> pd.DataFrame:
    """Build a minimal pilot DataFrame (subset of canonical manifest columns)."""
    rows = []
    for i in range(n):
        rows.append({
            "image_id": f"img_{i:04d}",
            "individual_id": f"bteh_ident_{i % 3}",
            "individual_name": f"Ident {i % 3}",
            "source_relative_path": f"Ident_{i % 3}/sess_{i}/img_{i:04d}.jpg",
            "year": str(2020 + (i % 4)),
            "session_source": ["exif", "folder", "year_folder"][i % 3],
            "dataset_role": "ref" if i % 5 == 0 else "source",
            "_pilot_role": "pilot",
        })
    return pd.DataFrame(rows)


def _make_head_manifest(
    image_ids: list[str],
    *,
    status: str = "accepted",
    source_used: str = "body_crop",
    crop_root: str | None = None,
    source_fingerprint: str | None = "sfp-1",
    split_fingerprint: str | None = "spfp-1",
    detector_fingerprint: str | None = "deadbeef01020304",
) -> pd.DataFrame:
    """Build a synthetic head experiment manifest."""
    rows: list[dict] = []
    for image_id in image_ids:
        crop_id = make_crop_id(image_id, "head", 0)
        crop_path: str | None = None
        if status == "accepted" and crop_root:
            crop_path = str(Path(crop_root) / f"{crop_id}.jpg")
        elif status == "accepted":
            crop_path = f"/fake/head_crops/{crop_id}.jpg"
        rows.append({
            "crop_id": crop_id,
            "image_id": image_id,
            "individual_id": "bteh_ident_0",
            "crop_kind": "head",
            "crop_ordinal": 0,
            "crop_path": crop_path,
            "detector_confidence": 0.82 if status == "accepted" else None,
            "detector_box": "[10,10,80,80]" if status == "accepted" else None,
            "detector_status": status,
            "review_status": "pending",
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "source_fingerprint": source_fingerprint,
            "split_fingerprint": split_fingerprint,
            "source_used": source_used if status == "accepted" else None,
            "detector_fingerprint": detector_fingerprint if status == "accepted" else None,
        })
    return pd.DataFrame(rows)


def _make_body_manifest(
    image_ids: list[str],
    *,
    status: str = "accepted",
    crop_root: str | None = None,
) -> pd.DataFrame:
    """Build a minimal body crop manifest for body-input availability tests."""
    rows = []
    for image_id in image_ids:
        crop_id = make_crop_id(image_id, "body", 0)
        rows.append({
            "crop_id": crop_id,
            "image_id": image_id,
            "individual_id": "bteh_ident_0",
            "crop_kind": "body",
            "crop_ordinal": 0,
            "crop_path": (
                str(Path(crop_root) / f"{crop_id}.jpg") if crop_root else f"/fake/{crop_id}.jpg"
            ),
            "detector_confidence": 0.91,
            "detector_box": "[0,0,100,100]",
            "detector_status": status,
            "review_status": "pending",
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "source_fingerprint": "sfp-1",
            "split_fingerprint": "spfp-1",
        })
    return pd.DataFrame(rows)


def _write_pilot_parquet(tmp_path: Path, pilot_df: pd.DataFrame) -> Path:
    """Write a minimal body-pilot parquet with sidecar for load_body_pilot_manifest."""
    pq = tmp_path / "bteh_pilot_manifest.parquet"
    pilot_df.to_parquet(pq, index=False)
    sidecar = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_fingerprint": "sfp-1",
        "split_fingerprint": "spfp-1",
        "pilot_fingerprint": "pfp-1",
        "n_pilot": len(pilot_df[pilot_df.get("_pilot_role", pd.Series("pilot", index=pilot_df.index)) == "pilot"]),
    }
    (tmp_path / "bteh_pilot_manifest.json").write_text(json.dumps(sidecar))
    return pq


# ===========================================================================
# 1. Schema validation
# ===========================================================================

def test_head_schema_missing_columns_raises():
    bad = pd.DataFrame({"crop_id": ["x"], "image_id": ["y"]})
    with pytest.raises(ValueError, match="missing required columns"):
        _fail_loud_head_schema(bad)


def test_head_schema_full_columns_passes():
    row = {col: None for col in HEAD_EXPERIMENT_MANIFEST_COLUMNS}
    row["crop_id"] = "c0"
    row["image_id"] = "i0"
    row["crop_kind"] = "head"
    _fail_loud_head_schema(pd.DataFrame([row]))


def test_head_crop_kind_non_head_raises():
    row = {col: None for col in HEAD_EXPERIMENT_MANIFEST_COLUMNS}
    row["crop_id"] = "c0"
    row["image_id"] = "i0"
    row["crop_kind"] = "ear"  # wrong
    with pytest.raises(ValueError, match="non-head rows"):
        _fail_loud_head_crop_kind(pd.DataFrame([row]))


def test_head_crop_kind_all_head_passes():
    df = _make_head_manifest(["img_0001"], status="accepted", crop_root=None)
    _fail_loud_head_crop_kind(df)


# ===========================================================================
# 2. Join validation
# ===========================================================================

def test_head_joins_missing_image_raises():
    pilot = _make_pilot_df(2)
    head = _make_head_manifest(["img_0099"])  # wrong image_id
    with pytest.raises(ValueError, match="no entry in the head manifest"):
        _fail_loud_head_joins(pilot, head)


def test_head_joins_accepted_null_path_raises():
    pilot = _make_pilot_df(1)
    head = _make_head_manifest([pilot["image_id"].iloc[0]])
    head.loc[0, "crop_path"] = None  # accepted but no path
    with pytest.raises(ValueError, match="null/empty crop_path"):
        _fail_loud_head_joins(pilot, head)


def test_head_joins_none_detected_passes():
    pilot = _make_pilot_df(2)
    head = _make_head_manifest(
        pilot["image_id"].tolist(), status="none_detected"
    )
    _fail_loud_head_joins(pilot, head)


# ===========================================================================
# 3. Missing accepted file
# ===========================================================================

def test_missing_accepted_head_file_raises(tmp_path):
    image_ids = ["img_0001", "img_0002"]
    crop_root = str(tmp_path / "head_crops")
    head = _make_head_manifest(image_ids, status="accepted", crop_root=crop_root)
    # Files NOT created → should raise
    with pytest.raises(ValueError, match="accepted head crop file"):
        _fail_loud_head_accepted_files_exist(head, set(image_ids))


def test_missing_accepted_head_file_passes_when_none_detected():
    image_ids = ["img_0001"]
    head = _make_head_manifest(image_ids, status="none_detected")
    # No accepted rows → no file check needed
    _fail_loud_head_accepted_files_exist(head, set(image_ids))


def test_missing_accepted_head_file_passes_when_files_exist(tmp_path):
    image_ids = ["img_0001"]
    crop_root = str(tmp_path / "head_crops")
    head = _make_head_manifest(image_ids, status="accepted", crop_root=crop_root)
    # Create the files
    for _, row in head.iterrows():
        p = Path(str(row["crop_path"]))
        _unique_image(p)
    _fail_loud_head_accepted_files_exist(head, set(image_ids))


# ===========================================================================
# 4. Detector fingerprint
# ===========================================================================

def test_detector_fingerprint_determinism():
    fp1 = compute_detector_config_fingerprint(0.30, 0.01, 0.70, 0.3, 3.0, 0.50, 0.10)
    fp2 = compute_detector_config_fingerprint(0.30, 0.01, 0.70, 0.3, 3.0, 0.50, 0.10)
    assert fp1 == fp2


def test_detector_fingerprint_sensitive_to_config():
    fp1 = compute_detector_config_fingerprint(0.30, 0.01, 0.70, 0.3, 3.0, 0.50, 0.10)
    fp2 = compute_detector_config_fingerprint(0.35, 0.01, 0.70, 0.3, 3.0, 0.50, 0.10)  # diff conf
    assert fp1 != fp2


def test_write_and_load_fingerprint_record(tmp_path):
    pilot = _make_pilot_df(3)
    config = {
        "conf_threshold": 0.30,
        "min_area_frac": 0.01,
        "max_area_frac": 0.70,
        "min_aspect": 0.3,
        "max_aspect": 3.0,
        "iou_threshold": 0.50,
        "pad_frac": 0.10,
    }
    fp = compute_detector_config_fingerprint(**config)
    body_sidecar = {"source_fingerprint": "sfp-1", "split_fingerprint": "spfp-1"}
    record_path = write_detector_fingerprint_record(fp, config, pilot, body_sidecar, tmp_path)
    assert record_path.exists()

    loaded = load_detector_fingerprint_record(tmp_path)
    assert loaded is not None
    assert loaded["detector_fingerprint"] == fp
    assert loaded["source_fingerprint"] == "sfp-1"
    assert "detector_config" in loaded
    assert "pilot_image_ids" in loaded


def test_fingerprint_record_match_passes():
    fp = compute_detector_config_fingerprint(0.30, 0.01, 0.70, 0.3, 3.0, 0.50, 0.10)
    existing = {"detector_fingerprint": fp, "source_fingerprint": "sfp", "split_fingerprint": None}
    # Should not raise
    _fail_loud_fingerprint_record_match(existing, fp, "sfp", None)


def test_fingerprint_record_mismatch_raises():
    fp = compute_detector_config_fingerprint(0.30, 0.01, 0.70, 0.3, 3.0, 0.50, 0.10)
    wrong_fp = "aaaa1111bbbb2222"
    existing = {"detector_fingerprint": wrong_fp, "source_fingerprint": "sfp"}
    with pytest.raises(ValueError, match="Detector fingerprint mismatch"):
        _fail_loud_fingerprint_record_match(existing, fp, "sfp", None)


def test_source_fingerprint_mismatch_raises():
    fp = compute_detector_config_fingerprint(0.30, 0.01, 0.70, 0.3, 3.0, 0.50, 0.10)
    existing = {"detector_fingerprint": fp, "source_fingerprint": "old-sfp"}
    with pytest.raises(ValueError, match="Source fingerprint mismatch"):
        _fail_loud_fingerprint_record_match(existing, fp, "new-sfp", None)


def test_head_manifest_detector_fingerprint_mismatch_raises():
    image_ids = ["img_0001"]
    head = _make_head_manifest(image_ids, detector_fingerprint="correct-fp")
    with pytest.raises(ValueError, match="detector_fingerprint mismatch"):
        _fail_loud_head_detector_fingerprint(head, "wrong-fp")


def test_head_manifest_detector_fingerprint_passes():
    fp = "deadbeef01020304"
    head = _make_head_manifest(["img_0001"], detector_fingerprint=fp)
    _fail_loud_head_detector_fingerprint(head, fp)


def test_head_source_split_fingerprint_mismatch_raises():
    image_ids = ["img_0001"]
    head = _make_head_manifest(image_ids, source_fingerprint="old-sfp")
    with pytest.raises(ValueError, match="source_fingerprint mismatch"):
        _fail_loud_head_source_split_fingerprints(
            head, {"source_fingerprint": "new-sfp"}
        )


# ===========================================================================
# 5. Metrics: 0 head crops (all none_detected)
# ===========================================================================

def test_metrics_zero_head():
    pilot = _make_pilot_df(3)
    head = _make_head_manifest(
        pilot["image_id"].tolist(), status="none_detected", crop_root=None
    )
    metrics = compute_head_metrics(pilot, head)
    det = metrics["detector"]
    assert det["n_images"] == 3
    assert det["n_head_accepted"] == 0
    assert det["head_accepted_coverage"] == 0.0
    assert det["n_from_body_crop"] == 0
    assert det["n_from_original_fallback"] == 0
    assert metrics["human"]["precision"] is None


# ===========================================================================
# 6. Metrics: 1 head from body_crop
# ===========================================================================

def test_metrics_one_head_from_body_crop(tmp_path):
    pilot = _make_pilot_df(2)
    ids = pilot["image_id"].tolist()
    crop_root = str(tmp_path / "head_crops")
    head = _make_head_manifest(
        [ids[0]], status="accepted", source_used="body_crop", crop_root=crop_root
    )
    # Second image: none_detected
    head2 = _make_head_manifest([ids[1]], status="none_detected")
    head_df = pd.concat([head, head2], ignore_index=True)

    metrics = compute_head_metrics(pilot, head_df)
    det = metrics["detector"]
    assert det["n_head_accepted"] == 1
    assert det["n_from_body_crop"] == 1
    assert det["n_from_original_fallback"] == 0
    assert det["body_crop_rate"] == 1.0
    assert det["original_fallback_rate"] == 0.0


# ===========================================================================
# 7. Metrics: 1 head from original (fallback)
# ===========================================================================

def test_metrics_one_head_from_original_fallback(tmp_path):
    pilot = _make_pilot_df(2)
    ids = pilot["image_id"].tolist()
    head = _make_head_manifest(
        [ids[0]], status="accepted", source_used="original", crop_root=None
    )
    head2 = _make_head_manifest([ids[1]], status="none_detected")
    head_df = pd.concat([head, head2], ignore_index=True)

    metrics = compute_head_metrics(pilot, head_df)
    det = metrics["detector"]
    assert det["n_from_original_fallback"] == 1
    assert det["n_from_body_crop"] == 0
    assert det["original_fallback_rate"] == 1.0


# ===========================================================================
# 8. Body input availability metrics
# ===========================================================================

def test_body_input_availability_metrics():
    pilot = _make_pilot_df(4)
    ids = pilot["image_id"].tolist()
    # Only first 2 have accepted body crops
    body = _make_body_manifest(ids[:2], status="accepted")
    head = _make_head_manifest(ids, status="none_detected")

    metrics = compute_head_metrics(pilot, head, body_df=body)
    det = metrics["detector"]
    assert det["n_body_input_available"] == 2
    assert det["body_input_coverage"] == pytest.approx(0.5, abs=1e-4)


def test_body_input_availability_none_when_body_df_absent():
    pilot = _make_pilot_df(2)
    head = _make_head_manifest(pilot["image_id"].tolist(), status="none_detected")
    metrics = compute_head_metrics(pilot, head, body_df=None)
    assert metrics["detector"]["n_body_input_available"] is None
    assert metrics["detector"]["body_input_coverage"] is None


# ===========================================================================
# 9. Contact sheets: 0 head crops
# ===========================================================================

def test_contact_sheet_zero_head(tmp_path):
    pilot = _make_pilot_df(2)
    head = _make_head_manifest(
        pilot["image_id"].tolist(), status="none_detected"
    )
    cs_dir = tmp_path / "cs"
    pages = generate_head_contact_sheets(pilot, head, cs_dir, page_size=4)
    assert len(pages) >= 1
    assert all(p.exists() for p in pages)


# ===========================================================================
# 10. Contact sheets: 1 head crop
# ===========================================================================

def test_contact_sheet_one_head(tmp_path):
    pilot = _make_pilot_df(2)
    ids = pilot["image_id"].tolist()
    crop_root = str(tmp_path / "head_crops")
    head = _make_head_manifest(
        [ids[0]], status="accepted", crop_root=crop_root
    )
    head2 = _make_head_manifest([ids[1]], status="none_detected")
    head_df = pd.concat([head, head2], ignore_index=True)

    Path(crop_root).mkdir(parents=True, exist_ok=True)
    for _, row in head.iterrows():
        if row["detector_status"] == "accepted":
            _unique_image(Path(str(row["crop_path"])))

    cs_dir = tmp_path / "cs"
    pages = generate_head_contact_sheets(pilot, head_df, cs_dir, page_size=4)
    assert len(pages) >= 1
    assert all(p.exists() for p in pages)


# ===========================================================================
# 11. Contact sheets: no absolute paths in labels
# ===========================================================================

def test_strip_absolute_prefix_removes_home():
    abs_path = "/workspace/someuser/projects/BTEH/Ident_0/img_001.jpg"
    result = _strip_absolute_prefix(abs_path, "/workspace/someuser/projects/BTEH")
    assert not result.startswith("/workspace/"), f"Absolute path not stripped: {result!r}"
    assert "img_001.jpg" in result


def test_strip_absolute_prefix_fallback_regex():
    abs_path = "/" + "home/user/myproject/data/img.jpg"
    result = _strip_absolute_prefix(abs_path, None)
    assert not result.startswith("/" + "home/"), f"Absolute path not stripped: {result!r}"


# ===========================================================================
# 12. Review coverage: every accepted crop reviewed exactly once
# ===========================================================================

def test_review_coverage_complete(tmp_path):
    pilot = _make_pilot_df(3)
    ids = pilot["image_id"].tolist()
    head = _make_head_manifest(ids, status="accepted", crop_root=None)
    crop_ids = head["crop_id"].tolist()

    review_path = tmp_path / "review.csv"
    with open(review_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["crop_id", "status"])
        w.writeheader()
        for cid in crop_ids:
            w.writerow({"crop_id": cid, "status": "accepted"})

    metrics = compute_head_metrics(pilot, head, review_csv_path=review_path)
    hum = metrics["human"]
    assert hum["coverage_complete"] is True
    assert hum["missing_from_review"] == []
    assert hum["terminal_ids_in_review"] == []


def test_review_coverage_incomplete_reports_missing(tmp_path):
    pilot = _make_pilot_df(3)
    ids = pilot["image_id"].tolist()
    head = _make_head_manifest(ids, status="accepted")
    crop_ids = head["crop_id"].tolist()

    # Only review first crop_id
    review_path = tmp_path / "review.csv"
    with open(review_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["crop_id", "status"])
        w.writeheader()
        w.writerow({"crop_id": crop_ids[0], "status": "accepted"})

    metrics = compute_head_metrics(pilot, head, review_csv_path=review_path)
    hum = metrics["human"]
    assert hum["coverage_complete"] is False
    assert len(hum["missing_from_review"]) == 2


# ===========================================================================
# 13. Review: terminal IDs must not appear in review CSV
# ===========================================================================

def test_review_terminal_ids_flagged(tmp_path):
    pilot = _make_pilot_df(2)
    ids = pilot["image_id"].tolist()
    # First: accepted, second: none_detected
    head_acc = _make_head_manifest([ids[0]], status="accepted")
    head_nd = _make_head_manifest([ids[1]], status="none_detected")
    head_df = pd.concat([head_acc, head_nd], ignore_index=True)

    terminal_cid = head_nd["crop_id"].iloc[0]
    accepted_cid = head_acc["crop_id"].iloc[0]

    review_path = tmp_path / "review.csv"
    with open(review_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["crop_id", "status"])
        w.writeheader()
        w.writerow({"crop_id": accepted_cid, "status": "accepted"})
        w.writerow({"crop_id": terminal_cid, "status": "rejected"})

    metrics = compute_head_metrics(pilot, head_df, review_csv_path=review_path)
    hum = metrics["human"]
    assert terminal_cid in hum["terminal_ids_in_review"]


# ===========================================================================
# 14. Review: duplicate entry raises
# ===========================================================================

def test_review_duplicate_crop_id_raises(tmp_path):
    pilot = _make_pilot_df(1)
    ids = pilot["image_id"].tolist()
    head = _make_head_manifest(ids, status="accepted")
    cid = head["crop_id"].iloc[0]

    review_path = tmp_path / "review.csv"
    with open(review_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["crop_id", "status"])
        w.writeheader()
        w.writerow({"crop_id": cid, "status": "accepted"})
        w.writerow({"crop_id": cid, "status": "rejected"})  # duplicate!

    with pytest.raises(ValueError, match="duplicate entries"):
        compute_head_metrics(pilot, head, review_csv_path=review_path)


# ===========================================================================
# 15. Review: precision gate pass
# ===========================================================================

def test_precision_gate_passes(tmp_path):
    pilot = _make_pilot_df(4)
    ids = pilot["image_id"].tolist()
    head = _make_head_manifest(ids, status="accepted")
    crop_ids = head["crop_id"].tolist()

    # All accepted → precision = 1.0
    review_path = tmp_path / "review.csv"
    with open(review_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["crop_id", "status"])
        w.writeheader()
        for cid in crop_ids:
            w.writerow({"crop_id": cid, "status": "accepted"})

    metrics = compute_head_metrics(pilot, head, review_csv_path=review_path)
    assert metrics["human"]["precision"] == 1.0
    assert check_precision_gate(metrics, gate=0.95) is True


def test_precision_gate_fails(tmp_path):
    pilot = _make_pilot_df(4)
    ids = pilot["image_id"].tolist()
    head = _make_head_manifest(ids, status="accepted")
    crop_ids = head["crop_id"].tolist()

    # 1/4 accepted, 3/4 rejected → precision = 0.25
    review_path = tmp_path / "review.csv"
    with open(review_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["crop_id", "status"])
        w.writeheader()
        for i, cid in enumerate(crop_ids):
            w.writerow({"crop_id": cid, "status": "accepted" if i == 0 else "rejected"})

    metrics = compute_head_metrics(pilot, head, review_csv_path=review_path)
    assert metrics["human"]["precision"] == pytest.approx(0.25, abs=1e-4)
    assert check_precision_gate(metrics, gate=0.95) is False


def test_precision_gate_no_review_passes():
    """When no review CSV, gate must pass (no data to fail on)."""
    pilot = _make_pilot_df(2)
    head = _make_head_manifest(pilot["image_id"].tolist(), status="accepted")
    metrics = compute_head_metrics(pilot, head, review_csv_path=None)
    assert metrics["human"]["precision"] is None
    assert check_precision_gate(metrics, gate=0.95) is True


def test_precision_gate_custom_threshold(tmp_path):
    pilot = _make_pilot_df(2)
    ids = pilot["image_id"].tolist()
    head = _make_head_manifest(ids, status="accepted")
    crop_ids = head["crop_id"].tolist()

    # 1/2 accepted → precision = 0.5
    review_path = tmp_path / "review.csv"
    with open(review_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["crop_id", "status"])
        w.writeheader()
        w.writerow({"crop_id": crop_ids[0], "status": "accepted"})
        w.writerow({"crop_id": crop_ids[1], "status": "rejected"})

    metrics = compute_head_metrics(pilot, head, review_csv_path=review_path)
    assert check_precision_gate(metrics, gate=0.50) is True
    assert check_precision_gate(metrics, gate=0.51) is False


# ===========================================================================
# 16. Review: invalid status raises
# ===========================================================================

def test_review_invalid_status_raises(tmp_path):
    pilot = _make_pilot_df(1)
    head = _make_head_manifest(pilot["image_id"].tolist(), status="accepted")
    cid = head["crop_id"].iloc[0]

    review_path = tmp_path / "review.csv"
    with open(review_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["crop_id", "status"])
        w.writeheader()
        w.writerow({"crop_id": cid, "status": "INVALID_STATUS"})

    with pytest.raises(ValueError, match="invalid status values"):
        compute_head_metrics(pilot, head, review_csv_path=review_path)


# ===========================================================================
# 17. Metric reports written to disk
# ===========================================================================

def test_metric_reports_written(tmp_path):
    pilot = _make_pilot_df(3)
    ids = pilot["image_id"].tolist()
    head = _make_head_manifest(ids, status="accepted")

    metrics = compute_head_metrics(pilot, head)
    json_p, csv_p, md_p = write_head_metric_reports(metrics, tmp_path)

    assert json_p.exists()
    assert csv_p.exists()
    assert md_p.exists()

    data = json.loads(json_p.read_text())
    assert "detector" in data
    assert "human" in data
    assert "by_stratum" in data

    md_text = md_p.read_text()
    assert "BTEH Head-Crop Pilot" in md_text
    assert "Detector Metrics" in md_text
    assert "Visual Review" in md_text


def test_metric_reports_filenames(tmp_path):
    pilot = _make_pilot_df(2)
    head = _make_head_manifest(pilot["image_id"].tolist(), status="none_detected")
    metrics = compute_head_metrics(pilot, head)
    json_p, csv_p, md_p = write_head_metric_reports(metrics, tmp_path)
    assert json_p.name == HEAD_PILOT_REPORT_JSON
    assert csv_p.name == HEAD_PILOT_REPORT_CSV
    assert md_p.name == HEAD_PILOT_REPORT_MD


# ===========================================================================
# 18. Output isolation: head pilot does not alter body-pilot files
# ===========================================================================

def test_output_isolation(tmp_path):
    pilot = _make_pilot_df(2)
    pq = _write_pilot_parquet(tmp_path, pilot)

    # Record mtime of body pilot files
    body_pq_mtime = pq.stat().st_mtime
    body_sc_mtime = (tmp_path / "bteh_pilot_manifest.json").stat().st_mtime

    # Run head report into a separate subdir
    head_dir = tmp_path / "head_output"
    head = _make_head_manifest(pilot["image_id"].tolist(), status="none_detected")
    metrics = compute_head_metrics(pilot, head)
    write_head_metric_reports(metrics, head_dir)
    write_head_pilot_sidecar(pilot, {}, head_dir)

    # Body pilot files must not have been modified
    assert pq.stat().st_mtime == body_pq_mtime, "Body pilot parquet was modified"
    assert (tmp_path / "bteh_pilot_manifest.json").stat().st_mtime == body_sc_mtime, (
        "Body pilot sidecar was modified"
    )

    # Head files go only to head_dir
    assert (head_dir / HEAD_PILOT_SIDECAR_FILENAME).exists()
    assert not (tmp_path / HEAD_PILOT_SIDECAR_FILENAME).exists()


# ===========================================================================
# 19. load_body_pilot_manifest
# ===========================================================================

def test_load_body_pilot_manifest_pilot_role_filter(tmp_path):
    pilot = _make_pilot_df(3)
    review_rows = pilot.iloc[:1].copy()
    review_rows["_pilot_role"] = "review"
    combined = pd.concat([pilot, review_rows], ignore_index=True)
    pq = tmp_path / "bteh_pilot_manifest.parquet"
    combined.to_parquet(pq)
    (tmp_path / "bteh_pilot_manifest.json").write_text("{}")

    loaded, _ = load_body_pilot_manifest(pq)
    assert all(loaded["_pilot_role"] == "pilot")


def test_load_body_pilot_manifest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Body-pilot manifest not found"):
        load_body_pilot_manifest(tmp_path / "nonexistent.parquet")


# ===========================================================================
# 20. Head pilot sidecar write
# ===========================================================================

def test_write_head_pilot_sidecar(tmp_path):
    pilot = _make_pilot_df(3)
    sidecar = {"source_fingerprint": "sfp", "split_fingerprint": "spfp", "pilot_fingerprint": "pfp"}
    sc_path = write_head_pilot_sidecar(pilot, sidecar, tmp_path)
    assert sc_path.exists()
    data = json.loads(sc_path.read_text())
    assert data["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert data["body_pilot_source_fingerprint"] == "sfp"
    assert len(data["pilot_image_ids"]) == 3


# ===========================================================================
# 21. Strata metrics by year/session_source/origin
# ===========================================================================

def test_strata_metrics_present():
    pilot = _make_pilot_df(4)
    head = _make_head_manifest(pilot["image_id"].tolist(), status="none_detected")
    metrics = compute_head_metrics(pilot, head)
    strata = metrics["by_stratum"]
    assert "year" in strata
    assert "session_source" in strata
    assert "origin" in strata


# ===========================================================================
# 22. Distributions for accepted crops
# ===========================================================================

def test_distributions_computed_for_accepted(tmp_path):
    pilot = _make_pilot_df(3)
    ids = pilot["image_id"].tolist()
    head = _make_head_manifest(ids, status="accepted")

    metrics = compute_head_metrics(pilot, head)
    dist = metrics["detector"]["distributions"]
    assert "confidence" in dist
    conf_stats = dist["confidence"]
    assert conf_stats.get("count", 0) == 3
    assert 0 <= conf_stats["min"] <= conf_stats["max"] <= 1.0


def test_distributions_empty_for_zero_accepted():
    pilot = _make_pilot_df(2)
    head = _make_head_manifest(pilot["image_id"].tolist(), status="none_detected")
    metrics = compute_head_metrics(pilot, head)
    dist = metrics["detector"]["distributions"]
    assert dist.get("confidence") == {}


# ===========================================================================
# 23. Review: no CSV → missing_from_review lists all accepted crops
# ===========================================================================

def test_no_review_csv_missing_lists_all_accepted():
    pilot = _make_pilot_df(2)
    head = _make_head_manifest(pilot["image_id"].tolist(), status="accepted")
    accepted_cids = set(head["crop_id"].tolist())

    metrics = compute_head_metrics(pilot, head, review_csv_path=None)
    hum = metrics["human"]
    assert set(hum["missing_from_review"]) == accepted_cids
    assert hum["coverage_complete"] is False
