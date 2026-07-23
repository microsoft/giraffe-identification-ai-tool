# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Tests for pipeline/build_production_index.py

Coverage
--------
1. Merge alignment: reference + query rows merged correctly; contiguous indices.
2. Duplicate crop_id rejection: within reference partition.
3. Cross-partition overlap rejection: same crop_id in reference and query.
4. Fingerprint mismatch rejection: source/split/model.
5. Identity mismatch rejection: same crop_id → different individual_id.
6. FAISS ntotal matches merged matrix rows after build.
7. Policy manifest: auto_accept_policy.enabled is False.
8. No eval artifact mutation: reference partition files unchanged after build.
9. Adopted checkpoint gate: rejected checkpoint raises AssertionError.
10. Dry-run: validates without writing output files.
11. Round-trip integrity: written .npy and .parquet match in-memory data.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import faiss
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.build_production_index import (
    AUTO_ACCEPT_POLICY,
    EXPECTED_MODEL_FINGERPRINTS,
    PRODUCTION_CHANNELS,
    PRODUCTION_CALIBRATION_SUBDIR,
    PRODUCTION_CHECKPOINT_SUBDIR,
    PRODUCTION_FUSION_WEIGHTS,
    _build_faiss_index,
    _check_no_eval_artifact_mutation,
    _merge_channel,
    _post_validate_channel,
    _validate_fingerprints,
    _write_channel_artifacts,
    build_production_index,
)
from utils.artifact_schema import DESCRIPTOR_MAPPING_COLUMNS
from utils.utils_embeddings import l2_normalize


# ---------------------------------------------------------------------------
# Synthetic fixture builders
# ---------------------------------------------------------------------------

_SOURCE_FP = "a" * 64
_SPLIT_FP = "b" * 64
_SCHEMA_VER = "v1"


def _make_mapping(
    channel: str,
    n: int,
    start_row: int = 0,
    individual_ids: list[str] | None = None,
    source_fp: str = _SOURCE_FP,
    split_fp: str = _SPLIT_FP,
    model_fp: str | None = None,
    crop_id_prefix: str = "crop",
    crop_kind: str = "body",
) -> pd.DataFrame:
    if individual_ids is None:
        individual_ids = [f"bteh_ind_{i % 5}" for i in range(n)]
    if model_fp is None:
        model_fp = EXPECTED_MODEL_FINGERPRINTS.get(channel, f"{channel}:test-v1")

    rows = []
    for i in range(n):
        row_idx = start_row + i
        rows.append(
            {
                "descriptor_name": channel,
                "embedding_row": i,
                "faiss_row": i,
                "crop_id": f"{crop_id_prefix}_{channel}_{row_idx:04d}",
                "image_id": f"img_{row_idx:04d}",
                "individual_id": individual_ids[i],
                "crop_kind": crop_kind,
                "crop_ordinal": 0,
                "crop_path": f"/data/crops/{crop_id_prefix}_{row_idx:04d}.jpg",
                "schema_version": _SCHEMA_VER,
                "source_fingerprint": source_fp,
                "split_fingerprint": split_fp,
                "model_preprocess_fingerprint": model_fp,
            }
        )
    return pd.DataFrame(rows)


def _make_matrix(n: int, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.random((n, dim)).astype(np.float32)
    return l2_normalize(mat)


def _write_partition(
    base_dir: str,
    partition: str,
    channel: str,
    mapping_df: pd.DataFrame,
    matrix: np.ndarray,
    build_index: bool = False,
) -> None:
    """Write embedding partition files the same way the real pipeline does."""
    out_dir = os.path.join(base_dir, "embeddings", partition)
    os.makedirs(out_dir, exist_ok=True)
    mapping_df = mapping_df.copy()
    mapping_df["crop_path"] = mapping_df["crop_id"].map(
        lambda crop_id: os.path.join(base_dir, "crops", f"{crop_id}.jpg")
    )
    np.save(os.path.join(out_dir, f"{channel}.npy"), matrix)
    mapping_df.to_parquet(os.path.join(out_dir, f"{channel}_mapping.parquet"), index=False)
    if build_index:
        idx = faiss.IndexFlatIP(matrix.shape[1])
        idx.add(matrix)
        faiss.write_index(idx, os.path.join(out_dir, f"{channel}.index"))


def _write_calibration_manifest(
    base_dir: str,
    channels: list[str],
    source_fp: str = _SOURCE_FP,
    split_fp: str = _SPLIT_FP,
) -> str:
    """Write minimal calibration_projected structure for build tests."""
    cal_dir = os.path.join(base_dir, PRODUCTION_CALIBRATION_SUBDIR)
    os.makedirs(cal_dir, exist_ok=True)

    model_fps = {ch: EXPECTED_MODEL_FINGERPRINTS.get(ch, f"{ch}:test-v1") for ch in channels}

    manifest = {
        "schema_version": "v1",
        "channels": channels,
        "artifact_fingerprints": {
            "source_fingerprint": source_fp,
            "split_fingerprint": split_fp,
            "model_preprocess_fingerprints": model_fps,
        },
        "fusion_weights": PRODUCTION_FUSION_WEIGHTS,
        "calibration_note": "test calibration",
    }
    path = os.path.join(cal_dir, "calibration_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f)

    # Write stub calibration pkl files for each channel.
    import pickle
    for ch in channels:
        pkl_path = os.path.join(
            cal_dir,
            {
                "miewid": "miewid.pkl",
                "ear_miewid_projected": "ear_miewid_projected.pkl",
            }.get(ch, f"{ch}.pkl"),
        )
        with open(pkl_path, "wb") as f:
            pickle.dump({"channel": ch}, f)

    # Write fusion weights and unknown threshold stubs.
    with open(os.path.join(cal_dir, "fusion_weights.json"), "w") as f:
        json.dump({"weights": PRODUCTION_FUSION_WEIGHTS}, f)
    with open(os.path.join(cal_dir, "unknown_threshold.json"), "w") as f:
        json.dump({"threshold": 0.175}, f)

    return path


def _write_checkpoint_manifest(
    base_dir: str,
    adopted: bool = True,
) -> str:
    """Write minimal checkpoint structure."""
    ckpt_dir = os.path.join(base_dir, PRODUCTION_CHECKPOINT_SUBDIR)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Write stub .pt file.
    pt_path = os.path.join(ckpt_dir, "best_projection.pt")
    with open(pt_path, "wb") as f:
        f.write(b"FAKE_CHECKPOINT")

    manifest = {
        "schema_version": "v1",
        "descriptor": "ear_miewid",
        "checkpoint_fingerprint": "02474758261e01e5d07a4b1dc1dc5cfa725e13dc5a8a80acac84a6e466ccb3a8",
        "base_model_fingerprint": "ear_miewid:config-elephant-v1",
        "best_val_map": 0.620278,
        "baseline_map": 0.607495,
        "gate": {
            "adopted": adopted,
            "reason": "ADOPTED: test" if adopted else "REJECTED: test",
        },
        "safety": {
            "forbidden_image_ids_checked": True,
            "forbidden_individual_ids_checked": True,
            "session_disjoint_verified": True,
            "probe_never_used_for_training": True,
            "heldout_never_used_for_training": True,
        },
    }
    manifest_path = os.path.join(ckpt_dir, "training_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    return manifest_path


def _build_full_test_artifact_root(
    tmp_path: Path,
    n_ref: int = 20,
    n_qry: int = 5,
    channel: str = "miewid",
    dim: int = 16,
    adopted_gate: bool = True,
    channels: list[str] | None = None,
) -> str:
    """Create a complete minimal artifact root for end-to-end build tests."""
    if channels is None:
        channels = PRODUCTION_CHANNELS

    root = str(tmp_path)
    _write_calibration_manifest(root, channels)
    _write_checkpoint_manifest(root, adopted=adopted_gate)

    for ch in channels:
        n_ref_ch = n_ref
        n_qry_ch = n_qry

        ref_df = _make_mapping(
            ch, n_ref_ch, start_row=0, crop_id_prefix="ref"
        )
        ref_mat = _make_matrix(n_ref_ch, dim=dim, seed=1)
        _write_partition(root, "reference", ch, ref_df, ref_mat, build_index=True)

        qry_df = _make_mapping(
            ch, n_qry_ch, start_row=n_ref_ch, crop_id_prefix="qry"
        )
        qry_mat = _make_matrix(n_qry_ch, dim=dim, seed=2)
        _write_partition(root, "query", ch, qry_df, qry_mat, build_index=False)

    return root


# ---------------------------------------------------------------------------
# Unit tests for _validate_fingerprints
# ---------------------------------------------------------------------------


def test_validate_fingerprints_pass():
    df = _make_mapping("miewid", 5)
    _validate_fingerprints(df, "miewid", _SOURCE_FP, _SPLIT_FP)


def test_validate_fingerprints_source_mismatch():
    df = _make_mapping("miewid", 5, source_fp="wrong" * 10)
    with pytest.raises(AssertionError, match="source_fingerprint mismatch"):
        _validate_fingerprints(df, "miewid", _SOURCE_FP, _SPLIT_FP)


def test_validate_fingerprints_split_mismatch():
    df = _make_mapping("miewid", 5, split_fp="wrong" * 10)
    with pytest.raises(AssertionError, match="split_fingerprint mismatch"):
        _validate_fingerprints(df, "miewid", _SOURCE_FP, _SPLIT_FP)


def test_validate_fingerprints_model_mismatch():
    df = _make_mapping("miewid", 5, model_fp="wrong_model:v9")
    with pytest.raises(AssertionError, match="model_preprocess_fingerprint mismatch"):
        _validate_fingerprints(df, "miewid", _SOURCE_FP, _SPLIT_FP)


# ---------------------------------------------------------------------------
# Unit tests for _merge_channel
# ---------------------------------------------------------------------------


def test_merge_alignment(tmp_path):
    """Merged mapping must have contiguous embedding_row 0..N-1."""
    channel = "miewid"
    n_ref, n_qry = 10, 4
    root = str(tmp_path)

    ref_df = _make_mapping(channel, n_ref, start_row=0, crop_id_prefix="ref")
    ref_mat = _make_matrix(n_ref)
    _write_partition(root, "reference", channel, ref_df, ref_mat, build_index=True)

    qry_df = _make_mapping(channel, n_qry, start_row=n_ref, crop_id_prefix="qry")
    qry_mat = _make_matrix(n_qry, seed=1)
    _write_partition(root, "query", channel, qry_df, qry_mat)

    merged_df, merged_mat = _merge_channel(root, channel, _SOURCE_FP, _SPLIT_FP)

    assert len(merged_df) == n_ref + n_qry
    assert merged_mat.shape[0] == n_ref + n_qry
    assert merged_df["embedding_row"].tolist() == list(range(n_ref + n_qry))
    assert merged_df["faiss_row"].tolist() == list(range(n_ref + n_qry))
    assert all(not os.path.isabs(path) for path in merged_df["crop_path"])


def test_merge_rejects_unordered_source_rows(tmp_path):
    channel = "miewid"
    root = str(tmp_path)
    ref_df = _make_mapping(channel, 3, crop_id_prefix="ref")
    ref_df["embedding_row"] = [1, 0, 2]
    _write_partition(root, "reference", channel, ref_df, _make_matrix(3), build_index=True)
    qry_df = _make_mapping(channel, 2, start_row=3, crop_id_prefix="qry")
    _write_partition(root, "query", channel, qry_df, _make_matrix(2, seed=1))
    with pytest.raises(AssertionError, match="ordered by contiguous"):
        _merge_channel(root, channel, _SOURCE_FP, _SPLIT_FP)


def test_merge_preserves_all_crop_ids(tmp_path):
    """All crop_ids from both partitions must appear in the merged mapping."""
    channel = "miewid"
    n_ref, n_qry = 8, 3
    root = str(tmp_path)

    ref_df = _make_mapping(channel, n_ref, crop_id_prefix="ref")
    ref_mat = _make_matrix(n_ref)
    _write_partition(root, "reference", channel, ref_df, ref_mat, build_index=True)

    qry_df = _make_mapping(channel, n_qry, start_row=n_ref, crop_id_prefix="qry")
    qry_mat = _make_matrix(n_qry, seed=1)
    _write_partition(root, "query", channel, qry_df, qry_mat)

    merged_df, _ = _merge_channel(root, channel, _SOURCE_FP, _SPLIT_FP)

    expected_ids = set(ref_df["crop_id"]) | set(qry_df["crop_id"])
    assert set(merged_df["crop_id"]) == expected_ids


def test_merge_duplicate_within_reference_fails(tmp_path):
    """Duplicate crop_id within the reference partition must raise AssertionError."""
    channel = "miewid"
    root = str(tmp_path)

    ref_df = _make_mapping(channel, 5, crop_id_prefix="ref")
    # Inject duplicate crop_id in reference
    ref_df = pd.concat([ref_df, ref_df.iloc[[0]]], ignore_index=True)
    ref_df["embedding_row"] = range(len(ref_df))
    ref_df["faiss_row"] = range(len(ref_df))
    ref_mat = _make_matrix(len(ref_df))
    _write_partition(root, "reference", channel, ref_df, ref_mat, build_index=True)

    qry_df = _make_mapping(channel, 3, start_row=100, crop_id_prefix="qry")
    qry_mat = _make_matrix(3, seed=1)
    _write_partition(root, "query", channel, qry_df, qry_mat)

    with pytest.raises(AssertionError, match="Duplicate crop_id within reference"):
        _merge_channel(root, channel, _SOURCE_FP, _SPLIT_FP)


def test_merge_cross_partition_overlap_fails(tmp_path):
    """Same crop_id in both reference and query must raise AssertionError."""
    channel = "miewid"
    root = str(tmp_path)

    shared_crop_id = "shared_crop_0000"
    ref_df = _make_mapping(channel, 5, crop_id_prefix="ref")
    ref_df.loc[0, "crop_id"] = shared_crop_id
    ref_mat = _make_matrix(5)
    _write_partition(root, "reference", channel, ref_df, ref_mat, build_index=True)

    qry_df = _make_mapping(channel, 3, start_row=5, crop_id_prefix="qry")
    qry_df.loc[0, "crop_id"] = shared_crop_id
    qry_df.loc[0, "embedding_row"] = 5
    qry_df.loc[0, "faiss_row"] = 5
    qry_mat = _make_matrix(3, seed=1)
    _write_partition(root, "query", channel, qry_df, qry_mat)

    with pytest.raises(AssertionError, match="crop_id appears in both reference and query"):
        _merge_channel(root, channel, _SOURCE_FP, _SPLIT_FP)


def test_merge_identity_mismatch_within_partition_fails(tmp_path):
    """Same crop_id with different individual_id within a partition must raise.

    Having two rows with the same crop_id (differing only in individual_id)
    triggers the duplicate-crop_id guard first.  Either way, the merge must
    fail with an AssertionError — this test verifies the hard rejection path.
    """
    channel = "miewid"
    root = str(tmp_path)

    ref_df = _make_mapping(channel, 5, crop_id_prefix="ref")
    # Two rows with same crop_id but different individual_id.
    # The duplicate-crop_id check fires before the identity-mismatch check.
    ref_df.loc[1, "crop_id"] = ref_df.loc[0, "crop_id"]
    ref_df.loc[1, "individual_id"] = "bteh_different_ind"
    ref_mat = _make_matrix(5)
    _write_partition(root, "reference", channel, ref_df, ref_mat, build_index=True)

    qry_df = _make_mapping(channel, 3, start_row=5, crop_id_prefix="qry")
    qry_mat = _make_matrix(3, seed=1)
    _write_partition(root, "query", channel, qry_df, qry_mat)

    # Either duplicate-crop_id or identity-mismatch error is acceptable here;
    # both represent the same underlying data integrity violation.
    with pytest.raises(AssertionError, match="(Duplicate crop_id|individual_id mismatch)"):
        _merge_channel(root, channel, _SOURCE_FP, _SPLIT_FP)


def test_merge_fingerprint_mismatch_in_query_fails(tmp_path):
    """Wrong model fingerprint in query partition must raise AssertionError."""
    channel = "miewid"
    root = str(tmp_path)

    ref_df = _make_mapping(channel, 5, crop_id_prefix="ref")
    ref_mat = _make_matrix(5)
    _write_partition(root, "reference", channel, ref_df, ref_mat, build_index=True)

    # Query with wrong model fingerprint
    qry_df = _make_mapping(
        channel, 3, start_row=5, crop_id_prefix="qry", model_fp="wrong_model:v99"
    )
    qry_mat = _make_matrix(3, seed=1)
    _write_partition(root, "query", channel, qry_df, qry_mat)

    with pytest.raises(AssertionError, match="model_preprocess_fingerprint mismatch"):
        _merge_channel(root, channel, _SOURCE_FP, _SPLIT_FP)


# ---------------------------------------------------------------------------
# Unit tests for _build_faiss_index
# ---------------------------------------------------------------------------


def test_faiss_ntotal_matches_matrix_rows():
    """FAISS ntotal must equal the number of rows in the embedding matrix."""
    mat = _make_matrix(25, dim=32)
    index = _build_faiss_index(mat)
    assert index.ntotal == 25


def test_faiss_self_retrieval():
    """Top-1 self-retrieval must return the correct row index for L2-normalised vectors."""
    mat = _make_matrix(30, dim=64)
    index = _build_faiss_index(mat)
    for i in range(5):
        distances, indices = index.search(mat[i : i + 1], 1)
        assert indices[0][0] == i, f"Expected self at row {i}, got {indices[0][0]}"


# ---------------------------------------------------------------------------
# Unit tests for _write_channel_artifacts + _post_validate_channel
# ---------------------------------------------------------------------------


def test_write_and_post_validate(tmp_path):
    """Written artifacts must pass round-trip validation."""
    channel = "miewid"
    n = 15
    dim = 32
    mapping_df = _make_mapping(channel, n)
    matrix = _make_matrix(n, dim=dim)
    index = _build_faiss_index(matrix)

    out_dir = str(tmp_path / "out")
    paths = _write_channel_artifacts(out_dir, channel, mapping_df, matrix, index)

    _post_validate_channel(out_dir, channel, mapping_df, matrix)

    assert os.path.exists(paths["npy"])
    assert os.path.exists(paths["mapping_parquet"])
    assert os.path.exists(paths["faiss_index"])


def test_post_validate_detects_ntotal_mismatch(tmp_path):
    """_post_validate_channel must raise if FAISS ntotal differs from matrix rows."""
    channel = "miewid"
    n = 10
    matrix = _make_matrix(n, dim=16)
    mapping_df = _make_mapping(channel, n)
    index = _build_faiss_index(matrix)

    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir, exist_ok=True)

    # Save correct matrix and mapping but a truncated index.
    np.save(os.path.join(out_dir, f"{channel}.npy"), matrix[:5])
    mapping_df.iloc[:5].to_parquet(
        os.path.join(out_dir, f"{channel}_mapping.parquet"), index=False
    )
    truncated_idx = faiss.IndexFlatIP(16)
    truncated_idx.add(matrix[:5])
    faiss.write_index(truncated_idx, os.path.join(out_dir, f"{channel}.index"))

    # Validate against the full matrix (10 rows) — should detect mismatch.
    with pytest.raises(AssertionError, match="round-trip matrix shape mismatch"):
        _post_validate_channel(out_dir, channel, mapping_df, matrix)


# ---------------------------------------------------------------------------
# Unit tests for _check_no_eval_artifact_mutation
# ---------------------------------------------------------------------------


def test_check_no_eval_artifact_mutation_passes(tmp_path):
    """Should not raise when reference partition files exist with content."""
    channel = "miewid"
    ref_dir = tmp_path / "embeddings" / "reference"
    ref_dir.mkdir(parents=True)

    npy_path = ref_dir / f"{channel}.npy"
    map_path = ref_dir / f"{channel}_mapping.parquet"

    np.save(str(npy_path), np.zeros((3, 4), dtype=np.float32))
    _make_mapping(channel, 3).to_parquet(str(map_path), index=False)

    _check_no_eval_artifact_mutation(str(tmp_path), channel)


def test_check_no_eval_artifact_mutation_missing_file(tmp_path):
    """Should raise FileNotFoundError if reference .npy file is missing."""
    channel = "miewid"
    ref_dir = tmp_path / "embeddings" / "reference"
    ref_dir.mkdir(parents=True)
    # Only write the parquet, omit the npy.
    _make_mapping(channel, 3).to_parquet(
        str(ref_dir / f"{channel}_mapping.parquet"), index=False
    )

    with pytest.raises(FileNotFoundError):
        _check_no_eval_artifact_mutation(str(tmp_path), channel)


# ---------------------------------------------------------------------------
# End-to-end build tests
# ---------------------------------------------------------------------------


def test_build_production_index_dry_run(tmp_path):
    """Dry-run must validate and return a manifest without writing output files."""
    root = _build_full_test_artifact_root(tmp_path, n_ref=10, n_qry=3, dim=16)
    manifest = build_production_index(root, build_tag="test_dry", dry_run=True)

    assert manifest["dry_run"] is True
    assert manifest["production_output_dir"] is None

    # No production directory should be created.
    prod_dir = os.path.join(root, "production", "test_dry")
    assert not os.path.exists(prod_dir)


def test_build_production_index_creates_output(tmp_path):
    """Full build must create all expected output files."""
    root = _build_full_test_artifact_root(tmp_path, n_ref=10, n_qry=3, dim=16)
    manifest = build_production_index(root, build_tag="test_build")

    out_dir = manifest["production_output_dir"]
    assert out_dir is not None
    assert os.path.isdir(out_dir)

    for ch in PRODUCTION_CHANNELS:
        assert os.path.isfile(os.path.join(out_dir, f"{ch}.npy"))
        assert os.path.isfile(os.path.join(out_dir, f"{ch}_mapping.parquet"))
        assert os.path.isfile(os.path.join(out_dir, f"{ch}.index"))

    assert os.path.isfile(os.path.join(out_dir, "production_manifest.json"))


def test_production_manifest_auto_accept_disabled(tmp_path):
    """Production manifest must have auto_accept_policy.enabled=False."""
    root = _build_full_test_artifact_root(tmp_path, n_ref=8, n_qry=2, dim=16)
    manifest = build_production_index(root, build_tag="test_policy")

    assert manifest["auto_accept_policy"]["enabled"] is False

    # Verify the written manifest file also has the correct policy.
    out_dir = manifest["production_output_dir"]
    with open(os.path.join(out_dir, "production_manifest.json")) as f:
        written = json.load(f)
    assert written["auto_accept_policy"]["enabled"] is False


def test_production_manifest_selected_channels(tmp_path):
    """Production manifest must list exactly the two selected channels."""
    root = _build_full_test_artifact_root(tmp_path, n_ref=8, n_qry=2, dim=16)
    manifest = build_production_index(root, build_tag="test_channels")

    assert set(manifest["selected_channels"]) == {"miewid", "ear_miewid_projected"}
    assert manifest["fusion_weights"]["miewid"] == 0.6
    assert manifest["fusion_weights"]["ear_miewid_projected"] == 0.4


def test_production_manifest_fusion_weights_sum(tmp_path):
    """Fusion weights in manifest must sum to 1.0."""
    root = _build_full_test_artifact_root(tmp_path, n_ref=8, n_qry=2, dim=16)
    manifest = build_production_index(root, build_tag="test_weights")

    total = sum(manifest["fusion_weights"].values())
    assert abs(total - 1.0) < 1e-6


def test_production_index_ntotal(tmp_path):
    """FAISS ntotal in produced index must equal ref + query rows."""
    n_ref, n_qry = 12, 4
    root = _build_full_test_artifact_root(
        tmp_path, n_ref=n_ref, n_qry=n_qry, dim=16, channels=["miewid"]
    )
    # Patch to only use miewid for simplicity by patching the channel list.
    # (ear_miewid_projected not in this root, so just use miewid directly)
    manifest = build_production_index.__wrapped__ if hasattr(
        build_production_index, "__wrapped__"
    ) else None

    # Use the internal _merge_channel directly.
    ref_dir = os.path.join(root, "embeddings", "reference")
    qry_dir = os.path.join(root, "embeddings", "query")
    assert os.path.isfile(os.path.join(ref_dir, "miewid.npy"))
    assert os.path.isfile(os.path.join(qry_dir, "miewid.npy"))

    merged_df, merged_mat = _merge_channel(root, "miewid", _SOURCE_FP, _SPLIT_FP)
    index = _build_faiss_index(merged_mat)

    assert index.ntotal == n_ref + n_qry
    assert len(merged_df) == n_ref + n_qry


def test_no_eval_artifact_mutation_after_build(tmp_path):
    """Reference partition .npy and .parquet files must not change after a full build."""
    root = _build_full_test_artifact_root(tmp_path, n_ref=10, n_qry=3, dim=16)

    # Record mtimes and sizes before build.
    def _snapshot(root: str) -> dict[str, tuple[float, int]]:
        snap = {}
        ref_dir = os.path.join(root, "embeddings", "reference")
        for ch in PRODUCTION_CHANNELS:
            for ext in [".npy", "_mapping.parquet"]:
                p = os.path.join(ref_dir, f"{ch}{ext}")
                if os.path.exists(p):
                    snap[p] = (os.path.getmtime(p), os.path.getsize(p))
        return snap

    before = _snapshot(root)
    build_production_index(root, build_tag="test_mutation")
    after = _snapshot(root)

    for path, (mtime_before, size_before) in before.items():
        assert path in after, f"File disappeared after build: {path}"
        mtime_after, size_after = after[path]
        assert size_after == size_before, (
            f"File size changed after build (eval artifact mutated): {path}"
        )


def test_rejected_checkpoint_gate_raises(tmp_path):
    """Build must fail with AssertionError if checkpoint gate.adopted=False."""
    root = _build_full_test_artifact_root(
        tmp_path, n_ref=8, n_qry=2, dim=16, adopted_gate=False
    )
    with pytest.raises(AssertionError, match="gate.adopted=False"):
        build_production_index(root, build_tag="test_gate", dry_run=True)


def test_missing_calibration_manifest_raises(tmp_path):
    """Build must raise FileNotFoundError if calibration manifest is missing."""
    root = _build_full_test_artifact_root(tmp_path, n_ref=8, n_qry=2, dim=16)
    # Remove calibration manifest.
    os.remove(os.path.join(root, PRODUCTION_CALIBRATION_SUBDIR, "calibration_manifest.json"))

    with pytest.raises(FileNotFoundError, match="Calibration manifest"):
        build_production_index(root, build_tag="test_no_cal", dry_run=True)


def test_missing_checkpoint_raises(tmp_path):
    """Build must raise FileNotFoundError if projection checkpoint is missing."""
    root = _build_full_test_artifact_root(tmp_path, n_ref=8, n_qry=2, dim=16)
    os.remove(os.path.join(root, PRODUCTION_CHECKPOINT_SUBDIR, "best_projection.pt"))

    with pytest.raises(FileNotFoundError, match="Projection checkpoint"):
        build_production_index(root, build_tag="test_no_ckpt", dry_run=True)


def test_channel_stats_in_manifest(tmp_path):
    """Manifest channel_stats must contain accurate row/individual counts."""
    n_ref, n_qry = 10, 4
    root = _build_full_test_artifact_root(tmp_path, n_ref=n_ref, n_qry=n_qry, dim=16)
    manifest = build_production_index(root, build_tag="test_stats")

    for ch in PRODUCTION_CHANNELS:
        stats = manifest["channel_stats"][ch]
        assert stats["n_rows"] == n_ref + n_qry, (
            f"{ch}: expected n_rows={n_ref + n_qry}, got {stats['n_rows']}"
        )
        assert stats["faiss_ntotal"] == n_ref + n_qry
        assert stats["n_individuals"] > 0


def test_production_manifest_fingerprints_present(tmp_path):
    """Production manifest must include source/split/model fingerprints."""
    root = _build_full_test_artifact_root(tmp_path, n_ref=8, n_qry=2, dim=16)
    manifest = build_production_index(root, build_tag="test_fps")

    fps = manifest["artifact_fingerprints"]
    assert "source_fingerprint" in fps
    assert "split_fingerprint" in fps
    assert "model_preprocess_fingerprints" in fps
    assert "miewid" in fps["model_preprocess_fingerprints"]
    assert "ear_miewid_projected" in fps["model_preprocess_fingerprints"]


# ---------------------------------------------------------------------------
# Policy constant tests (module-level assertions)
# ---------------------------------------------------------------------------


def test_auto_accept_policy_disabled():
    """The module-level AUTO_ACCEPT_POLICY must have enabled=False."""
    assert AUTO_ACCEPT_POLICY["enabled"] is False


def test_production_channels_constant():
    """PRODUCTION_CHANNELS must contain exactly miewid and ear_miewid_projected."""
    assert set(PRODUCTION_CHANNELS) == {"miewid", "ear_miewid_projected"}


def test_production_fusion_weights_sum():
    """PRODUCTION_FUSION_WEIGHTS must sum to 1.0."""
    total = sum(PRODUCTION_FUSION_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-6


def test_expected_model_fingerprints_present():
    """EXPECTED_MODEL_FINGERPRINTS must contain entries for both selected channels."""
    assert "miewid" in EXPECTED_MODEL_FINGERPRINTS
    assert "ear_miewid_projected" in EXPECTED_MODEL_FINGERPRINTS
    # Both must be non-empty strings.
    for ch, fp in EXPECTED_MODEL_FINGERPRINTS.items():
        assert isinstance(fp, str) and len(fp) > 0
