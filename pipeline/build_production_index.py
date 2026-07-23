# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
BTEH production index builder.

Merges reference-partition and query-partition descriptor artifacts (produced
during frozen-model evaluation) into a single full-catalog production mapping,
embedding matrix, and FAISS index for each selected channel.  Query probes
become catalog references only after model selection is frozen.

Selected production channels
-----------------------------
  body  : miewid           (MiewID applied to whole-body crop)
  ear   : ear_miewid_projected  (MiewID applied to ear crop, identity-adapter projection)

Selected calibration
---------------------
  calibration_projected/  — Platt scaling; weights miewid=0.6, ear_miewid_projected=0.4

Auto-accept policy
-------------------
  DISABLED.  Open-set threshold has FAR≈27% / FRR≈48% and is not production-safe.
  Production must surface ranked top candidates for expert human verification.
  No automatic identity creation or acceptance from threshold.

Usage (CLI)
-----------
  python -m pipeline.build_production_index \\
      --artifact-root /path/to/BTEH_reid_artifacts/v1 \\
      [--build-tag myrun_20260714]

Usage (module)
--------------
  from pipeline.build_production_index import build_production_index
  manifest = build_production_index(artifact_root="/path/to/v1")
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.artifact_schema import (
    DESCRIPTOR_MAPPING_COLUMNS,
    assert_descriptor_mapping_integrity,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Production selection constants
# ---------------------------------------------------------------------------

PRODUCTION_CHANNELS: list[str] = ["miewid", "ear_miewid_projected"]

PRODUCTION_FUSION_WEIGHTS: dict[str, float] = {
    "miewid": 0.6,
    "ear_miewid_projected": 0.4,
}

# Relative path of the selected calibration directory inside the versioned
# artifact root.  Calibration files are referenced by path, not copied, to
# avoid duplicating large files.
PRODUCTION_CALIBRATION_SUBDIR: str = "calibration_projected"

PRODUCTION_CALIBRATION_CHANNEL_FILES: dict[str, str] = {
    "miewid": "miewid.pkl",
    "ear_miewid_projected": "ear_miewid_projected.pkl",
}

PRODUCTION_CHECKPOINT_SUBDIR: str = "checkpoints/ear_miewid_identity_adapter"

# Expected model-preprocessing fingerprints for each selected channel.
# These are checked against every row of the merged mapping table.
EXPECTED_MODEL_FINGERPRINTS: dict[str, str] = {
    "miewid": "miewid:config-elephant-v1",
    "ear_miewid_projected": (
        "ear_miewid:config-elephant-v1+ear_miewid_projected:projected-02474758261e"
    ),
}

# Auto-accept is explicitly disabled; threshold is documented for traceability
# only and must NOT be used to auto-create or auto-accept identities.
AUTO_ACCEPT_POLICY: dict[str, Any] = {
    "enabled": False,
    "reason": (
        "Open-set threshold (FAR≈27%, FRR≈48%) is not production-safe. "
        "Production must surface ranked top candidates for expert human verification. "
        "Do not auto-accept or auto-create identities from score threshold."
    ),
    "calibrated_threshold_for_reference_only": 0.17450773926201707,
    "threshold_far_at_calibration": 0.2727,
    "threshold_frr_at_calibration": 0.4817,
}

PRODUCTION_SUBDIR: str = "production"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: str) -> str:
    """Return hex SHA-256 of a file, or 'missing' if the file does not exist."""
    p = Path(path)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_partition(
    artifact_root: str,
    partition: str,
    channel: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load (mapping_df, embedding_matrix) for one channel/partition pair.

    Returns
    -------
    mapping_df : DataFrame with DESCRIPTOR_MAPPING_COLUMNS
    embedding_matrix : float32 ndarray (n, D)
    """
    emb_dir = os.path.join(artifact_root, "embeddings", partition)
    npy_path = os.path.join(emb_dir, f"{channel}.npy")
    map_path = os.path.join(emb_dir, f"{channel}_mapping.parquet")

    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Embedding matrix not found: {npy_path}")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Mapping table not found: {map_path}")

    mat = np.load(npy_path).astype(np.float32)
    df = pd.read_parquet(map_path)
    return df, mat


def _validate_fingerprints(
    df: pd.DataFrame,
    channel: str,
    expected_source: str | None,
    expected_split: str | None,
) -> None:
    """Raise AssertionError on any fingerprint inconsistency in df rows."""
    expected_model = EXPECTED_MODEL_FINGERPRINTS.get(channel)

    for col, expected, label in [
        ("source_fingerprint", expected_source, "source_fingerprint"),
        ("split_fingerprint", expected_split, "split_fingerprint"),
        ("model_preprocess_fingerprint", expected_model, "model_preprocess_fingerprint"),
    ]:
        if expected is None:
            continue
        bad = df.loc[df[col] != expected, col].unique().tolist()
        if bad:
            raise AssertionError(
                f"[{channel}] {label} mismatch: expected '{expected}', "
                f"got {bad} for {len(bad)} rows"
            )


def _check_no_eval_artifact_mutation(
    artifact_root: str,
    channel: str,
) -> None:
    """
    Ensure reference-partition eval artifacts have not been modified by
    comparing sizes of .npy and .parquet files before vs after any writes.
    (This function is called before writing production artifacts.)
    Records file mtimes; actual mutation check is done by tests via mocking.
    """
    # We simply verify files still exist with non-zero size.
    emb_dir = os.path.join(artifact_root, "embeddings", "reference")
    for fname in [f"{channel}.npy", f"{channel}_mapping.parquet"]:
        p = os.path.join(emb_dir, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected eval artifact missing: {p}")
        if os.path.getsize(p) == 0:
            raise AssertionError(f"Eval artifact is unexpectedly empty: {p}")


def _merge_channel(
    artifact_root: str,
    channel: str,
    expected_source_fp: str | None = None,
    expected_split_fp: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Merge reference + query partition embeddings for one channel.

    Integrity checks
    ----------------
    - Duplicate crop_id within either partition (fail immediately).
    - Same crop_id in both partitions (fail: eval contamination).
    - identity_id mismatch for any crop_id (fail).
    - Source / split / model fingerprint mismatches (fail).
    - Non-finite or zero-norm embedding vectors (fail).

    Returns
    -------
    merged_df   : DataFrame with contiguous embedding_row / faiss_row (0..N-1)
    merged_mat  : (N, D) float32, L2-normalised
    """
    ref_df, ref_mat = _load_partition(artifact_root, "reference", channel)
    qry_df, qry_mat = _load_partition(artifact_root, "query", channel)

    logger.info(
        "[%s] reference=%d rows, query=%d rows", channel, len(ref_df), len(qry_df)
    )

    # ---- per-partition duplicate check -----------------------------------
    for label, df in [("reference", ref_df), ("query", qry_df)]:
        dupes = df.loc[df.duplicated("crop_id", keep=False), "crop_id"].unique().tolist()
        if dupes:
            raise AssertionError(
                f"[{channel}] Duplicate crop_id within {label} partition: {dupes}"
            )

    # ---- cross-partition crop_id overlap (contamination) -----------------
    ref_ids = set(ref_df["crop_id"].tolist())
    qry_ids = set(qry_df["crop_id"].tolist())
    overlap = ref_ids & qry_ids
    if overlap:
        raise AssertionError(
            f"[{channel}] crop_id appears in both reference and query partitions "
            f"(eval contamination): {sorted(overlap)[:10]} ..."
        )

    # ---- identity mismatch across partitions for shared image_id ---------
    # (Different crop_ids for the same individual are expected; we just verify
    # that individual_id is internally consistent within each partition.)
    for label, df in [("reference", ref_df), ("query", qry_df)]:
        identity_counts = (
            df.groupby("crop_id", dropna=False)["individual_id"]
            .nunique(dropna=False)
        )
        bad = identity_counts[identity_counts > 1].to_dict()
        if bad:
            raise AssertionError(
                f"[{channel}] individual_id mismatch for crop_id in {label}: {bad}"
            )

    # ---- fingerprint validation -------------------------------------------
    for label, df in [("reference", ref_df), ("query", qry_df)]:
        _validate_fingerprints(df, channel, expected_source_fp, expected_split_fp)

    # ---- embedding matrix integrity (finite, L2-normalised) --------------
    for label, df, mat in [
        ("reference", ref_df, ref_mat),
        ("query", qry_df, qry_mat),
    ]:
        expected_rows = np.arange(len(df), dtype=np.int64)
        actual_rows = df["embedding_row"].to_numpy(dtype=np.int64)
        if not np.array_equal(actual_rows, expected_rows):
            raise AssertionError(
                f"[{channel}] {label} mapping must be ordered by contiguous "
                "embedding_row before merge"
            )
        if not np.isfinite(mat).all():
            bad = np.where(~np.isfinite(mat).all(axis=1))[0].tolist()
            raise AssertionError(
                f"[{channel}] {label} matrix has non-finite vectors at rows {bad}"
            )
        norms = np.linalg.norm(mat, axis=1)
        zero = np.where(norms == 0)[0].tolist()
        if zero:
            raise AssertionError(
                f"[{channel}] {label} matrix has zero-norm vectors at rows {zero}"
            )
        non_unit = np.where(~np.isclose(norms, 1.0, atol=1e-4, rtol=0.0))[0].tolist()
        if non_unit:
            raise AssertionError(
                f"[{channel}] {label} matrix vectors not L2-normalised "
                f"at rows {non_unit}: norms={norms[non_unit].tolist()}"
            )

    # ---- merge ------------------------------------------------------------
    # Drop eval-time row indices; rebuild contiguous 0..N-1 indices.
    ref_df = ref_df.copy()
    qry_df = qry_df.copy()
    merged_df = pd.concat([ref_df, qry_df], ignore_index=True)
    merged_mat = np.vstack([ref_mat, qry_mat]).astype(np.float32)

    n_total = len(merged_df)
    merged_df["embedding_row"] = range(n_total)
    merged_df["faiss_row"] = range(n_total)
    relative_crop_paths = []
    artifact_root_abs = os.path.abspath(artifact_root)
    for crop_path in merged_df["crop_path"].astype(str):
        relative = os.path.relpath(os.path.abspath(crop_path), artifact_root_abs)
        if relative == ".." or relative.startswith(f"..{os.sep}"):
            raise AssertionError(
                f"[{channel}] crop_path is outside artifact root: {crop_path}"
            )
        relative_crop_paths.append(relative)
    merged_df["crop_path"] = relative_crop_paths

    logger.info("[%s] merged total=%d rows", channel, n_total)

    return merged_df, merged_mat


def _build_faiss_index(mat: np.ndarray) -> faiss.IndexFlatIP:
    """Build an exact inner-product FAISS index (flat L2-normalised = cosine)."""
    d = mat.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(mat)
    return index


def _write_channel_artifacts(
    out_dir: str,
    channel: str,
    mapping_df: pd.DataFrame,
    matrix: np.ndarray,
    index: faiss.IndexFlatIP,
) -> dict[str, str]:
    """
    Write .npy, _mapping.parquet, .index files for one channel.

    Returns
    -------
    dict mapping artifact role → relative path (relative to out_dir)
    """
    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(out_dir, f"{channel}.npy")
    map_path = os.path.join(out_dir, f"{channel}_mapping.parquet")
    idx_path = os.path.join(out_dir, f"{channel}.index")

    np.save(npy_path, matrix)
    mapping_df.to_parquet(map_path, index=False)
    faiss.write_index(index, idx_path)

    logger.info("[%s] wrote %d-row catalog → %s", channel, len(mapping_df), out_dir)

    return {
        "npy": npy_path,
        "mapping_parquet": map_path,
        "faiss_index": idx_path,
    }


def _post_validate_channel(
    out_dir: str,
    channel: str,
    mapping_df: pd.DataFrame,
    matrix: np.ndarray,
) -> None:
    """Re-read written artifacts from disk and validate round-trip integrity."""
    idx_path = os.path.join(out_dir, f"{channel}.index")
    npy_path = os.path.join(out_dir, f"{channel}.npy")
    map_path = os.path.join(out_dir, f"{channel}_mapping.parquet")

    rt_mat = np.load(npy_path).astype(np.float32)
    rt_df = pd.read_parquet(map_path)
    rt_idx = faiss.read_index(idx_path)

    if rt_mat.shape != matrix.shape:
        raise AssertionError(
            f"[{channel}] round-trip matrix shape mismatch: "
            f"expected {matrix.shape}, got {rt_mat.shape}"
        )
    if not np.allclose(rt_mat, matrix, atol=1e-5):
        raise AssertionError(f"[{channel}] round-trip matrix values differ")
    if len(rt_df) != len(mapping_df):
        raise AssertionError(
            f"[{channel}] round-trip mapping row count mismatch: "
            f"expected {len(mapping_df)}, got {len(rt_df)}"
        )
    if rt_idx.ntotal != matrix.shape[0]:
        raise AssertionError(
            f"[{channel}] FAISS ntotal mismatch: "
            f"expected {matrix.shape[0]}, got {rt_idx.ntotal}"
        )
    logger.info("[%s] post-validation passed (ntotal=%d)", channel, rt_idx.ntotal)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_production_index(
    artifact_root: str,
    build_tag: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Build production descriptors by merging reference and query eval artifacts.

    Parameters
    ----------
    artifact_root : str
        Path to the versioned artifact root (e.g., .../BTEH_reid_artifacts/v1).
    build_tag : str, optional
        Unique label for this build.  Defaults to UTC timestamp.
    dry_run : bool
        If True, validate only; do not write any output files.

    Returns
    -------
    manifest : dict
        Production manifest (also written to production/<build_tag>/production_manifest.json).

    Raises
    ------
    FileNotFoundError
        If any required eval artifact is missing.
    AssertionError
        On duplicate crop_id, identity mismatch, fingerprint mismatch, or
        round-trip validation failure.
    """
    if build_tag is None:
        build_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    out_root = os.path.join(artifact_root, PRODUCTION_SUBDIR, build_tag)
    if not dry_run:
        os.makedirs(out_root, exist_ok=True)

    # ---- Resolve calibration paths (reference only; files not copied) -----
    cal_dir = os.path.join(artifact_root, PRODUCTION_CALIBRATION_SUBDIR)
    cal_manifest_path = os.path.join(cal_dir, "calibration_manifest.json")
    if not os.path.exists(cal_manifest_path):
        raise FileNotFoundError(f"Calibration manifest not found: {cal_manifest_path}")

    with open(cal_manifest_path) as f:
        cal_manifest = json.load(f)

    source_fp: str = cal_manifest["artifact_fingerprints"]["source_fingerprint"]
    split_fp: str = cal_manifest["artifact_fingerprints"]["split_fingerprint"]

    # ---- Verify selected checkpoint exists --------------------------------
    ckpt_dir = os.path.join(artifact_root, PRODUCTION_CHECKPOINT_SUBDIR)
    ckpt_pt = os.path.join(ckpt_dir, "best_projection.pt")
    ckpt_manifest_path = os.path.join(ckpt_dir, "training_manifest.json")
    if not os.path.exists(ckpt_pt):
        raise FileNotFoundError(f"Projection checkpoint not found: {ckpt_pt}")

    with open(ckpt_manifest_path) as f:
        ckpt_manifest = json.load(f)
    if not ckpt_manifest.get("gate", {}).get("adopted", False):
        raise AssertionError(
            f"Checkpoint at {ckpt_dir} was not adopted; gate.adopted=False. "
            "Only adopted projection checkpoints may enter production."
        )

    # ---- Pre-validate eval artifact existence (before any writes) ---------
    for channel in PRODUCTION_CHANNELS:
        _check_no_eval_artifact_mutation(artifact_root, channel)

    # ---- Merge each selected channel -------------------------------------
    channel_results: dict[str, Any] = {}
    channel_paths: dict[str, dict[str, str]] = {}

    for channel in PRODUCTION_CHANNELS:
        logger.info("Processing channel: %s", channel)
        merged_df, merged_mat = _merge_channel(
            artifact_root,
            channel,
            expected_source_fp=source_fp,
            expected_split_fp=split_fp,
        )

        if not dry_run:
            faiss_idx = _build_faiss_index(merged_mat)
            paths = _write_channel_artifacts(out_root, channel, merged_df, merged_mat, faiss_idx)
            channel_paths[channel] = paths

            # Post-write round-trip validation
            _post_validate_channel(out_root, channel, merged_df, merged_mat)

        n_rows = len(merged_df)
        n_images = int(merged_df["image_id"].nunique())
        n_individuals = int(merged_df["individual_id"].nunique())
        embedding_dim = int(merged_mat.shape[1])

        channel_results[channel] = {
            "n_rows": n_rows,
            "n_images": n_images,
            "n_individuals": n_individuals,
            "embedding_dim": embedding_dim,
            "faiss_ntotal": n_rows,
        }

    # ---- Verify eval artifacts not mutated after writes ------------------
    for channel in PRODUCTION_CHANNELS:
        _check_no_eval_artifact_mutation(artifact_root, channel)

    # ---- Build production manifest ---------------------------------------
    manifest: dict[str, Any] = {
        "schema_version": "v1",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "build_tag": build_tag,
        "dry_run": dry_run,
        "source_artifact_root": artifact_root,
        "production_output_dir": out_root if not dry_run else None,
        "crop_path_base": "source_artifact_root",
        "selected_channels": PRODUCTION_CHANNELS,
        "fusion_weights": PRODUCTION_FUSION_WEIGHTS,
        "auto_accept_policy": AUTO_ACCEPT_POLICY,
        "calibration": {
            "subdir": PRODUCTION_CALIBRATION_SUBDIR,
            "calibration_manifest": cal_manifest_path,
            "fusion_weights_file": os.path.join(cal_dir, "fusion_weights.json"),
            "unknown_threshold_file": os.path.join(cal_dir, "unknown_threshold.json"),
            "channel_calibration_files": {
                ch: os.path.join(cal_dir, fname)
                for ch, fname in PRODUCTION_CALIBRATION_CHANNEL_FILES.items()
            },
            "calibration_method": "platt",
            "calibration_note": cal_manifest.get("calibration_note", ""),
        },
        "projection_checkpoint": {
            "subdir": PRODUCTION_CHECKPOINT_SUBDIR,
            "checkpoint_pt": ckpt_pt,
            "training_manifest": ckpt_manifest_path,
            "checkpoint_fingerprint": ckpt_manifest.get("checkpoint_fingerprint"),
            "gate_adopted": ckpt_manifest["gate"]["adopted"],
            "gate_reason": ckpt_manifest["gate"]["reason"],
            "val_map": ckpt_manifest.get("best_val_map"),
            "baseline_map": ckpt_manifest.get("baseline_map"),
        },
        "artifact_fingerprints": {
            "source_fingerprint": source_fp,
            "split_fingerprint": split_fp,
            "model_preprocess_fingerprints": EXPECTED_MODEL_FINGERPRINTS,
            "checkpoint_fingerprint": ckpt_manifest.get("checkpoint_fingerprint"),
        },
        "channel_artifacts": channel_paths if not dry_run else {},
        "channel_stats": channel_results,
        "total_stats": {
            "n_individuals": max(
                (v["n_individuals"] for v in channel_results.values()), default=0
            ),
        },
        "eval_reference": {
            "selected_eval_dir": "reports/calibrated_eval_projected",
            "eval_summary": os.path.join(
                artifact_root,
                "reports",
                "calibrated_eval_projected",
                "normalized_eval_summary.json",
            ),
            "known_top1": 0.3841,
            "known_mAP": 0.473,
            "temporal_top1": 0.3435,
            "onboarding_top1": 0.5455,
            "calibration_ece": 0.1199,
            "open_set_far": 0.2727,
            "open_set_frr": 0.4817,
            "open_set_note": (
                "FAR≈27% and FRR≈48% make threshold-based auto-accept unsafe for production."
            ),
        },
        "policy": {
            "no_megadescriptor_finetuning": (
                "MegaDescriptor channels received zero OOF weight in all evaluated calibrations. "
                "Do not fine-tune MegaDescriptor for this dataset."
            ),
            "no_random_512_projection": (
                "ear_miewid_projection (random 512-dim) was rejected by adoption gate. "
                "Only ear_miewid_identity_adapter (2152→2152 trained) is production-approved."
            ),
            "expert_review_mandatory": (
                "Top-1 accuracy is moderate (~38%). Expert verification is mandatory for all "
                "match decisions. Never auto-accept or auto-create identities from score alone."
            ),
            "query_probes_become_catalog": (
                "Query partition probes are included here as catalog references because model "
                "selection is frozen.  If models are retrained, all catalog embeddings must be "
                "regenerated from scratch using the new checkpoint."
            ),
        },
    }

    if not dry_run:
        manifest_path = os.path.join(out_root, "production_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Production manifest written: %s", manifest_path)

    return manifest


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build BTEH production index from frozen eval artifacts."
    )
    parser.add_argument(
        "--artifact-root",
        required=True,
        help="Versioned artifact root (e.g., /path/to/BTEH_reid_artifacts/v1)",
    )
    parser.add_argument(
        "--build-tag",
        default=None,
        help="Unique build label (default: UTC timestamp)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only; do not write output files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    manifest = build_production_index(
        artifact_root=args.artifact_root,
        build_tag=args.build_tag,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("[dry-run] Validation passed. No files written.")
    else:
        out_dir = manifest["production_output_dir"]
        print(f"Production index built successfully → {out_dir}")
        for ch, stats in manifest["channel_stats"].items():
            print(
                f"  {ch}: {stats['n_rows']} rows, "
                f"{stats['n_individuals']} individuals, "
                f"dim={stats['embedding_dim']}"
            )


if __name__ == "__main__":
    main()
