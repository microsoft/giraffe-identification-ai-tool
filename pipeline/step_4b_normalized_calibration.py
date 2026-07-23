# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Step 4b (normalized) – Grouped out-of-fold calibration and fusion weight fitting
using the BTEH normalized artifact schema.

This module replaces the all-pairs calibration approach in the legacy
step_4b_train_calibration.py for the normalized BTEH pipeline.  It operates
exclusively on gallery/reference artifacts and never touches probe images.

Usage
-----
    python -m pipeline.step_4b_normalized_calibration \\
        --artifact-root /path/to/BTEH_reid_artifacts/v1 \\
        --splits-file /path/to/bteh_splits.parquet \\
        --out-dir /path/to/BTEH_reid_artifacts/v1/calibration \\
        [--channels megadescriptor miewid ear_megadescriptor ear_miewid] \\
        [--grid-step 0.05]

Output
------
  <out-dir>/
    {channel}.pkl              – Calibrator objects (one per channel)
    fusion_weights.json        – Fitted fusion weights + diagnostics
    unknown_threshold.json     – Unknown/open-set threshold + diagnostics
    calibration_manifest.json  – Full provenance, support counts, fold
                                 diagnostics, schema/split fingerprints

Legacy giraffe behaviour is isolated in step_4b_train_calibration.py and
must not be imported from this module.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Repository root on path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from configs.config_bteh import (
    ARTIFACT_SCHEMA_VERSION,
    ARTIFACT_VERSION_ROOT,
    CALIBRATION_SUBDIR_BTEH,
    EMBEDDINGS_SUBDIR_BTEH,
    SPLITS_FILENAME,
    SPLITS_SUBDIR,
)
from configs.config_elephant import (
    ACTIVE_DESCRIPTORS,
    MIN_POSITIVE_PAIRS_FOR_ISOTONIC,
)
from models.calibration import Calibrator
from models.identity_fusion import (
    build_oof_identity_scores,
    estimate_unknown_threshold,
    fit_fusion_weights,
    simulate_gallery_unknown_scores,
    _apply_weights_and_rank,
)
from models.oof_calibration import (
    CalibrationSupportError,
    ChannelOOFResult,
    compute_oof_scores,
    roc_auc,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Artifact loading helpers
# ---------------------------------------------------------------------------

def _resolve_embedding_dir(artifact_root: Path, partition: str = "reference") -> Path:
    """Return the directory that holds {descriptor}.npy and *_mapping.parquet."""
    return artifact_root / EMBEDDINGS_SUBDIR_BTEH / partition


def _load_embedding(emb_dir: Path, channel: str) -> np.ndarray:
    npy_path = emb_dir / f"{channel}.npy"
    if not npy_path.is_file():
        raise FileNotFoundError(
            f"Embedding file not found for channel '{channel}': {npy_path}"
        )
    mat = np.load(str(npy_path)).astype(np.float32)
    logger.info("Loaded embedding %s: shape %s", npy_path.name, mat.shape)
    return mat


def _load_descriptor_mapping(emb_dir: Path, channel: str) -> pd.DataFrame:
    parquet_path = emb_dir / f"{channel}_mapping.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Descriptor mapping not found for channel '{channel}': {parquet_path}"
        )
    dm = pd.read_parquet(str(parquet_path))
    required = {"image_id", "individual_id", "embedding_row", "crop_kind"}
    missing = required - set(dm.columns)
    if missing:
        raise ValueError(
            f"Descriptor mapping for '{channel}' is missing columns: {sorted(missing)}"
        )
    logger.info(
        "Loaded descriptor mapping %s: %d rows", parquet_path.name, len(dm)
    )
    return dm


def _verify_fingerprint_consistency(
    channels: List[str],
    descriptor_mappings: Dict[str, pd.DataFrame],
) -> Dict:
    """Verify shared data fingerprints and channel-local model fingerprints."""
    shared_fields = [
        "schema_version",
        "source_fingerprint",
        "split_fingerprint",
    ]
    collected: Dict[str, set] = {f: set() for f in shared_fields}
    model_fingerprints: Dict[str, str | None] = {}

    for ch in channels:
        dm = descriptor_mappings.get(ch)
        if dm is None or dm.empty:
            continue
        for f in shared_fields:
            if f in dm.columns:
                vals = set(dm[f].dropna().astype(str).unique())
                collected[f].update(vals)
        model_values = {
            value
            for value in dm.get(
                "model_preprocess_fingerprint", pd.Series(dtype=str)
            ).dropna().astype(str).unique()
            if value and value != "nan"
        }
        if len(model_values) > 1:
            raise AssertionError(
                f"Descriptor mapping {ch!r} has multiple model fingerprints: "
                f"{sorted(model_values)}"
            )
        model_fingerprints[ch] = (
            next(iter(model_values)) if model_values else None
        )

    mismatches = []
    fingerprints = {}
    for f, vals in collected.items():
        clean = {v for v in vals if v and v != "nan"}
        if len(clean) > 1:
            mismatches.append(f"{f}: {sorted(clean)}")
        fingerprints[f] = sorted(clean)[0] if clean else None

    if mismatches:
        raise AssertionError(
            "Descriptor mapping fingerprint mismatch across channels: "
            + "; ".join(mismatches)
        )

    fingerprints["model_preprocess_fingerprints"] = model_fingerprints
    return fingerprints


def _load_gallery_with_sessions(
    splits_path: Path,
    artifact_root: Path,
    partition: str = "reference",
) -> pd.DataFrame:
    """
    Build the gallery image DataFrame with session_id, fold, and split.

    Merges the crop manifest (which has split/fold) with the splits parquet
    (which has session_id).
    """
    splits_df = pd.read_parquet(str(splits_path))
    emb_dir = _resolve_embedding_dir(artifact_root, partition)
    crop_manifest_path = emb_dir / "crop_manifest.parquet"

    if not crop_manifest_path.is_file():
        raise FileNotFoundError(
            f"Crop manifest not found at expected path: {crop_manifest_path}"
        )
    crop_df = pd.read_parquet(str(crop_manifest_path))

    # Keep only gallery rows (one row per accepted body crop, deduped by image_id).
    gallery_crop = crop_df[
        (crop_df["split"] == "gallery") & (crop_df["crop_kind"] == "body")
    ].copy()
    # If no body crops exist with split='gallery', fall back to all crop kinds.
    if gallery_crop.empty:
        gallery_crop = crop_df[crop_df["split"] == "gallery"].copy()

    # Deduplicate to one row per image_id for the image-level gallery view.
    gallery_images = (
        gallery_crop[["image_id", "individual_id"]]
        .drop_duplicates(subset="image_id")
        .copy()
    )

    # Merge session_id and fold from splits.
    splits_cols = ["image_id", "session_id"]
    if "fold" in splits_df.columns:
        splits_cols.append("fold")
    gallery_with_sessions = gallery_images.merge(
        splits_df[splits_cols].drop_duplicates(subset="image_id"),
        on="image_id",
        how="left",
    )

    missing_sessions = gallery_with_sessions["session_id"].isna().sum()
    if missing_sessions > 0:
        logger.warning(
            "%d gallery images have no session_id in splits manifest.",
            missing_sessions,
        )
        gallery_with_sessions["session_id"] = (
            gallery_with_sessions["session_id"].fillna("unknown_session")
        )

    gallery_with_sessions["image_id"] = gallery_with_sessions["image_id"].astype(str)
    gallery_with_sessions["individual_id"] = gallery_with_sessions["individual_id"].astype(str)
    gallery_with_sessions["session_id"] = gallery_with_sessions["session_id"].astype(str)

    logger.info(
        "Gallery: %d images, %d individuals, %d sessions",
        len(gallery_with_sessions),
        gallery_with_sessions["individual_id"].nunique(),
        gallery_with_sessions["session_id"].nunique(),
    )
    return gallery_with_sessions


# ---------------------------------------------------------------------------
# Calibration manifest helpers
# ---------------------------------------------------------------------------

def _fold_diagnostics_to_dict(ch_result: ChannelOOFResult) -> dict:
    included = [d for d in ch_result.fold_diagnostics if d.included]
    skipped = [d for d in ch_result.fold_diagnostics if not d.included]
    skip_reasons: Dict[str, int] = {}
    for d in skipped:
        reason = d.exclusion_reason or "unknown"
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    return {
        "n_folds_total": len(ch_result.fold_diagnostics),
        "n_folds_included": ch_result.n_included_folds,
        "n_folds_skipped": ch_result.n_skipped_folds,
        "skip_reasons": skip_reasons,
        "included_fold_sample": [
            {
                "session_id": d.session_id,
                "n_pos": d.n_pos_pairs,
                "n_neg": d.n_neg_pairs,
            }
            for d in included[:10]  # sample for readability
        ],
    }


# ---------------------------------------------------------------------------
# Main calibration pipeline
# ---------------------------------------------------------------------------

def run_normalized_calibration(
    artifact_root: Path,
    splits_path: Path,
    out_dir: Path,
    channels: List[str],
    partition: str = "reference",
    grid_step: float = 0.05,
    hard_neg_k: int = 5,
    calibration_method: str = "auto",
) -> dict:
    """
    Full OOF calibration pipeline.

    Returns the calibration manifest dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. Load gallery
    # -----------------------------------------------------------------
    gallery_df = _load_gallery_with_sessions(splits_path, artifact_root, partition)

    # -----------------------------------------------------------------
    # 2. Load embeddings and mappings
    # -----------------------------------------------------------------
    emb_dir = _resolve_embedding_dir(artifact_root, partition)
    embedding_matrices: Dict[str, np.ndarray] = {}
    descriptor_mappings: Dict[str, pd.DataFrame] = {}

    for ch in channels:
        try:
            embedding_matrices[ch] = _load_embedding(emb_dir, ch)
            descriptor_mappings[ch] = _load_descriptor_mapping(emb_dir, ch)
        except FileNotFoundError as exc:
            raise CalibrationSupportError(
                f"Channel '{ch}' is enabled but its artifacts are missing: {exc}"
            ) from exc

    # -----------------------------------------------------------------
    # 3. Verify fingerprint consistency
    # -----------------------------------------------------------------
    fingerprints = _verify_fingerprint_consistency(channels, descriptor_mappings)
    logger.info("Artifact fingerprints: %s", fingerprints)

    # -----------------------------------------------------------------
    # 4. Grouped OOF scoring
    # -----------------------------------------------------------------
    logger.info("Running grouped OOF scoring ...")
    oof_results: Dict[str, ChannelOOFResult] = compute_oof_scores(
        gallery_df,
        descriptor_mappings,
        embedding_matrices,
        hard_neg_k=hard_neg_k,
    )

    # -----------------------------------------------------------------
    # 5. Fit one Calibrator per channel
    # -----------------------------------------------------------------
    calibrators: Dict[str, Calibrator] = {}
    manifest_channels: Dict[str, dict] = {}

    for ch in channels:
        ch_result = oof_results[ch]
        scores_arr = np.array(ch_result.scores, dtype=np.float64)
        labels_arr = np.array(ch_result.labels, dtype=np.float64)

        n_pos = int(labels_arr.sum())
        n_neg = int((1 - labels_arr).sum())

        auc_val = roc_auc(ch_result.scores, ch_result.labels)

        logger.info(
            "[%s] OOF pairs: %d total, %d pos, %d neg | AUC=%.4f",
            ch, len(scores_arr), n_pos, n_neg,
            auc_val if not np.isnan(auc_val) else float("nan"),
        )

        try:
            cal = Calibrator()
            cal.fit(scores_arr, labels_arr, method=calibration_method)
        except ValueError as exc:
            raise CalibrationSupportError(
                f"Channel '{ch}': calibrator fit failed: {exc}"
            ) from exc

        save_path = out_dir / f"{ch}.pkl"
        cal.save(str(save_path))
        calibrators[ch] = cal

        logger.info(
            "[%s] method=%s reason=%s | saved→%s",
            ch, cal.method, cal.fit_reason, save_path,
        )

        manifest_channels[ch] = {
            "method": cal.method,
            "fit_reason": cal.fit_reason,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_total": len(scores_arr),
            "auc": round(float(auc_val), 6) if not np.isnan(auc_val) else None,
            "fold_diagnostics": _fold_diagnostics_to_dict(ch_result),
        }

    # -----------------------------------------------------------------
    # 6. Build OOF identity-level scores for weight fitting
    # -----------------------------------------------------------------
    logger.info("Building OOF identity-level scores for weight fitting ...")
    oof_identity_results = build_oof_identity_scores(
        gallery_df,
        descriptor_mappings,
        embedding_matrices,
        calibrators,
        all_channels=channels,
    )

    # -----------------------------------------------------------------
    # 7. Fit fusion weights
    # -----------------------------------------------------------------
    logger.info("Fitting fusion weights (grid_step=%.2f) ...", grid_step)
    best_weights, weight_diag = fit_fusion_weights(
        oof_identity_results,
        all_channels=channels,
        grid_step=grid_step,
    )

    weights_path = out_dir / "fusion_weights.json"
    weights_payload = {"weights": best_weights, "diagnostics": weight_diag}
    with open(str(weights_path), "w") as fh:
        json.dump(weights_payload, fh, indent=2)
    logger.info("Fusion weights saved to %s", weights_path)

    # -----------------------------------------------------------------
    # 8. Unknown threshold estimation
    # -----------------------------------------------------------------
    # Apply fitted fusion weights to the raw OOF identity results so that
    # fused_score values are meaningful (they were 0.0 in the raw results
    # because weights were not yet available during build_oof_identity_scores).
    weighted_oof = _apply_weights_and_rank(oof_identity_results, best_weights, channels)

    # Known OOF queries: those where the query identity was present in the
    # OOF rest-gallery fold (identity_in_oof_gallery=True).  Queries for
    # identities that only appear in one session (rest gallery lacks them)
    # are excluded from the known distribution.
    known_oof = [qr for qr in weighted_oof if qr.identity_in_oof_gallery is True]

    # Simulated unknown trials: score each gallery identity against a gallery
    # from which all its own crops have been excluded.  This yields a top
    # non-match fused confidence distribution that does not require probe data.
    logger.info("Simulating gallery unknown trials via identity removal ...")
    try:
        unknown_oof = simulate_gallery_unknown_scores(
            gallery_df,
            descriptor_mappings,
            embedding_matrices,
            calibrators,
            best_weights,
            channels,
        )
    except CalibrationSupportError as exc:
        logger.error(
            "Unknown trial simulation failed (hard error): %s", exc
        )
        sys.exit(2)

    logger.info(
        "Threshold estimation: %d known OOF queries, %d unknown trials.",
        len(known_oof),
        len(unknown_oof),
    )
    try:
        threshold, thresh_diag = estimate_unknown_threshold(known_oof, unknown_oof)
    except CalibrationSupportError as exc:
        logger.error(
            "Unknown threshold estimation failed (hard error): %s", exc
        )
        sys.exit(2)

    thresh_path = out_dir / "unknown_threshold.json"
    with open(str(thresh_path), "w") as fh:
        json.dump({"threshold": threshold, "diagnostics": thresh_diag}, fh, indent=2)
    logger.info("Unknown threshold saved to %s (threshold=%.4f)", thresh_path, threshold)

    # -----------------------------------------------------------------
    # 9. Write calibration manifest
    # -----------------------------------------------------------------
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "partition": partition,
        "artifact_root": str(artifact_root),
        "splits_path": str(splits_path),
        "channels": channels,
        "n_gallery_images": len(gallery_df),
        "n_gallery_individuals": int(gallery_df["individual_id"].nunique()),
        "n_gallery_sessions": int(gallery_df["session_id"].nunique()),
        "artifact_fingerprints": fingerprints,
        "channel_results": manifest_channels,
        "fusion_weights": best_weights,
        "fusion_weight_diagnostics": weight_diag,
        "unknown_threshold": threshold,
        "unknown_threshold_diagnostics": thresh_diag,
        "grid_step": grid_step,
        "hard_neg_k": hard_neg_k,
        "min_positive_for_isotonic": MIN_POSITIVE_PAIRS_FOR_ISOTONIC,
        "calibration_note": (
            "Temperature expit(s/T) replaced by Platt (logistic) scaling as "
            "fallback. Temperature cannot represent probs < 0.5 for positive "
            "cosine scores; Platt scaling with intercept covers full [0,1] range."
        ),
    }

    manifest_path = out_dir / "calibration_manifest.json"
    with open(str(manifest_path), "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Calibration manifest saved to %s", manifest_path)

    return manifest


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Step 4b (normalized): grouped OOF calibration and fusion "
            "weight fitting for BTEH normalized artifacts."
        )
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ARTIFACT_VERSION_ROOT,
        help=f"Versioned artifact root (default: {ARTIFACT_VERSION_ROOT})",
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=None,
        help="Path to bteh_splits.parquet (default: <artifact-root>/splits/bteh_splits.parquet)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <artifact-root>/calibration)",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=ACTIVE_DESCRIPTORS,
        help=f"Channels to calibrate (default: {ACTIVE_DESCRIPTORS})",
    )
    parser.add_argument(
        "--partition",
        default="reference",
        help="Reference partition name under embeddings/ (default: reference)",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.05,
        help="Weight-grid step for fusion weight fitting (default: 0.05)",
    )
    parser.add_argument(
        "--hard-neg-k",
        type=int,
        default=5,
        help="Hard negatives per pseudo-query per channel (default: 5)",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["auto", "isotonic", "platt"],
        default="auto",
        help="Per-channel calibration method (default: auto).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    artifact_root = args.artifact_root
    splits_path = args.splits_file or (
        artifact_root / SPLITS_SUBDIR / SPLITS_FILENAME
    )
    out_dir = args.out_dir or (artifact_root / CALIBRATION_SUBDIR_BTEH)

    if not artifact_root.is_dir():
        logger.error("Artifact root not found: %s", artifact_root)
        sys.exit(1)
    if not splits_path.is_file():
        logger.error("Splits file not found: %s", splits_path)
        sys.exit(1)

    logger.info("Step 4b (normalized) starting.")
    logger.info("  artifact_root=%s", artifact_root)
    logger.info("  splits_path=%s", splits_path)
    logger.info("  out_dir=%s", out_dir)
    logger.info("  channels=%s", args.channels)

    try:
        manifest = run_normalized_calibration(
            artifact_root=artifact_root,
            splits_path=splits_path,
            out_dir=out_dir,
            channels=args.channels,
            partition=args.partition,
            grid_step=args.grid_step,
            hard_neg_k=args.hard_neg_k,
            calibration_method=args.calibration_method,
        )
    except CalibrationSupportError as exc:
        logger.error("Calibration support error (hard fail): %s", exc)
        sys.exit(2)
    except AssertionError as exc:
        logger.error("Integrity assertion failed: %s", exc)
        sys.exit(3)

    logger.info("Step 4b (normalized) complete.")
    logger.info(
        "Fusion weights: %s",
        {k: round(v, 4) for k, v in manifest["fusion_weights"].items()},
    )
    logger.info(
        "Unknown threshold: %.4f", manifest["unknown_threshold"]
    )


if __name__ == "__main__":
    main()
