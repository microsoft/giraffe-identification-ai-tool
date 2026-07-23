# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Step 4c – Normalized identity-level fusion evaluation on fixed query probes.

This module evaluates the calibrated, fused identity-level matcher against
the probe/fixed-query split.  It never touches gallery OOF predictions.

Usage
-----
    python -m pipeline.step_4c_normalized_eval \\
        --artifact-root /path/to/BTEH_reid_artifacts/v1 \\
        --splits-file /path/to/bteh_splits.parquet \\
        --calibration-dir /path/to/BTEH_reid_artifacts/v1/calibration \\
        [--query-partition query] \\
        [--ref-partition reference] \\
        [--channels megadescriptor miewid ear_megadescriptor ear_miewid] \\
        [--out-dir /path/to/BTEH_reid_artifacts/v1/reports]

Output (under out-dir)
----------------------
  normalized_eval_rankings.parquet    – per-query ranked identity table
  normalized_eval_summary.json        – aggregate metrics
  normalized_eval_channel_breakdown.json – per-channel ablations
  normalized_eval_unknown_report.json – open-set unknown detection metrics

Metric glossary
---------------
  top1 / top5 / mAP / CMC  – standard retrieval metrics for known queries
  precision / recall / FAR / FRR  – open-set detection metrics
  ECE / Brier  – calibration quality metrics
  coverage  – fraction of probe queries with at least one channel present
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from configs.config_bteh import (
    ARTIFACT_SCHEMA_VERSION,
    ARTIFACT_VERSION_ROOT,
    CALIBRATION_SUBDIR_BTEH,
    EMBEDDINGS_SUBDIR_BTEH,
    REPORTS_SUBDIR,
    SPLITS_FILENAME,
    SPLITS_SUBDIR,
)
from configs.config_elephant import ACTIVE_DESCRIPTORS
from models.calibration import Calibrator
from models.identity_fusion import (
    IdentityLevelScorer,
    IdentityScore,
    QueryResult,
    _average_precision,
    check_calibration_flatness,
    compute_map,
    compute_top1,
    compute_top5,
    simulate_probe_unknown_trials,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (histogram binning)."""
    if len(probs) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if not mask.any():
            continue
        avg_confidence = float(probs[mask].mean())
        avg_accuracy = float(labels[mask].mean())
        ece += mask.mean() * abs(avg_confidence - avg_accuracy)
    return float(ece)


def _compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score."""
    if len(probs) == 0:
        return float("nan")
    return float(np.mean((probs - labels) ** 2))


def _compute_cmc(results: List[QueryResult], max_rank: int = 20) -> List[float]:
    """Cumulative Match Characteristic curve up to max_rank."""
    cmc = np.zeros(max_rank, dtype=np.float64)
    count = 0
    for qr in results:
        if qr.query_individual_id is None or qr.unknown_query:
            continue
        count += 1
        ranked_ids = [x.individual_id for x in qr.ranked_identities]
        for r in range(max_rank):
            if qr.query_individual_id in ranked_ids[: r + 1]:
                cmc[r:] = cmc[r:] + 1
                break
    if count > 0:
        cmc /= count
    return cmc.tolist()


# ---------------------------------------------------------------------------
# Channel-ablation evaluation
# ---------------------------------------------------------------------------

def _channel_ablation(
    results: List[QueryResult], channels: List[str]
) -> Dict[str, dict]:
    """
    Per-channel calibrated-score ablation using known retrieval queries only.

    Each channel is evaluated independently using its per-identity calibrated
    score (``channel_calibrated[ch]``) as the sole ranking signal.  Only
    queries where the channel is available for at least one candidate are
    included.  Simulated-unknown and open-set trials are excluded so the
    ablation reflects known-identity retrieval performance.
    """
    ablation: Dict[str, dict] = {}
    for ch in channels:
        ch_results_ranked = []
        for qr in results:
            # Known retrieval queries only: skip unknown/simulated-unknown probes.
            if qr.query_individual_id is None or qr.unknown_query or qr.simulated_unknown:
                continue
            single_ch_ranked = [
                IdentityScore(
                    individual_id=ident.individual_id,
                    channel_raw=ident.channel_raw,
                    channel_calibrated=ident.channel_calibrated,
                    channels_available=ident.channels_available,
                    fused_score=ident.channel_calibrated.get(ch, 0.0),
                )
                for ident in qr.ranked_identities
                if ch in ident.channels_available
            ]
            if not single_ch_ranked:
                continue
            single_ch_ranked.sort(key=lambda x: x.fused_score, reverse=True)
            ch_results_ranked.append(
                QueryResult(
                    query_image_id=qr.query_image_id,
                    query_individual_id=qr.query_individual_id,
                    ranked_identities=single_ch_ranked,
                    channels_present=[ch],
                    channels_absent=[c for c in channels if c != ch],
                )
            )

        n = len(ch_results_ranked)
        ablation[ch] = {
            "n_queries": n,
            "top1_calibrated": round(compute_top1(ch_results_ranked), 4) if n > 0 else None,
            "mAP_calibrated": round(compute_map(ch_results_ranked), 4) if n > 0 else None,
        }

    return ablation


def _compute_split_metrics(results: List[QueryResult]) -> dict:
    """
    Compute retrieval metrics for a list of QueryResult objects.

    Intended for a homogeneous list (e.g., all temporal probes or all
    onboarding probes).  Unknown/simulated-unknown queries are excluded.
    """
    known = [r for r in results if not r.unknown_query and not r.simulated_unknown
             and r.query_individual_id is not None]
    n = len(known)
    if n == 0:
        return {"n": 0, "top1": None, "top5": None, "mAP": None, "cmc@5": None, "cmc@10": None}

    top1 = round(compute_top1(known), 4)
    top5 = round(compute_top5(known), 4)
    mAP = round(compute_map(known), 4)
    cmc = _compute_cmc(known, max_rank=20)
    return {
        "n": n,
        "top1": top1,
        "top5": top5,
        "mAP": mAP,
        "cmc@5": round(cmc[4], 4) if len(cmc) > 4 else None,
        "cmc@10": round(cmc[9], 4) if len(cmc) > 9 else None,
    }



# ---------------------------------------------------------------------------
# Unknown/open-set detection
# ---------------------------------------------------------------------------

def _open_set_metrics(
    known_results: List[QueryResult],
    simulated_unknown_results: List[QueryResult],
    threshold: float,
) -> dict:
    """
    Compute FAR, FRR, precision, recall for open-set detection.

    Parameters
    ----------
    known_results : known-identity retrieval results (truth identity present
        in reference gallery; NOT simulated-unknown trials).
    simulated_unknown_results : results from ``simulate_probe_unknown_trials``,
        where truth identity has been explicitly removed from the reference.
        Every entry must have ``simulated_unknown=True``.
    threshold : acceptance threshold on fused_score.

    Notes
    -----
    - ``tp``: known query accepted (fused_score >= threshold).
    - ``fn``: known query rejected.
    - ``fp``: simulated-unknown query accepted (false alarm).
    - ``tn``: simulated-unknown query rejected.
    - Relabelling a query as "unknown" without removing its identity from the
      reference is NOT a valid unknown trial and must not be included here.
    """
    tp = fp = fn = tn = 0

    # Verify no simulated trials have their identity still indexed.
    n_invalid = sum(
        1 for qr in simulated_unknown_results
        if not qr.simulated_unknown
    )
    if n_invalid > 0:
        logger.warning(
            "_open_set_metrics: %d simulated-unknown results lack the "
            "simulated_unknown=True flag. Excluded from metrics.",
            n_invalid,
        )
        simulated_unknown_results = [r for r in simulated_unknown_results if r.simulated_unknown]

    for qr in known_results:
        if qr.simulated_unknown:
            continue
        accepted = bool(qr.ranked_identities and
                        qr.ranked_identities[0].fused_score >= threshold)
        if accepted:
            tp += 1
        else:
            fn += 1

    for qr in simulated_unknown_results:
        accepted = bool(qr.ranked_identities and
                        qr.ranked_identities[0].fused_score >= threshold)
        if accepted:
            fp += 1
        else:
            tn += 1

    far = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    frr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    # Confidence comparison: known vs simulated-unknown top-1 scores.
    known_scores = [
        qr.ranked_identities[0].fused_score
        for qr in known_results if qr.ranked_identities and not qr.simulated_unknown
    ]
    unk_scores = [
        qr.ranked_identities[0].fused_score
        for qr in simulated_unknown_results if qr.ranked_identities
    ]

    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "far": round(far, 4) if not np.isnan(far) else None,
        "frr": round(frr, 4) if not np.isnan(frr) else None,
        "precision": round(precision, 4) if not np.isnan(precision) else None,
        "recall": round(recall, 4) if not np.isnan(recall) else None,
        "n_known_trials": len(known_results),
        "n_simulated_unknown_trials": len(simulated_unknown_results),
        "known_mean_top1_score": round(float(np.mean(known_scores)), 4) if known_scores else None,
        "unknown_mean_top1_score": round(float(np.mean(unk_scores)), 4) if unk_scores else None,
        "provenance": "identity_removed_reference_scoring",
        "note": (
            "Unknown trials use explicit identity removal from reference. "
            "Truth identity is absent from every simulated-unknown candidate set."
        ),
    }


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def _load_query_emb_rows(
    emb_dir: Path,
    channels: List[str],
    query_image_ids: set,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    """Load descriptor mappings and embedding matrices for query partition."""
    descriptor_mappings: Dict[str, pd.DataFrame] = {}
    embedding_matrices: Dict[str, np.ndarray] = {}

    for ch in channels:
        mapping_path = emb_dir / f"{ch}_mapping.parquet"
        npy_path = emb_dir / f"{ch}.npy"

        if not mapping_path.is_file():
            logger.warning("Query descriptor mapping not found for '%s': %s", ch, mapping_path)
            continue
        if not npy_path.is_file():
            logger.warning("Query embedding not found for '%s': %s", ch, npy_path)
            continue

        dm = pd.read_parquet(str(mapping_path))
        dm["image_id"] = dm["image_id"].astype(str)
        dm = dm[dm["image_id"].isin(query_image_ids)].reset_index(drop=True)
        descriptor_mappings[ch] = dm

        mat = np.load(str(npy_path)).astype(np.float32)
        embedding_matrices[ch] = mat
        logger.info("Loaded query %s: %d rows, shape %s", ch, len(dm), mat.shape)

    return descriptor_mappings, embedding_matrices


def _build_query_emb_per_image(
    descriptor_mappings: Dict[str, pd.DataFrame],
    embedding_matrices: Dict[str, np.ndarray],
    channels: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Return {image_id: {channel: np.ndarray shape (n_crops, D)}}."""
    image_ch_emb: Dict[str, Dict[str, np.ndarray]] = {}

    for ch in channels:
        dm = descriptor_mappings.get(ch)
        mat = embedding_matrices.get(ch)
        if dm is None or mat is None or dm.empty:
            continue

        for _, row in dm.iterrows():
            iid = str(row["image_id"])
            emb_row = int(row["embedding_row"])
            vec = mat[emb_row:emb_row + 1]  # shape (1, D)
            if iid not in image_ch_emb:
                image_ch_emb[iid] = {}
            if ch not in image_ch_emb[iid]:
                image_ch_emb[iid][ch] = vec
            else:
                image_ch_emb[iid][ch] = np.vstack([image_ch_emb[iid][ch], vec])

    return image_ch_emb


def _load_calibrators(calib_dir: Path, channels: List[str]) -> Dict[str, Calibrator]:
    calibrators: Dict[str, Calibrator] = {}
    for ch in channels:
        pkl_path = calib_dir / f"{ch}.pkl"
        if pkl_path.is_file():
            cal = Calibrator().load(str(pkl_path))
            calibrators[ch] = cal
            logger.info("Loaded calibrator '%s' (method=%s)", ch, cal.method)
        else:
            logger.warning(
                "Calibrator not found for channel '%s': %s; using raw scores.",
                ch,
                pkl_path,
            )
    return calibrators


def _load_fusion_weights(
    calib_dir: Path,
    channels: List[str],
) -> Dict[str, float]:
    weights_path = calib_dir / "fusion_weights.json"
    if weights_path.is_file():
        with open(str(weights_path)) as fh:
            payload = json.load(fh)
        weights = payload.get("weights", {})
        # Validate.
        total = sum(weights.get(ch, 0.0) for ch in channels)
        if not np.isclose(total, 1.0, atol=0.01):
            logger.warning(
                "Loaded fusion weights sum to %.4f (not 1.0); normalising.", total
            )
            if total > 0:
                weights = {ch: weights.get(ch, 0.0) / total for ch in channels}
            else:
                weights = {ch: 1.0 / len(channels) for ch in channels}
        logger.info("Loaded fusion weights: %s", {k: round(v, 4) for k, v in weights.items()})
        return weights
    else:
        logger.warning(
            "fusion_weights.json not found in %s; using equal weights.", calib_dir
        )
        return {ch: 1.0 / len(channels) for ch in channels}


# ---------------------------------------------------------------------------
# Report serialisation helpers
# ---------------------------------------------------------------------------

def _results_to_dataframe(results: List[QueryResult]) -> pd.DataFrame:
    rows = []
    for qr in results:
        for rank, ident in enumerate(qr.ranked_identities, start=1):
            rows.append(
                {
                    "query_image_id": qr.query_image_id,
                    "query_individual_id": qr.query_individual_id,
                    "probe_type": qr.probe_type,
                    "rank": rank,
                    "candidate_individual_id": ident.individual_id,
                    "fused_score": ident.fused_score,
                    "channels_available": ",".join(ident.channels_available),
                    "channels_present_query": ",".join(qr.channels_present),
                    "channels_absent_query": ",".join(qr.channels_absent),
                    "unknown_query": qr.unknown_query,
                    "simulated_unknown": qr.simulated_unknown,
                    **{f"raw_{ch}": ident.channel_raw.get(ch) for ch in ident.channel_raw},
                    **{
                        f"cal_{ch}": ident.channel_calibrated.get(ch)
                        for ch in ident.channel_calibrated
                    },
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reference gallery loading helpers
# ---------------------------------------------------------------------------

def _load_ref_embeddings_for_gallery(
    ref_emb_dir: Path,
    channels: List[str],
    gallery_image_ids: set,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    """
    Load reference embedding files and filter to the given gallery image IDs.

    Returns (descriptor_mappings, embedding_matrices) where each mapping
    DataFrame is already filtered to ``gallery_image_ids``.
    No probe images may enter the returned mappings.
    """
    descriptor_mappings: Dict[str, pd.DataFrame] = {}
    embedding_matrices: Dict[str, np.ndarray] = {}

    for ch in channels:
        mapping_path = ref_emb_dir / f"{ch}_mapping.parquet"
        npy_path = ref_emb_dir / f"{ch}.npy"
        if not mapping_path.is_file() or not npy_path.is_file():
            logger.warning("Reference artifacts missing for channel '%s'; skipping.", ch)
            continue
        dm = pd.read_parquet(str(mapping_path))
        dm["image_id"] = dm["image_id"].astype(str)
        dm = dm[dm["image_id"].isin(gallery_image_ids)].reset_index(drop=True)
        descriptor_mappings[ch] = dm
        embedding_matrices[ch] = np.load(str(npy_path)).astype(np.float32)
        logger.info("Ref channel '%s': %d gallery crops", ch, len(dm))

    return descriptor_mappings, embedding_matrices


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def run_normalized_eval(
    artifact_root: Path,
    splits_path: Path,
    calib_dir: Path,
    out_dir: Path,
    channels: List[str],
    query_partition: str = "query",
    ref_partition: str = "reference",
) -> dict:
    """
    Full normalized evaluation pipeline.

    Split semantics
    ---------------
    ``probe`` (temporal):
        Known-identity probes; reference is the ordinary ``gallery``.
    ``held_out_probe`` (unseen-identity onboarding):
        Known-identity retrieval after onboarding.  Reference is
        ``held_out_gallery ∪ gallery`` (probe images excluded).
        Identity is *known* if it appears in held_out_gallery.
    open-set simulation:
        For each held_out_probe image, truth identity is removed from the
        combined reference; query is re-scored → simulated unknown trial.
        Truth identity is confirmed absent from every candidate set.

    Returns the summary metrics dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. Load splits
    # -----------------------------------------------------------------
    splits_df = pd.read_parquet(str(splits_path))

    def _load_split(split_values: List[str]) -> pd.DataFrame:
        mask = splits_df["split"].isin(split_values)
        df = splits_df[mask][["image_id", "individual_id", "session_id"]].copy()
        df = df.drop_duplicates(subset="image_id").reset_index(drop=True)
        df["image_id"] = df["image_id"].astype(str)
        df["individual_id"] = df["individual_id"].astype(str)
        df["session_id"] = df["session_id"].fillna("unknown").astype(str)
        return df

    temporal_probe_df = _load_split(["probe"])
    onboarding_probe_df = _load_split(["held_out_probe"])
    gallery_df = _load_split(["gallery"])
    held_out_gallery_df = _load_split(["held_out_gallery"])

    logger.info(
        "Splits loaded – temporal_probe: %d, onboarding_probe: %d, "
        "gallery: %d, held_out_gallery: %d",
        len(temporal_probe_df), len(onboarding_probe_df),
        len(gallery_df), len(held_out_gallery_df),
    )

    all_probe_df = pd.concat(
        [temporal_probe_df, onboarding_probe_df], ignore_index=True
    ).drop_duplicates(subset="image_id")
    probe_ids_set = set(all_probe_df["image_id"])

    if all_probe_df.empty:
        raise ValueError(
            "No probe images found (split in ['probe', 'held_out_probe'])."
        )
    if gallery_df.empty:
        raise ValueError("No gallery images found (split == 'gallery').")

    # -----------------------------------------------------------------
    # 2. Build combined gallery for onboarding probes
    #    = held_out_gallery ∪ gallery, with NO probe image IDs included.
    # -----------------------------------------------------------------
    combined_gallery_df = (
        pd.concat([gallery_df, held_out_gallery_df], ignore_index=True)
        .drop_duplicates(subset="image_id")
        .reset_index(drop=True)
    )
    # Strict hygiene: remove any probe images that somehow appear in gallery.
    probe_in_gal = probe_ids_set & set(combined_gallery_df["image_id"])
    if probe_in_gal:
        logger.warning(
            "Removing %d probe image(s) found in gallery/held_out_gallery: %s",
            len(probe_in_gal), sorted(probe_in_gal)[:5],
        )
        combined_gallery_df = combined_gallery_df[
            ~combined_gallery_df["image_id"].isin(probe_in_gal)
        ].reset_index(drop=True)

    gallery_ids_set = set(gallery_df["image_id"])
    combined_gallery_ids_set = set(combined_gallery_df["image_id"])
    held_out_gallery_known_ids = set(held_out_gallery_df["individual_id"])
    gallery_known_ids = set(gallery_df["individual_id"])

    # -----------------------------------------------------------------
    # 3. Load reference embeddings
    # -----------------------------------------------------------------
    ref_emb_dir = artifact_root / EMBEDDINGS_SUBDIR_BTEH / ref_partition

    # Temporal reference: gallery only.
    temporal_ref_dm, temporal_ref_emb = _load_ref_embeddings_for_gallery(
        ref_emb_dir, channels, gallery_ids_set
    )
    active_channels = [ch for ch in channels if ch in temporal_ref_dm]
    if not active_channels:
        raise ValueError(
            "No reference descriptor mappings found for any channel. "
            "Run step_2 first."
        )

    # Combined reference: held_out_gallery ∪ gallery (same embedding files).
    combined_ref_dm, combined_ref_emb = _load_ref_embeddings_for_gallery(
        ref_emb_dir, active_channels, combined_gallery_ids_set
    )

    # -----------------------------------------------------------------
    # 4. Load calibrators and fusion weights
    # -----------------------------------------------------------------
    calibrators = _load_calibrators(calib_dir, active_channels)
    fusion_weights = _load_fusion_weights(calib_dir, active_channels)

    thresh_path = calib_dir / "unknown_threshold.json"
    accept_threshold = 0.5
    if thresh_path.is_file():
        with open(str(thresh_path)) as fh:
            thresh_payload = json.load(fh)
        accept_threshold = float(thresh_payload.get("threshold", 0.5))
        logger.info("Loaded unknown threshold: %.4f", accept_threshold)

    # -----------------------------------------------------------------
    # 5. Calibration flatness diagnostic
    # -----------------------------------------------------------------
    flatness_diag = check_calibration_flatness(calibrators)

    # -----------------------------------------------------------------
    # 6. Build scorers
    #    temporal_scorer  : gallery only
    #    onboarding_scorer: combined gallery (held_out_gallery + gallery)
    # -----------------------------------------------------------------
    temporal_scorer = IdentityLevelScorer(
        gallery_image_df=gallery_df,
        descriptor_mappings=temporal_ref_dm,
        embedding_matrices=temporal_ref_emb,
        calibrators=calibrators,
        weights=fusion_weights,
        all_channels=active_channels,
        accept_threshold=accept_threshold,
    )

    onboarding_scorer: Optional[IdentityLevelScorer] = None
    if not onboarding_probe_df.empty:
        onboarding_scorer = IdentityLevelScorer(
            gallery_image_df=combined_gallery_df,
            descriptor_mappings=combined_ref_dm,
            embedding_matrices=combined_ref_emb,
            calibrators=calibrators,
            weights=fusion_weights,
            all_channels=active_channels,
            accept_threshold=accept_threshold,
        )

    # -----------------------------------------------------------------
    # 7. Load query embeddings
    # -----------------------------------------------------------------
    query_emb_dir = artifact_root / EMBEDDINGS_SUBDIR_BTEH / query_partition
    query_descriptor_mappings, query_embedding_matrices = _load_query_emb_rows(
        query_emb_dir, active_channels, probe_ids_set
    )
    query_image_ch_emb = _build_query_emb_per_image(
        query_descriptor_mappings, query_embedding_matrices, active_channels
    )

    # -----------------------------------------------------------------
    # 8. Run known-retrieval evaluation
    #    temporal : probe → gallery scorer
    #    onboarding: held_out_probe → combined scorer
    # -----------------------------------------------------------------
    temporal_results: List[QueryResult] = []
    for q_img_id in sorted(temporal_probe_df["image_id"]):
        q_indiv = temporal_probe_df.loc[
            temporal_probe_df["image_id"] == q_img_id, "individual_id"
        ].iloc[0]
        q_emb_rows = query_image_ch_emb.get(q_img_id, {})
        result = temporal_scorer.score_query(
            query_image_id=q_img_id,
            query_emb_rows=q_emb_rows,
            query_individual_id=q_indiv,
        )
        result.probe_type = "temporal"
        temporal_results.append(result)

    onboarding_results: List[QueryResult] = []
    if onboarding_scorer is not None:
        for q_img_id in sorted(onboarding_probe_df["image_id"]):
            q_indiv = onboarding_probe_df.loc[
                onboarding_probe_df["image_id"] == q_img_id, "individual_id"
            ].iloc[0]
            q_emb_rows = query_image_ch_emb.get(q_img_id, {})
            result = onboarding_scorer.score_query(
                query_image_id=q_img_id,
                query_emb_rows=q_emb_rows,
                query_individual_id=q_indiv,
            )
            # An onboarding probe is "known" if its identity exists in
            # held_out_gallery (checked by combined scorer gallery_image_df).
            # Scorer sets unknown_query=False when identity is in combined gallery.
            result.probe_type = "unseen_identity_onboarding"
            onboarding_results.append(result)

    all_known_probe_results = temporal_results + onboarding_results
    logger.info(
        "Known retrieval evaluation: %d temporal, %d onboarding.",
        len(temporal_results), len(onboarding_results),
    )

    # -----------------------------------------------------------------
    # 9. Open-set simulation via identity removal
    #    Re-score each held_out_probe against reference with its truth
    #    identity removed.  Truth identity is verified absent from
    #    candidate set in simulate_probe_unknown_trials.
    # -----------------------------------------------------------------
    simulated_unknown_results: List[QueryResult] = []
    if not onboarding_probe_df.empty and not combined_gallery_df.empty:
        try:
            simulated_unknown_results = simulate_probe_unknown_trials(
                probe_df=onboarding_probe_df,
                combined_gallery_df=combined_gallery_df,
                descriptor_mappings=combined_ref_dm,
                embedding_matrices=combined_ref_emb,
                calibrators=calibrators,
                weights=fusion_weights,
                all_channels=active_channels,
                probe_emb_mappings=query_descriptor_mappings,
                probe_emb_matrices=query_embedding_matrices,
            )
        except Exception as exc:
            logger.warning("Open-set simulation failed: %s", exc)
    logger.info(
        "Open-set simulation: %d simulated-unknown trials.", len(simulated_unknown_results)
    )

    # -----------------------------------------------------------------
    # 10. Compute aggregate metrics
    # -----------------------------------------------------------------
    # "known" = not flagged as unknown_query AND not simulated_unknown.
    known_results = [
        r for r in all_known_probe_results
        if not r.unknown_query and not r.simulated_unknown
        and r.query_individual_id is not None
    ]

    n_probe_total = len(all_known_probe_results)
    n_known = len(known_results)
    n_simulated_unknown = len(simulated_unknown_results)

    coverage = (
        sum(1 for r in all_known_probe_results if r.channels_present)
        / max(n_probe_total, 1)
    )

    # Overall known retrieval.
    overall_metrics = _compute_split_metrics(known_results)

    # Per split-type.
    temporal_known = [r for r in temporal_results
                      if not r.unknown_query and not r.simulated_unknown]
    onboarding_known = [r for r in onboarding_results
                        if not r.unknown_query and not r.simulated_unknown]
    temporal_metrics = _compute_split_metrics(temporal_known)
    onboarding_metrics = _compute_split_metrics(onboarding_known)

    # Count checks: n_temporal + n_onboarding + n_unknown_onboarding == n_probe_total.
    n_temporal_known = len(temporal_known)
    n_temporal_not_known = len([r for r in temporal_results if r.unknown_query])
    n_onboarding_known = len(onboarding_known)
    n_onboarding_not_known = len([r for r in onboarding_results if r.unknown_query])
    assert (n_temporal_known + n_temporal_not_known + n_onboarding_known + n_onboarding_not_known
            == n_probe_total), "Protocol count mismatch"

    # Calibration quality on known-retrieval positive pairs.
    cal_probs, cal_labels = [], []
    for qr in known_results:
        for ident in qr.ranked_identities:
            for ch, cal_score in ident.channel_calibrated.items():
                label = 1.0 if ident.individual_id == qr.query_individual_id else 0.0
                cal_probs.append(cal_score)
                cal_labels.append(label)

    cal_probs_arr = np.array(cal_probs, dtype=np.float64)
    cal_labels_arr = np.array(cal_labels, dtype=np.float64)
    ece = _compute_ece(cal_probs_arr, cal_labels_arr)
    brier = _compute_brier(cal_probs_arr, cal_labels_arr)

    # Open-set detection metrics (identity-removed simulation).
    open_set = _open_set_metrics(known_results, simulated_unknown_results, accept_threshold)

    # Channel ablation (known retrieval queries only, calibrated scores).
    channel_ablation = _channel_ablation(known_results, active_channels)

    # Missing-channel breakdown.
    missing_channel_breakdown: Dict[str, dict] = {}
    for ch in active_channels:
        ch_absent = [r for r in known_results if ch in r.channels_absent]
        missing_channel_breakdown[ch] = {
            "n_queries_missing_channel": len(ch_absent),
            "fraction_missing": round(len(ch_absent) / max(n_probe_total, 1), 4),
        }

    summary = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "channels": active_channels,
        "fusion_weights": fusion_weights,
        "accept_threshold": accept_threshold,
        "n_probe_total": n_probe_total,
        "n_known": n_known,
        "n_simulated_unknown": n_simulated_unknown,
        "n_temporal_known": n_temporal_known,
        "n_onboarding_known": n_onboarding_known,
        "coverage": round(coverage, 4),
        # Overall known retrieval (temporal + onboarding combined).
        "known_top1": overall_metrics["top1"],
        "known_top5": overall_metrics["top5"],
        "known_mAP": overall_metrics["mAP"],
        "known_cmc@5": overall_metrics["cmc@5"],
        "known_cmc@10": overall_metrics["cmc@10"],
        # Per-split breakdown.
        "temporal": temporal_metrics,
        "unseen_identity_onboarding": onboarding_metrics,
        # Calibration quality.
        "calibration_ece": round(ece, 4) if not np.isnan(ece) else None,
        "calibration_brier": round(brier, 4) if not np.isnan(brier) else None,
        "calibration_flatness": flatness_diag,
        # Open-set.
        "open_set_detection": open_set,
        # Channel ablations (calibrated scores, known queries).
        "channel_ablation": channel_ablation,
        "missing_channel_breakdown": missing_channel_breakdown,
        # Protocol count sanity.
        "protocol_counts": {
            "n_temporal": len(temporal_results),
            "n_temporal_known": n_temporal_known,
            "n_temporal_not_known": n_temporal_not_known,
            "n_onboarding": len(onboarding_results),
            "n_onboarding_known": n_onboarding_known,
            "n_onboarding_not_known": n_onboarding_not_known,
            "n_simulated_unknown": n_simulated_unknown,
            "sum_known": n_temporal_known + n_onboarding_known,
        },
    }

    # -----------------------------------------------------------------
    # 11. Write outputs
    # -----------------------------------------------------------------
    all_results_for_report = (
        temporal_results + onboarding_results + simulated_unknown_results
    )
    rankings_df = _results_to_dataframe(all_results_for_report)
    rankings_path = out_dir / "normalized_eval_rankings.parquet"
    rankings_df.to_parquet(str(rankings_path), index=False)
    logger.info("Rankings saved to %s (%d rows)", rankings_path, len(rankings_df))

    summary_path = out_dir / "normalized_eval_summary.json"
    with open(str(summary_path), "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Summary saved to %s", summary_path)

    ch_breakdown_path = out_dir / "normalized_eval_channel_breakdown.json"
    with open(str(ch_breakdown_path), "w") as fh:
        json.dump(
            {
                "channel_ablation": channel_ablation,
                "note": "Calibrated-score ablations on known retrieval queries only.",
            },
            fh,
            indent=2,
        )

    unknown_path = out_dir / "normalized_eval_unknown_report.json"
    with open(str(unknown_path), "w") as fh:
        json.dump(
            {
                "n_simulated_unknown_trials": n_simulated_unknown,
                "provenance": "identity_removed_reference_scoring",
                "note": (
                    "Each trial re-scores a held_out_probe image against the combined "
                    "reference gallery (held_out_gallery + gallery) after removing ALL "
                    "crops of the truth identity. Truth identity is verified absent "
                    "from every candidate set."
                ),
                "open_set_metrics": open_set,
                "threshold_source": str(thresh_path),
                "calibration_flatness": flatness_diag,
            },
            fh,
            indent=2,
        )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Step 4c: normalized identity-level fusion evaluation."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ARTIFACT_VERSION_ROOT,
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="Directory with calibrator pkl files (default: <artifact-root>/calibration)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=ACTIVE_DESCRIPTORS,
    )
    parser.add_argument(
        "--query-partition",
        default="query",
        help="Query partition name under embeddings/ (default: query)",
    )
    parser.add_argument(
        "--ref-partition",
        default="reference",
        help="Reference partition name under embeddings/ (default: reference)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    artifact_root = args.artifact_root
    splits_path = args.splits_file or (
        artifact_root / SPLITS_SUBDIR / SPLITS_FILENAME
    )
    calib_dir = args.calibration_dir or (artifact_root / CALIBRATION_SUBDIR_BTEH)
    out_dir = args.out_dir or (artifact_root / REPORTS_SUBDIR)

    if not artifact_root.is_dir():
        logger.error("Artifact root not found: %s", artifact_root)
        sys.exit(1)
    if not splits_path.is_file():
        logger.error("Splits file not found: %s", splits_path)
        sys.exit(1)

    logger.info("Step 4c (normalized eval) starting.")
    summary = run_normalized_eval(
        artifact_root=artifact_root,
        splits_path=splits_path,
        calib_dir=calib_dir,
        out_dir=out_dir,
        channels=args.channels,
        query_partition=args.query_partition,
        ref_partition=args.ref_partition,
    )

    logger.info(
        "Step 4c complete: top1=%.4f top5=%.4f mAP=%.4f coverage=%.4f",
        summary["known_top1"],
        summary["known_top5"],
        summary["known_mAP"],
        summary["coverage"],
    )


if __name__ == "__main__":
    main()
