#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Gallery-only OOF calibration pipeline for the full local-ensemble fusion.

Pipeline stages
---------------
1. Global shortlisting  – Use the selected-v1 IdentityLevelScorer
   (miewid + ear_miewid_projected, Platt-calibrated) in OOF mode (session
   exclusion).  For each pseudo-query, produce a ranked top-K candidate
   shortlist.  Save content-addressed shortlist registration.

2. K selection  – Compute truth-recall at K ∈ {5,10,20,30,50} across all
   OOF pseudo-queries.  Freeze the smallest K achieving ≥ 0.95 OOF recall;
   fall back to K=50 if no K qualifies.  Truth is never forced.

3. Local scoring  – For every (query, candidate) pair in the frozen top-K,
   compute body-local and ear-local identity scores via the canonical
   LocalIdentityScorer.  Persist to atomic, resumable shards.

4. Platt calibration  – Fit separate Platt calibrators on identity-level
   body-local and ear-local scores (positive vs. deterministic negative rows).
   Hard support/fingerprint/flatness guards.  Missing region = unavailable,
   not zero evidence.

5. Fusion weight fitting  – Refit non-negative weights over
   {miewid, ear_miewid_projected, body_local, ear_local} maximising
   identity-macro MRR (primary) and top-1 (secondary tie-break).
   Renormalise over available channels.

6. Artifact save  – Calibrators, weights, OOF metrics, local coverage,
   frozen K, runtime/cache diagnostics, provenance.

Hard guarantees
---------------
* split=='gallery' rows only; hard-fail on probe/held_out_probe contamination.
* selected-v1 production artifacts loaded read-only; never mutated.
* Truth identity NEVER forced into the candidate shortlist.
* Atomic shards: each completed query writes one shard file.
* Parallel workers: disjoint query assignment via SHA256 % worker_count;
  no shared temp-file names; FeatureCache writes are process-safe (uuid4 tmp).

Usage (estimate)
----------------
    python pipeline/local_oof_calibration.py estimate \\
        [--manifest PATH] [--splits PATH] [--crop-manifest PATH]

Usage (run)
-----------
    python pipeline/local_oof_calibration.py run \\
        [--manifest PATH] [--splits PATH] [--crop-manifest PATH] \\
        [--embeddings-dir PATH] [--calibration-dir PATH] \\
        [--output-dir PATH] [--device DEVICE] \\
        [--max-keypoints N] [--cache-dir PATH] \\
        [--override-budget]

Usage (worker)
--------------
    python pipeline/local_oof_calibration.py worker \\
        --worker-index N --worker-count M \\
        [--manifest PATH] [--splits PATH] [--crop-manifest PATH] \\
        [--embeddings-dir PATH] [--output-dir PATH] [--device DEVICE] \\
        [--max-keypoints N] [--cache-dir PATH] [--disable-cudnn] \\
        [--max-sessions N]

    Requires existing frozen shortlist_registration.parquet + config.json +
    fingerprint.json in output-dir (written by ``run``).  Validates source/
    split fingerprints against frozen provenance.  Loads gallery data,
    embeddings, and local scorers exactly like ``run``.  Scores only the
    Stage-4 queries assigned to this worker (deterministic SHA256 partition).
    Does NOT recompute global rankings, shortlist, or final calibration.

Usage (finalize)
----------------
    python pipeline/local_oof_calibration.py finalize \\
        --output-dir PATH [--calibration-dir PATH]

    Requires completed shards for every query in the frozen shortlist.
    Validates every shard fingerprint, merges shards, then fits Platt
    calibrators and fusion weights.  Saves final OOF artifacts.
    Does NOT rescore any queries or modify the frozen shortlist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_bteh import EXPERIMENT_ROOT
from configs.config_elephant import (
    LOCAL_FEATURE_CACHE_MAX_LRU,
    LOCAL_IDENTITY_SCORER_MAX_SESSIONS,
    LOCAL_IDENTITY_SCORER_TOP_K,
    LOCAL_SCORE_SCHEMA_VERSION,
    PRODUCTION_FUSION_WEIGHTS,
    PRODUCTION_SELECTED_CHANNELS,
)
from models.calibration import Calibrator
from models.identity_scorer import LocalIdentityScorer, QueryCrop, ReferenceImage
from models.identity_fusion import (
    IdentityScore,
    QueryResult,
    build_oof_identity_scores,
    _cosine_identity_max,
)
from models.local_matcher import (
    GEOM_HOMOGRAPHY,
    REGION_BODY,
    REGION_EAR,
)
from models.local_score_schema import LocalIdentityScore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real-data loading helpers (CLI wiring; unit-tested independently)
# ---------------------------------------------------------------------------

def _extract_single_fingerprint(df: pd.DataFrame, col: str) -> str:
    """
    Extract a single unique fingerprint value from *col* in *df*.

    Returns "" if the column is absent or all-null.
    Raises FingerprintMismatchError if more than one distinct value is present.
    """
    if col not in df.columns:
        return ""
    fps = df[col].dropna().astype(str).unique()
    if len(fps) == 0:
        return ""
    if len(fps) > 1:
        raise FingerprintMismatchError(
            f"Multiple values in {col!r}: {sorted(fps.tolist())}. "
            "Expected a single consistent fingerprint across the manifest."
        )
    return str(fps[0])


def _load_gallery_data(
    manifest_path: str,
    splits_path: str,
    crop_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """
    Load gallery-only image manifest, crop manifest, and extract fingerprints.

    Uses parquet push-down to read only split=='gallery' rows from splits,
    then filters the manifest and crop manifest to gallery image IDs only.
    Probe rows are never loaded into any structure.

    Returns
    -------
    gallery_df : merged gallery-only image DataFrame
    crop_df : gallery-only crop manifest
    source_fingerprint : single source_fingerprint from crop manifest (or "")
    split_fingerprint  : single split_fingerprint from crop manifest (or "")
    """
    # Push-down filter: only gallery rows loaded from disk.
    splits_df = pd.read_parquet(splits_path, filters=[("split", "==", "gallery")])
    gallery_ids: List[str] = splits_df["image_id"].astype(str).unique().tolist()
    gallery_id_set: set = set(gallery_ids)

    manifest_df = pd.read_parquet(
        manifest_path,
        filters=[("image_id", "in", gallery_ids)],
    )
    crop_df = pd.read_parquet(
        crop_path,
        filters=[("image_id", "in", gallery_ids)],
    )

    # Merge splits (session_id, split) onto manifest rows.
    splits_cols = [c for c in ["image_id", "split", "session_id"] if c in splits_df.columns]
    manifest_df = manifest_df.drop(
        columns=[
            column
            for column in ("split", "session_id")
            if column in manifest_df.columns
        ]
    )
    gallery_df = manifest_df.merge(
        splits_df[splits_cols].drop_duplicates("image_id"),
        on="image_id",
        how="inner",
    )
    # Hard-fail if any probe row slipped through.
    gallery_df = _filter_gallery_only(gallery_df)

    # Safety: remove any crop rows whose image_id is not in the gallery set.
    if "image_id" in crop_df.columns:
        crop_df = crop_df[crop_df["image_id"].astype(str).isin(gallery_id_set)].copy()

    source_fp = _extract_single_fingerprint(crop_df, "source_fingerprint")
    split_fp = _extract_single_fingerprint(crop_df, "split_fingerprint")

    return gallery_df, crop_df, source_fp, split_fp


def _load_embedding_matrices_and_mappings(
    embeddings_dir: Path,
    channels: List[str],
    gallery_ids: Optional[set] = None,
    source_fingerprint: str = "",
    split_fingerprint: str = "",
) -> Tuple[Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
    """
    Load embedding matrices (.npy) and descriptor mapping DataFrames for *channels*.

    If *gallery_ids* is given, the mapping is filtered to those image IDs.

    Validates
    ---------
    * Mapping rows are contiguous (no gaps) in the embedding_row column.
    * Matrix row count covers all mapping rows.
    * source_fingerprint / split_fingerprint consistency when provided.

    Raises FingerprintMismatchError on fingerprint or contiguity violations.
    """
    embeddings_dir = Path(embeddings_dir)
    embedding_matrices: Dict[str, np.ndarray] = {}
    descriptor_mappings: Dict[str, pd.DataFrame] = {}

    for ch in channels:
        npy_path = embeddings_dir / f"{ch}.npy"
        mapping_path = embeddings_dir / f"{ch}_mapping.parquet"
        if not npy_path.exists():
            raise FileNotFoundError(f"Embedding matrix not found: {npy_path}")
        if not mapping_path.exists():
            raise FileNotFoundError(f"Descriptor mapping not found: {mapping_path}")

        mat = np.load(str(npy_path))
        mapping_df = pd.read_parquet(mapping_path).sort_values(
            "embedding_row", kind="mergesort"
        ).reset_index(drop=True)
        full_rows = mapping_df["embedding_row"].to_numpy(dtype=np.int64)
        if not np.array_equal(full_rows, np.arange(len(mapping_df), dtype=np.int64)):
            raise FingerprintMismatchError(
                f"{ch} source mapping embedding_rows are not contiguous/aligned"
            )
        if len(mapping_df) != len(mat):
            raise RuntimeError(
                f"{ch} source mapping/matrix row count mismatch: "
                f"{len(mapping_df)} != {len(mat)}"
            )

        if gallery_ids is not None:
            mapping_df = mapping_df[
                mapping_df["image_id"].astype(str).isin(gallery_ids)
            ].copy()

        mapping_df = mapping_df.sort_values("embedding_row", kind="mergesort").reset_index(drop=True)
        rows = mapping_df["embedding_row"].to_numpy(dtype=np.int64)

        if len(rows) == 0:
            raise RuntimeError(
                f"No gallery rows in {ch} descriptor mapping after filter."
            )

        if rows[-1] >= len(mat):
            raise RuntimeError(
                f"{ch} embedding row index {rows[-1]} is out of range "
                f"(matrix has {len(mat)} rows)."
            )

        if source_fingerprint and "source_fingerprint" in mapping_df.columns:
            fps = set(mapping_df["source_fingerprint"].dropna().astype(str).unique())
            if fps and fps != {source_fingerprint}:
                raise FingerprintMismatchError(
                    f"{ch} mapping source_fingerprint mismatch: "
                    f"{sorted(fps)} vs {source_fingerprint!r}"
                )

        if split_fingerprint and "split_fingerprint" in mapping_df.columns:
            fps = set(mapping_df["split_fingerprint"].dropna().astype(str).unique())
            if fps and fps != {split_fingerprint}:
                raise FingerprintMismatchError(
                    f"{ch} mapping split_fingerprint mismatch: "
                    f"{sorted(fps)} vs {split_fingerprint!r}"
                )

        embedding_matrices[ch] = mat
        descriptor_mappings[ch] = mapping_df
        logger.info(
            "Loaded %s: matrix=%s, gallery rows=%d",
            ch, mat.shape, len(mapping_df),
        )

    return embedding_matrices, descriptor_mappings


def _load_production_calibrators(
    calibration_dir: Path,
    channels: List[str],
) -> Tuple[Dict[str, Calibrator], Dict[str, float]]:
    """
    Load Platt calibrators and fusion weights from *calibration_dir*.

    Expects:
      <calibration_dir>/<channel>.pkl   per channel
      <calibration_dir>/fusion_weights.json

    The JSON may be either ``{channel: float}`` or ``{"weights": {channel: float}}``.
    """
    calibration_dir = Path(calibration_dir)
    calibrators: Dict[str, Calibrator] = {}
    for ch in channels:
        cal_path = calibration_dir / f"{ch}.pkl"
        if not cal_path.is_file():
            raise FileNotFoundError(f"Missing calibrator: {cal_path}")
        calibrators[ch] = Calibrator().load(str(cal_path))
        logger.info("Loaded calibrator for %s.", ch)

    weights_path = calibration_dir / "fusion_weights.json"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing fusion weights: {weights_path}")
    raw = json.loads(weights_path.read_text())
    # Support both {"weights": {...}} and direct {channel: float} layouts.
    if isinstance(raw, dict) and "weights" in raw and isinstance(raw["weights"], dict):
        raw = raw["weights"]
    return calibrators, {k: float(v) for k, v in raw.items()}


def _instantiate_local_scorers(
    device: str,
    disable_cudnn: bool,
    max_keypoints: int,
    max_sessions: int,
    cache_dir: Optional[Path] = None,
) -> Tuple["LocalIdentityScorer", "LocalIdentityScorer"]:
    """
    Create a shared StrictLocalMatcher (LightGlue) and a persistent FeatureCache,
    then build body and ear LocalIdentityScorers.

    Both scorers share the same matcher and cache to avoid double model loading.

    Process-safety of FeatureCache
    --------------------------------
    ``FeatureCache`` writes individual feature bundles atomically using
    ``uuid4``-prefixed temp files followed by ``os.replace``.  Multiple workers
    may safely read the same reference crops from the shared cache concurrently;
    a worker will at worst redundantly recompute and atomically overwrite a
    cache entry, producing the same bytes.  Query assignments are disjoint so
    *query-crop* writes from different workers never target the same cache entry.
    """
    from models.local_matcher import StrictLocalMatcher
    from models.feature_cache import FeatureCache

    matcher = StrictLocalMatcher(
        backend="lightglue",
        max_keypoints=max_keypoints,
        device=device,
        disable_cudnn=disable_cudnn,
    )

    cache: Optional[FeatureCache] = None
    if cache_dir is not None:
        lg_cache_dir = Path(cache_dir) / "lightglue"
        cache = FeatureCache(
            lg_cache_dir,
            matcher.model_fingerprint,
            max_lru_entries=LOCAL_FEATURE_CACHE_MAX_LRU,
        )

    scorer_body = LocalIdentityScorer(
        matcher=matcher,
        cache=cache,
        max_sessions=max_sessions,
        top_k=LOCAL_IDENTITY_SCORER_TOP_K,
        geom_model=GEOM_HOMOGRAPHY,
    )
    scorer_ear = LocalIdentityScorer(
        matcher=matcher,
        cache=cache,
        max_sessions=max_sessions,
        top_k=LOCAL_IDENTITY_SCORER_TOP_K,
        geom_model=GEOM_HOMOGRAPHY,
    )
    return scorer_body, scorer_ear


# ---------------------------------------------------------------------------
# Sub-directory / file name constants
# ---------------------------------------------------------------------------

OOF_CALIBRATION_SUBDIR: str = "oof_calibration"
SHORTLIST_REGISTRATION_PARQUET: str = "shortlist_registration.parquet"
OOF_TABLE_PARQUET: str = "oof_table.parquet"
SHARD_SUBDIR: str = "shards"
CALIBRATORS_SUBDIR: str = "calibrators"
BODY_LOCAL_CALIBRATOR_FILE: str = "body_local_platt.pkl"
EAR_LOCAL_CALIBRATOR_FILE: str = "ear_local_platt.pkl"
WEIGHTS_JSON: str = "fusion_weights.json"
OOF_METRICS_JSON: str = "oof_metrics.json"
OOF_CONFIG_JSON: str = "config.json"
OOF_FINGERPRINT_JSON: str = "fingerprint.json"

# ---------------------------------------------------------------------------
# K-selection grid and threshold
# ---------------------------------------------------------------------------

K_GRID: List[int] = [5, 10, 20, 30, 50]
K_RECALL_THRESHOLD: float = 0.95
K_DEFAULT: int = 50

# ---------------------------------------------------------------------------
# Budget gates
# ---------------------------------------------------------------------------

GATE_MAX_H100_HOURS: float = 24.0
GATE_MAX_CACHE_GB: float = 50.0
_H100_PAIRS_PER_SEC_LIGHTGLUE: float = 15.0
_BYTES_PER_FEATURE_BUNDLE: int = 64 * 1024  # 64 KB

# ---------------------------------------------------------------------------
# Channel names
# ---------------------------------------------------------------------------

CHANNEL_MIEWID: str = "miewid"
CHANNEL_EAR: str = "ear_miewid_projected"
CHANNEL_BODY_LOCAL: str = "body_local"
CHANNEL_EAR_LOCAL: str = "ear_local"

GLOBAL_CHANNELS: List[str] = [CHANNEL_MIEWID, CHANNEL_EAR]
LOCAL_CHANNELS: List[str] = [CHANNEL_BODY_LOCAL, CHANNEL_EAR_LOCAL]
ALL_CHANNELS: List[str] = GLOBAL_CHANNELS + LOCAL_CHANNELS

# Probe splits that must NEVER enter any table.
_PROBE_SPLITS: frozenset = frozenset({"probe", "held_out_probe"})

# Minimum positive / negative support for Platt fitting.
MIN_POSITIVE_SUPPORT_PLATT: int = 5
MIN_NEGATIVE_SUPPORT_PLATT: int = 5

# Flatness guard: std of calibrated output must exceed this.
CALIBRATOR_FLATNESS_THRESHOLD: float = 1e-4


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

class ProbePollutionError(RuntimeError):
    """Hard error: probe / held_out_probe rows entered a gallery-only table."""


class BudgetExceededError(RuntimeError):
    """Hard error: projected GPU time or cache size exceeds configured limits."""


class LocalSupportError(RuntimeError):
    """Hard error: insufficient positive or negative support for Platt calibration."""


class LocalFlatnessError(RuntimeError):
    """Hard error: fitted Platt calibrator output is flat (no discriminative signal)."""


class FingerprintMismatchError(RuntimeError):
    """Hard error: OOF fingerprint mismatch invalidates cached shards."""


class WorkerRangeError(RuntimeError):
    """Hard error: worker_index is out of range [0, worker_count) or worker_count < 1."""


class ShardCoverageError(RuntimeError):
    """Hard error: finalize requires all queries scored but some shards are missing."""


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class OOFPipelineConfig:
    k_grid: List[int] = field(default_factory=lambda: list(K_GRID))
    k_recall_threshold: float = K_RECALL_THRESHOLD
    k_default: int = K_DEFAULT
    max_sessions: int = LOCAL_IDENTITY_SCORER_MAX_SESSIONS
    top_k_aggregation: int = LOCAL_IDENTITY_SCORER_TOP_K
    global_channels: List[str] = field(default_factory=lambda: list(GLOBAL_CHANNELS))
    all_channels: List[str] = field(default_factory=lambda: list(ALL_CHANNELS))
    geom_model_body: str = GEOM_HOMOGRAPHY
    geom_model_ear: str = GEOM_HOMOGRAPHY
    min_positive_support_platt: int = MIN_POSITIVE_SUPPORT_PLATT
    min_negative_support_platt: int = MIN_NEGATIVE_SUPPORT_PLATT
    flatness_threshold: float = CALIBRATOR_FLATNESS_THRESHOLD
    weight_grid_step: float = 0.05
    gate_max_h100_hours: float = GATE_MAX_H100_HOURS
    gate_max_cache_gb: float = GATE_MAX_CACHE_GB
    override_budget: bool = False
    resume: bool = True
    seed: int = 42

    def fingerprint(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Parallel-worker helpers
# ---------------------------------------------------------------------------

def _worker_query_assignment(query_image_id: str, worker_count: int) -> int:
    """
    Deterministic worker assignment: ``int(SHA256(query_image_id), 16) % worker_count``.

    The result is stable (same inputs → same output across processes and
    restarts), produces a near-uniform partition, and requires no inter-process
    coordination.  Call ``_validate_worker_index`` before using the result.
    """
    digest = hashlib.sha256(query_image_id.encode()).hexdigest()
    return int(digest, 16) % worker_count


def _validate_worker_index(worker_index: int, worker_count: int) -> None:
    """
    Raise ``WorkerRangeError`` when *worker_index* / *worker_count* are invalid.

    Conditions that raise:
      * ``worker_count < 1``
      * ``worker_index < 0`` or ``worker_index >= worker_count``
    """
    if worker_count < 1:
        raise WorkerRangeError(
            f"worker_count must be >= 1, got {worker_count}."
        )
    if worker_index < 0 or worker_index >= worker_count:
        raise WorkerRangeError(
            f"worker_index {worker_index} is out of range "
            f"[0, {worker_count}). "
            "Provide a zero-based index strictly smaller than worker_count."
        )


# ---------------------------------------------------------------------------
# Probe guard helpers
# ---------------------------------------------------------------------------

def _assert_no_probe_ids(
    df: pd.DataFrame,
    split_col: str = "split",
    context: str = "table",
) -> None:
    """Hard-fail if any probe or held_out_probe rows exist in *df*."""
    if split_col not in df.columns:
        return
    bad = df[df[split_col].isin(_PROBE_SPLITS)]
    if not bad.empty:
        ids = (
            bad["image_id"].tolist()
            if "image_id" in bad.columns
            else bad.index.tolist()
        )
        raise ProbePollutionError(
            f"PROBE GUARD VIOLATION: {len(bad)} probe/held_out_probe rows "
            f"entered {context}. image_ids: {ids[:10]}"
        )


def _filter_gallery_only(df: pd.DataFrame, split_col: str = "split") -> pd.DataFrame:
    """Return gallery rows.  Mixed probe/gallery input is a hard error."""
    _assert_no_probe_ids(df, split_col, context="gallery_filter input")
    gallery = df[df[split_col] == "gallery"].copy()
    _assert_no_probe_ids(gallery, split_col, context="gallery_filter output")
    return gallery


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------

def _shortlist_fingerprint(query_image_id: str, candidate_ids: List[str]) -> str:
    """Content-addressed fingerprint of a (query, sorted-candidates) shortlist."""
    payload = query_image_id + "|" + ",".join(sorted(candidate_ids))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _config_fingerprint(config: OOFPipelineConfig) -> str:
    return config.fingerprint()


# ---------------------------------------------------------------------------
# K selection
# ---------------------------------------------------------------------------

def compute_recall_at_k(
    oof_results: List[QueryResult],
    k_values: List[int],
) -> Dict[int, float]:
    """
    Compute truth-recall at each K for a list of OOF QueryResults.

    Only includes queries where:
      - query_individual_id is known
      - identity_in_oof_gallery is True (truth existed in the fold rest-gallery)

    Truth is never forced; recall measures how often truth naturally ranks ≤ K.
    """
    results_by_k: Dict[int, List[bool]] = {k: [] for k in k_values}

    for qr in oof_results:
        if not qr.query_individual_id:
            continue
        if not qr.identity_in_oof_gallery:
            continue
        gt_id = qr.query_individual_id
        ranked_ids = [x.individual_id for x in qr.ranked_identities]
        for k in k_values:
            results_by_k[k].append(gt_id in ranked_ids[:k])

    recalls = {}
    for k in k_values:
        hits = results_by_k[k]
        recalls[k] = float(sum(hits) / len(hits)) if hits else 0.0
    return recalls


def select_k_threshold(
    recalls_at_k: Dict[int, float],
    k_grid: List[int],
    threshold: float = K_RECALL_THRESHOLD,
    default_k: int = K_DEFAULT,
) -> int:
    """
    Return the smallest K in *k_grid* with recall ≥ *threshold*.
    Falls back to *default_k* if no K qualifies.
    """
    for k in sorted(k_grid):
        if recalls_at_k.get(k, 0.0) >= threshold:
            return k
    return default_k


# ---------------------------------------------------------------------------
# Global OOF ranking (reuses build_oof_identity_scores from identity_fusion)
# ---------------------------------------------------------------------------

def compute_global_oof_rankings(
    gallery_df: pd.DataFrame,
    descriptor_mappings: Dict[str, pd.DataFrame],
    embedding_matrices: Dict[str, np.ndarray],
    calibrators: Dict[str, Calibrator],
    channels: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> List[QueryResult]:
    """
    Compute session-based OOF global rankings using selected-v1 channels.

    Gallery rows only; probe contamination is a hard error.
    Returns one QueryResult per gallery image that has at least one embedding.
    """
    _assert_no_probe_ids(gallery_df, context="global OOF ranking input")

    results = build_oof_identity_scores(
        gallery_image_df=gallery_df,
        descriptor_mappings={ch: descriptor_mappings[ch] for ch in channels if ch in descriptor_mappings},
        embedding_matrices={ch: embedding_matrices[ch] for ch in channels if ch in embedding_matrices},
        calibrators=calibrators,
        all_channels=channels,
    )
    weights = weights or PRODUCTION_FUSION_WEIGHTS
    for query_result in results:
        for identity_score in query_result.ranked_identities:
            available = [
                channel
                for channel in channels
                if channel in identity_score.channel_calibrated
                and float(weights.get(channel, 0.0)) > 0
            ]
            total_weight = sum(float(weights[channel]) for channel in available)
            identity_score.fused_score = (
                sum(
                    float(weights[channel])
                    * float(identity_score.channel_calibrated[channel])
                    for channel in available
                )
                / total_weight
                if total_weight > 0
                else 0.0
            )
        query_result.ranked_identities.sort(
            key=lambda score: (-score.fused_score, score.individual_id)
        )
    return results


# ---------------------------------------------------------------------------
# Shortlist registration
# ---------------------------------------------------------------------------

def build_shortlist_registration(
    oof_results: List[QueryResult],
    frozen_k: int,
    gallery_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a content-addressed shortlist registration table.

    Columns: fold_session_id, query_image_id, query_session_id,
             query_individual_id, candidate_individual_id,
             candidate_rank, global_fused_score, shortlist_fingerprint, K.

    The fold_session_id is the session that was excluded for each query.
    Truth is never forced; the shortlist is whatever naturally ranked top-K.
    """
    img_to_session = dict(
        zip(
            gallery_df["image_id"].astype(str),
            gallery_df["session_id"].astype(str),
        )
    )
    rows = []
    for qr in oof_results:
        q_id = qr.query_image_id
        q_sess = img_to_session.get(q_id, "")
        q_indiv = qr.query_individual_id or ""
        top_k = qr.ranked_identities[:frozen_k]
        candidate_ids = [x.individual_id for x in top_k]
        fp = _shortlist_fingerprint(q_id, candidate_ids)
        for rank, ident in enumerate(top_k, start=1):
            rows.append({
                "fold_session_id": q_sess,
                "query_image_id": q_id,
                "query_session_id": q_sess,
                "query_individual_id": q_indiv,
                "candidate_individual_id": ident.individual_id,
                "candidate_rank": rank,
                "global_fused_score": float(ident.fused_score),
                "global_miewid_raw": float(
                    ident.channel_raw.get(CHANNEL_MIEWID, float("nan"))
                ),
                "global_miewid_calibrated": float(
                    ident.channel_calibrated.get(CHANNEL_MIEWID, float("nan"))
                ),
                "global_ear_raw": float(
                    ident.channel_raw.get(CHANNEL_EAR, float("nan"))
                ),
                "global_ear_calibrated": float(
                    ident.channel_calibrated.get(CHANNEL_EAR, float("nan"))
                ),
                "shortlist_fingerprint": fp,
                "K": frozen_k,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reference image selection (globally strongest per session)
# ---------------------------------------------------------------------------

def _cosine_to_query(
    query_image_id: str,
    ref_image_id: str,
    emb_matrix: np.ndarray,
    desc_mapping: pd.DataFrame,
) -> float:
    """
    Cosine similarity between a query image and a reference image using the
    provided embedding matrix.  Returns -1.0 if either has no embedding.
    """
    desc_mapping = desc_mapping.copy()
    desc_mapping["image_id"] = desc_mapping["image_id"].astype(str)

    q_rows = desc_mapping.loc[
        desc_mapping["image_id"] == str(query_image_id), "embedding_row"
    ].astype(int).tolist()
    r_rows = desc_mapping.loc[
        desc_mapping["image_id"] == str(ref_image_id), "embedding_row"
    ].astype(int).tolist()

    if not q_rows or not r_rows:
        return -1.0

    q_emb = emb_matrix[q_rows].mean(axis=0)
    q_norm = np.linalg.norm(q_emb)
    if q_norm < 1e-12:
        return -1.0
    q_emb = q_emb / q_norm

    r_emb = emb_matrix[r_rows].mean(axis=0)
    r_norm = np.linalg.norm(r_emb)
    if r_norm < 1e-12:
        return -1.0
    r_emb = r_emb / r_norm

    return float(q_emb @ r_emb)


def select_globally_strongest_refs(
    candidate_individual_id: str,
    fold_session_id: str,
    gallery_df: pd.DataFrame,
    crop_df: pd.DataFrame,
    query_image_id: str,
    emb_matrix_miewid: Optional[np.ndarray],
    desc_mapping_miewid: Optional[pd.DataFrame],
    crop_kind: str,
    max_sessions: int = LOCAL_IDENTITY_SCORER_MAX_SESSIONS,
) -> List[ReferenceImage]:
    """
    Select up to *max_sessions* reference images for *candidate_individual_id*,
    excluding the *fold_session_id*.

    Within each session, picks the image with highest miewid cosine similarity
    to the query (globally strongest reference).  Falls back to first by
    crop_ordinal if embeddings are unavailable.

    One ReferenceImage per session; sessions ordered by descending global
    similarity so LocalIdentityScorer.select_references can apply the cap.
    """
    # Gallery images for this candidate, not in the excluded session.
    cand_imgs = gallery_df[
        (gallery_df["individual_id"].astype(str) == str(candidate_individual_id))
        & (gallery_df["session_id"].astype(str) != str(fold_session_id))
    ].copy()
    if cand_imgs.empty:
        return []

    eligible_ids = set(cand_imgs["image_id"].astype(str))

    # Accepted crops of the right kind.
    accepted = crop_df[
        (crop_df["detector_status"] == "accepted")
        & (crop_df["crop_kind"] == crop_kind)
        & (crop_df["image_id"].astype(str).isin(eligible_ids))
    ].copy()
    if accepted.empty:
        return []

    accepted["image_id"] = accepted["image_id"].astype(str)

    # Merge to get session_id.
    merged = accepted.merge(
        cand_imgs[["image_id", "session_id"]].assign(
            image_id=lambda d: d["image_id"].astype(str)
        ).drop_duplicates("image_id"),
        on="image_id",
        how="left",
    )

    # Compute global similarity per image and pick best per session.
    session_best: Dict[str, Tuple[float, Any]] = {}  # session -> (sim, row)

    for _, row in merged.iterrows():
        img_id = str(row["image_id"])
        sess = str(row["session_id"])

        if emb_matrix_miewid is not None and desc_mapping_miewid is not None:
            sim = _cosine_to_query(
                query_image_id,
                img_id,
                emb_matrix_miewid,
                desc_mapping_miewid,
            )
        else:
            # Fall back: use crop_ordinal as a tie-break (lower is stronger)
            sim = -float(row.get("crop_ordinal", 0))

        if sess not in session_best or sim > session_best[sess][0]:
            session_best[sess] = (sim, row)

    # Sort sessions by descending global similarity.
    sessions_sorted = sorted(
        session_best.items(), key=lambda kv: kv[1][0], reverse=True
    )

    refs: List[ReferenceImage] = []
    for sess, (_, row) in sessions_sorted[:max_sessions]:
        refs.append(
            ReferenceImage(
                crop_id=str(row["crop_id"]),
                crop_path=str(row["crop_path"]),
                crop_kind=crop_kind,
                session_id=sess,
                individual_id=str(candidate_individual_id),
            )
        )
    return refs


# ---------------------------------------------------------------------------
# Query crop builder (reused from pilot pattern)
# ---------------------------------------------------------------------------

def _build_query_crops(
    query_image_id: str,
    crop_df: pd.DataFrame,
    crop_kind: str,
) -> List[QueryCrop]:
    """Return accepted query crops of *crop_kind* for *query_image_id*."""
    accepted = crop_df[
        (crop_df["detector_status"] == "accepted")
        & (crop_df["crop_kind"] == crop_kind)
        & (crop_df["image_id"].astype(str) == str(query_image_id))
    ]
    return [
        QueryCrop(
            crop_id=str(r["crop_id"]),
            crop_path=str(r["crop_path"]),
            crop_kind=crop_kind,
        )
        for _, r in accepted.iterrows()
    ]


# ---------------------------------------------------------------------------
# Local scoring for a single (query, candidate) pair
# ---------------------------------------------------------------------------

def score_local_for_candidate(
    query_image_id: str,
    fold_session_id: str,
    candidate_individual_id: str,
    gallery_df: pd.DataFrame,
    crop_df: pd.DataFrame,
    emb_matrix_miewid: Optional[np.ndarray],
    desc_mapping_miewid: Optional[pd.DataFrame],
    local_scorer_body: Optional[LocalIdentityScorer],
    local_scorer_ear: Optional[LocalIdentityScorer],
    *,
    source_fingerprint: str = "",
    split_fingerprint: str = "",
) -> Dict[str, Any]:
    """
    Score one (query, candidate) pair for body-local and ear-local.

    Returns a dict with keys:
        body_local_score, body_local_n_pairs, body_local_n_valid,
        body_local_n_sessions, body_local_fingerprint, body_local_available,
        ear_local_score, ear_local_n_pairs, ear_local_n_valid,
        ear_local_n_sessions, ear_local_fingerprint, ear_local_available.

    Missing region (no crops) is recorded as available=False; score=NaN.
    """
    result: Dict[str, Any] = {
        "body_local_score": float("nan"),
        "body_local_n_pairs": 0,
        "body_local_n_valid": 0,
        "body_local_n_sessions": 0,
        "body_local_fingerprint": "",
        "body_local_available": False,
        "ear_local_score": float("nan"),
        "ear_local_n_pairs": 0,
        "ear_local_n_valid": 0,
        "ear_local_n_sessions": 0,
        "ear_local_fingerprint": "",
        "ear_local_available": False,
    }

    # --- Body ---
    if local_scorer_body is not None:
        q_crops_body = _build_query_crops(query_image_id, crop_df, REGION_BODY)
        refs_body = select_globally_strongest_refs(
            candidate_individual_id, fold_session_id, gallery_df, crop_df,
            query_image_id, emb_matrix_miewid, desc_mapping_miewid,
            REGION_BODY, local_scorer_body.max_sessions,
        )
        if q_crops_body and refs_body:
            lis: LocalIdentityScore = local_scorer_body.score_identity(
                query_crops=q_crops_body,
                reference_sessions=refs_body,
                candidate_individual_id=candidate_individual_id,
                source_fingerprint=source_fingerprint,
                split_fingerprint=split_fingerprint,
            )
            result["body_local_score"] = float(lis.score)
            result["body_local_n_pairs"] = lis.n_pairs_attempted
            result["body_local_n_valid"] = lis.n_pairs_valid
            result["body_local_n_sessions"] = lis.n_sessions_used
            result["body_local_fingerprint"] = lis.scoring_fingerprint
            result["body_local_available"] = True

    # --- Ear ---
    if local_scorer_ear is not None:
        q_crops_ear = _build_query_crops(query_image_id, crop_df, REGION_EAR)
        refs_ear = select_globally_strongest_refs(
            candidate_individual_id, fold_session_id, gallery_df, crop_df,
            query_image_id, emb_matrix_miewid, desc_mapping_miewid,
            REGION_EAR, local_scorer_ear.max_sessions,
        )
        if q_crops_ear and refs_ear:
            lis_ear: LocalIdentityScore = local_scorer_ear.score_identity(
                query_crops=q_crops_ear,
                reference_sessions=refs_ear,
                candidate_individual_id=candidate_individual_id,
                source_fingerprint=source_fingerprint,
                split_fingerprint=split_fingerprint,
            )
            result["ear_local_score"] = float(lis_ear.score)
            result["ear_local_n_pairs"] = lis_ear.n_pairs_attempted
            result["ear_local_n_valid"] = lis_ear.n_pairs_valid
            result["ear_local_n_sessions"] = lis_ear.n_sessions_used
            result["ear_local_fingerprint"] = lis_ear.scoring_fingerprint
            result["ear_local_available"] = True

    return result


# ---------------------------------------------------------------------------
# Shard management (atomic writes, resume)
# ---------------------------------------------------------------------------

def _shard_path(output_dir: Path, query_image_id: str) -> Path:
    safe_id = str(query_image_id).replace("/", "_").replace("\\", "_")
    return output_dir / SHARD_SUBDIR / f"shard_{safe_id}.parquet"


def _write_shard_atomic(shard_path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    Write rows to *shard_path* atomically via a temporary file.

    Process-safety
    --------------
    The temp file name includes ``uuid4().hex`` so concurrent workers writing
    *different* shards share no deterministic temp-file names and cannot
    collide even when running in the same output directory.  Because query
    assignments are disjoint (each query belongs to exactly one worker),
    the final ``shard_<query_id>.parquet`` paths themselves never overlap
    between workers — no file-level locking is required.
    """
    if not rows:
        return
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = shard_path.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
    df = pd.DataFrame(rows)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, shard_path)


def _load_completed_queries(
    output_dir: Path,
    shortlist_fingerprints: Dict[str, str],
    expected_body_fingerprint: Optional[str] = None,
    expected_ear_fingerprint: Optional[str] = None,
) -> set:
    """Return completed query IDs only when shard fingerprints still match."""
    shard_dir = output_dir / SHARD_SUBDIR
    if not shard_dir.exists():
        return set()
    completed = set()
    for p in shard_dir.glob("shard_*.parquet"):
        try:
            df = pd.read_parquet(p)
            if "query_image_id" in df.columns and not df.empty:
                query_ids = df["query_image_id"].astype(str).unique()
                if len(query_ids) != 1:
                    raise FingerprintMismatchError(
                        f"Shard {p} contains multiple query IDs"
                    )
                query_id = str(query_ids[0])
                actual_shortlist = set(
                    df["shortlist_fingerprint"].dropna().astype(str).unique()
                )
                expected_shortlist = shortlist_fingerprints.get(query_id)
                if actual_shortlist != {expected_shortlist}:
                    raise FingerprintMismatchError(
                        f"Stale shortlist shard for {query_id}: "
                        f"actual={sorted(actual_shortlist)}, expected={expected_shortlist}"
                    )
                for column, expected in (
                    ("body_local_fingerprint", expected_body_fingerprint),
                    ("ear_local_fingerprint", expected_ear_fingerprint),
                ):
                    if not expected or column not in df.columns:
                        continue
                    actual = {
                        value
                        for value in df[column].dropna().astype(str).unique()
                        if value
                    }
                    if actual and actual != {expected}:
                        raise FingerprintMismatchError(
                            f"Stale {column} in shard {p}: "
                            f"actual={sorted(actual)}, expected={expected}"
                        )
                completed.add(query_id)
        except FingerprintMismatchError:
            raise
        except Exception as exc:
            raise FingerprintMismatchError(
                f"Could not validate resume shard {p}: {exc}"
            ) from exc
    return completed


def _merge_shards(output_dir: Path) -> pd.DataFrame:
    """Merge all shard files into a single OOF table DataFrame."""
    shard_dir = output_dir / SHARD_SUBDIR
    if not shard_dir.exists():
        return pd.DataFrame()
    parts = []
    for p in sorted(shard_dir.glob("shard_*.parquet")):
        try:
            parts.append(pd.read_parquet(p))
        except Exception as exc:
            logger.warning("Could not read shard %s: %s", p, exc)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def check_shard_coverage(
    output_dir: Path,
    shortlist_df: pd.DataFrame,
    shortlist_fingerprints: Optional[Dict[str, str]] = None,
    expected_body_fingerprint: Optional[str] = None,
    expected_ear_fingerprint: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """
    Check that every query in *shortlist_df* has a completed, fingerprint-valid shard.

    Parameters
    ----------
    output_dir : directory containing the ``shards/`` subdirectory
    shortlist_df : frozen shortlist registration (needs ``query_image_id`` +
        ``shortlist_fingerprint`` columns)
    shortlist_fingerprints : pre-built ``{query_image_id: fingerprint}`` dict;
        built from *shortlist_df* when None
    expected_body_fingerprint : if provided, validate ``body_local_fingerprint``
        column in every shard
    expected_ear_fingerprint  : if provided, validate ``ear_local_fingerprint``
        column in every shard

    Returns
    -------
    ``(all_covered, missing_query_ids)``
        *all_covered* is True when every query has a valid, fingerprint-matching shard.
        *missing_query_ids* is a sorted list of query IDs that still need scoring.

    Raises
    ------
    FingerprintMismatchError
        Propagated from ``_load_completed_queries`` when an existing shard's
        fingerprint does not match the frozen shortlist fingerprint.
    """
    if shortlist_fingerprints is None:
        shortlist_fingerprints = (
            shortlist_df.groupby("query_image_id")["shortlist_fingerprint"]
            .first()
            .astype(str)
            .to_dict()
        )
    all_queries: set = set(shortlist_df["query_image_id"].astype(str).unique())
    completed = _load_completed_queries(
        output_dir,
        shortlist_fingerprints,
        expected_body_fingerprint,
        expected_ear_fingerprint,
    )
    missing = sorted(all_queries - completed)
    return len(missing) == 0, missing


# ---------------------------------------------------------------------------
# OOF table builder (per-query local scoring with resume)
# ---------------------------------------------------------------------------

def build_oof_table(
    shortlist_df: pd.DataFrame,
    gallery_df: pd.DataFrame,
    crop_df: pd.DataFrame,
    emb_matrix_miewid: Optional[np.ndarray],
    desc_mapping_miewid: Optional[pd.DataFrame],
    local_scorer_body: Optional[LocalIdentityScorer],
    local_scorer_ear: Optional[LocalIdentityScorer],
    output_dir: Path,
    *,
    resume: bool = True,
    source_fingerprint: str = "",
    split_fingerprint: str = "",
    worker_index: Optional[int] = None,
    worker_count: Optional[int] = None,
) -> pd.DataFrame:
    """
    Score each (query, candidate) pair in *shortlist_df* using local scorers,
    writing atomic shard files per query.

    Parallel-worker mode
    --------------------
    When *worker_index* and *worker_count* are both provided, each query is
    deterministically assigned to a worker by
    ``SHA256(query_image_id) % worker_count``.  Only queries assigned to
    *worker_index* are processed in this call; the partition is disjoint so
    shards written by different workers never target the same file.

    When *worker_index* / *worker_count* are None (the default), every query is
    processed — identical to the original single-process behaviour.

    Resume behaviour is independent of worker mode: completed shards (with
    matching fingerprints) are always skipped regardless of which worker wrote
    them.

    Returns the merged OOF table DataFrame (all shards present on disk,
    not just this worker's).
    """
    output_dir = Path(output_dir)

    # Validate worker params when provided.
    if worker_index is not None or worker_count is not None:
        if worker_index is None or worker_count is None:
            raise WorkerRangeError(
                "worker_index and worker_count must both be provided together."
            )
        _validate_worker_index(worker_index, worker_count)

    shortlist_fingerprints = (
        shortlist_df.groupby("query_image_id")["shortlist_fingerprint"]
        .first()
        .astype(str)
        .to_dict()
    )
    completed = (
        _load_completed_queries(
            output_dir,
            shortlist_fingerprints,
            getattr(local_scorer_body, "_identity_scoring_fingerprint", None),
            getattr(local_scorer_ear, "_identity_scoring_fingerprint", None),
        )
        if resume
        else set()
    )
    if completed:
        logger.info("Resume: %d queries already scored.", len(completed))

    unique_queries = (
        shortlist_df[["query_image_id", "query_session_id", "fold_session_id",
                       "query_individual_id"]]
        .drop_duplicates("query_image_id")
        .to_dict("records")
    )

    for q_info in unique_queries:
        q_id = str(q_info["query_image_id"])

        # Skip queries assigned to a different worker.
        if worker_count is not None and _worker_query_assignment(q_id, worker_count) != worker_index:
            continue

        if resume and q_id in completed:
            continue

        fold_sess = str(q_info["fold_session_id"])
        q_sess = str(q_info["query_session_id"])
        q_indiv = str(q_info["query_individual_id"])

        q_rows = shortlist_df[shortlist_df["query_image_id"].astype(str) == q_id]
        shard_rows: List[Dict[str, Any]] = []

        for _, sr in q_rows.iterrows():
            cand_id = str(sr["candidate_individual_id"])
            label = int(cand_id == q_indiv)

            local_result = score_local_for_candidate(
                query_image_id=q_id,
                fold_session_id=fold_sess,
                candidate_individual_id=cand_id,
                gallery_df=gallery_df,
                crop_df=crop_df,
                emb_matrix_miewid=emb_matrix_miewid,
                desc_mapping_miewid=desc_mapping_miewid,
                local_scorer_body=local_scorer_body,
                local_scorer_ear=local_scorer_ear,
                source_fingerprint=source_fingerprint,
                split_fingerprint=split_fingerprint,
            )

            row: Dict[str, Any] = {
                "fold_session_id": fold_sess,
                "query_image_id": q_id,
                "query_session_id": q_sess,
                "query_individual_id": q_indiv,
                "candidate_individual_id": cand_id,
                "global_miewid_raw": float(sr.get("global_miewid_raw", float("nan"))),
                "global_miewid_calibrated": float(sr.get("global_miewid_calibrated", float("nan"))),
                "global_ear_raw": float(sr.get("global_ear_raw", float("nan"))),
                "global_ear_calibrated": float(sr.get("global_ear_calibrated", float("nan"))),
                "global_fused_score": float(sr.get("global_fused_score", float("nan"))),
                "candidate_global_rank": int(sr.get("candidate_rank", 0)),
                "label": label,
                "shortlist_fingerprint": str(sr.get("shortlist_fingerprint", "")),
                "K": int(sr.get("K", 0)),
                **local_result,
            }
            shard_rows.append(row)

        _write_shard_atomic(_shard_path(output_dir, q_id), shard_rows)

    return _merge_shards(output_dir)


# ---------------------------------------------------------------------------
# Platt calibration for local channels
# ---------------------------------------------------------------------------

def fit_local_platt(
    oof_df: pd.DataFrame,
    channel: str,
    *,
    min_positive: int = MIN_POSITIVE_SUPPORT_PLATT,
    min_negative: int = MIN_NEGATIVE_SUPPORT_PLATT,
    flatness_threshold: float = CALIBRATOR_FLATNESS_THRESHOLD,
    expected_fingerprint: str = "",
) -> Calibrator:
    """
    Fit a Platt calibrator on OOF local scores for *channel*.

    Parameters
    ----------
    channel : 'body_local' or 'ear_local'
    min_positive : minimum positive rows required (hard error if fewer)
    min_negative : minimum negative rows required (hard error if fewer)
    flatness_threshold : minimum std of calibrated output (hard error if flat)
    expected_fingerprint : if non-empty, validate scoring_fingerprint column

    Raises
    ------
    LocalSupportError : insufficient support
    LocalFlatnessError : calibrator output is flat
    """
    score_col = f"{channel}_score"
    available_col = f"{channel}_available"

    # Only use rows where the region was available.
    avail_mask = oof_df[available_col].fillna(False) if available_col in oof_df.columns else pd.Series([True] * len(oof_df))
    sub = oof_df[avail_mask].copy()

    if score_col not in sub.columns:
        raise LocalSupportError(
            f"Column {score_col!r} not found in OOF table for channel {channel!r}."
        )

    # Drop rows with NaN scores.
    sub = sub.dropna(subset=[score_col])

    if expected_fingerprint and f"{channel}_fingerprint" in sub.columns:
        fps = sub[f"{channel}_fingerprint"].dropna().unique().tolist()
        if fps and fps != [expected_fingerprint]:
            raise FingerprintMismatchError(
                f"Scoring fingerprint mismatch for {channel!r}: "
                f"expected {expected_fingerprint!r}, found {fps}"
            )

    n_pos = int((sub["label"] == 1).sum())
    n_neg = int((sub["label"] == 0).sum())

    if n_pos < min_positive:
        raise LocalSupportError(
            f"Channel {channel!r}: insufficient positive support "
            f"({n_pos} < {min_positive}). "
            "Cannot fit a Platt calibrator."
        )
    if n_neg < min_negative:
        raise LocalSupportError(
            f"Channel {channel!r}: insufficient negative support "
            f"({n_neg} < {min_negative}). "
            "Cannot fit a Platt calibrator."
        )

    scores = sub[score_col].to_numpy(dtype=np.float64)
    labels = sub["label"].to_numpy(dtype=np.float64)

    cal = Calibrator()
    cal.fit(scores, labels, method="platt")

    # Flatness guard: calibrated output must have meaningful variance.
    cal_out = cal.transform(scores)
    if float(np.std(cal_out)) < flatness_threshold:
        raise LocalFlatnessError(
            f"Channel {channel!r}: Platt calibrator output is flat "
            f"(std={np.std(cal_out):.6f} < {flatness_threshold}). "
            "Scoring signal is insufficient."
        )

    logger.info(
        "Fitted Platt calibrator for %s: n_pos=%d, n_neg=%d, "
        "output_std=%.4f",
        channel, n_pos, n_neg, float(np.std(cal_out)),
    )
    return cal


# ---------------------------------------------------------------------------
# Identity-macro MRR
# ---------------------------------------------------------------------------

def identity_macro_mrr(
    results: List[QueryResult],
    *,
    only_in_gallery: bool = True,
) -> float:
    """
    Compute identity-macro MRR: mean over identities of mean reciprocal rank.

    Each identity contributes one value (mean RR across its queries).
    If *only_in_gallery* is True (default), only include queries where
    identity_in_oof_gallery is True.
    """
    per_identity: Dict[str, List[float]] = {}
    for qr in results:
        gt_id = qr.query_individual_id
        if not gt_id:
            continue
        if only_in_gallery and not qr.identity_in_oof_gallery:
            continue
        ranked = [x.individual_id for x in qr.ranked_identities]
        rr = 0.0
        if gt_id in ranked:
            rr = 1.0 / (ranked.index(gt_id) + 1)
        per_identity.setdefault(gt_id, []).append(rr)

    if not per_identity:
        return 0.0
    return float(np.mean([np.mean(rrs) for rrs in per_identity.values()]))


def identity_macro_top1(
    results: List[QueryResult],
    *,
    only_in_gallery: bool = True,
) -> float:
    """Fraction of identities where mean top-1 accuracy > 0."""
    per_identity: Dict[str, List[float]] = {}
    for qr in results:
        gt_id = qr.query_individual_id
        if not gt_id:
            continue
        if only_in_gallery and not qr.identity_in_oof_gallery:
            continue
        ranked = [x.individual_id for x in qr.ranked_identities]
        hit = float(bool(ranked and ranked[0] == gt_id))
        per_identity.setdefault(gt_id, []).append(hit)
    if not per_identity:
        return 0.0
    return float(np.mean([np.mean(hits) for hits in per_identity.values()]))


# ---------------------------------------------------------------------------
# 4-channel fusion weight fitting (identity-macro MRR, grid search)
# ---------------------------------------------------------------------------

def _apply_4channel_weights_and_rank(
    oof_df: pd.DataFrame,
    calibrators_global: Dict[str, Calibrator],
    calibrators_local: Dict[str, Calibrator],
    weights: Dict[str, float],
    all_channels: List[str],
) -> List[QueryResult]:
    """
    Apply 4-channel calibrated scores and weights, returning QueryResult list.

    Channel availability per row:
      - miewid: global_miewid_calibrated is finite
      - ear_miewid_projected: global_ear_calibrated is finite
      - body_local: body_local_available == True and body_local_score is finite
      - ear_local: ear_local_available == True and ear_local_score is finite

    Missing channels are skipped (renormalised over available channels).
    """
    results = []

    # Pre-calibrate local scores once.
    body_scores_cal: Dict[str, float] = {}
    ear_scores_cal: Dict[str, float] = {}

    for _, row in oof_df.iterrows():
        key = (str(row["query_image_id"]), str(row["candidate_individual_id"]))
        if (
            CHANNEL_BODY_LOCAL in calibrators_local
            and row.get("body_local_available", False)
            and pd.notna(row.get("body_local_score"))
        ):
            s = np.array([float(row["body_local_score"])])
            body_scores_cal[key] = float(calibrators_local[CHANNEL_BODY_LOCAL].transform(s)[0])
        if (
            CHANNEL_EAR_LOCAL in calibrators_local
            and row.get("ear_local_available", False)
            and pd.notna(row.get("ear_local_score"))
        ):
            s = np.array([float(row["ear_local_score"])])
            ear_scores_cal[key] = float(calibrators_local[CHANNEL_EAR_LOCAL].transform(s)[0])

    for q_id, q_df in oof_df.groupby("query_image_id"):
        q_id = str(q_id)
        q_indiv = str(q_df["query_individual_id"].iloc[0])
        fold_sess = str(q_df["fold_session_id"].iloc[0])

        # Check identity_in_oof_gallery: truth is in at least one candidate row.
        cand_ids_set = set(q_df["candidate_individual_id"].astype(str))
        identity_in_oof = q_indiv in cand_ids_set

        identity_scores: List[IdentityScore] = []
        for _, row in q_df.iterrows():
            cand_id = str(row["candidate_individual_id"])
            key = (q_id, cand_id)

            ch_cal: Dict[str, float] = {}
            ch_raw: Dict[str, float] = {}

            if pd.notna(row.get("global_miewid_calibrated")):
                ch_cal[CHANNEL_MIEWID] = float(row["global_miewid_calibrated"])
                ch_raw[CHANNEL_MIEWID] = float(row.get("global_miewid_raw", float("nan")))

            if pd.notna(row.get("global_ear_calibrated")):
                ch_cal[CHANNEL_EAR] = float(row["global_ear_calibrated"])
                ch_raw[CHANNEL_EAR] = float(row.get("global_ear_raw", float("nan")))

            if key in body_scores_cal:
                ch_cal[CHANNEL_BODY_LOCAL] = body_scores_cal[key]
                ch_raw[CHANNEL_BODY_LOCAL] = float(row.get("body_local_score", float("nan")))

            if key in ear_scores_cal:
                ch_cal[CHANNEL_EAR_LOCAL] = ear_scores_cal[key]
                ch_raw[CHANNEL_EAR_LOCAL] = float(row.get("ear_local_score", float("nan")))

            available = [ch for ch in all_channels if ch in ch_cal]
            total_w = sum(weights.get(ch, 0.0) for ch in available)
            if total_w > 0:
                fused = sum(weights.get(ch, 0.0) * ch_cal[ch] / total_w for ch in available)
            else:
                fused = 0.0

            identity_scores.append(
                IdentityScore(
                    individual_id=cand_id,
                    channel_raw=ch_raw,
                    channel_calibrated=ch_cal,
                    channels_available=available,
                    fused_score=float(fused),
                )
            )

        identity_scores.sort(key=lambda x: x.fused_score, reverse=True)
        results.append(
            QueryResult(
                query_image_id=q_id,
                query_individual_id=q_indiv,
                ranked_identities=identity_scores,
                channels_present=list(all_channels),
                channels_absent=[],
                identity_in_oof_gallery=identity_in_oof,
            )
        )
    return results


def _grid_weight_gen(n: int, remaining: float, step: float):
    """Generate all non-negative integer multiples of step summing to remaining."""
    if n == 1:
        yield (round(remaining, 10),)
        return
    k = 0
    while k * step <= remaining + 1e-9:
        w_k = round(k * step, 10)
        for rest in _grid_weight_gen(n - 1, round(remaining - w_k, 10), step):
            yield (w_k,) + rest
        k += 1


def fit_4channel_weights(
    oof_df: pd.DataFrame,
    calibrators_global: Dict[str, Calibrator],
    calibrators_local: Dict[str, Calibrator],
    all_channels: List[str],
    *,
    grid_step: float = 0.05,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Grid search over the weight simplex {w ≥ 0, sum = 1} maximising
    identity-macro MRR (primary) and top-1 (secondary tie-break).

    Weights must be non-negative and sum to 1 over all channels.
    Missing channels per query are renormalised over available channels.

    Returns
    -------
    best_weights : {channel: float}
    diagnostics : dict with search summary
    """
    n_ch = len(all_channels)
    if n_ch == 0:
        raise ValueError("fit_4channel_weights: no channels provided.")
    if oof_df.empty:
        raise ValueError("fit_4channel_weights: empty OOF table.")

    frame = oof_df.reset_index(drop=True)
    channel_values = []
    for channel in all_channels:
        if channel == CHANNEL_MIEWID:
            values = frame["global_miewid_calibrated"].to_numpy(dtype=float)
        elif channel == CHANNEL_EAR:
            values = frame["global_ear_calibrated"].to_numpy(dtype=float)
        elif channel == CHANNEL_BODY_LOCAL:
            raw = frame["body_local_score"].to_numpy(dtype=float)
            available = frame["body_local_available"].fillna(False).to_numpy(dtype=bool)
            values = np.full(len(frame), np.nan, dtype=float)
            mask = available & np.isfinite(raw)
            if mask.any() and channel in calibrators_local:
                values[mask] = calibrators_local[channel].transform(raw[mask])
        elif channel == CHANNEL_EAR_LOCAL:
            raw = frame["ear_local_score"].to_numpy(dtype=float)
            available = frame["ear_local_available"].fillna(False).to_numpy(dtype=bool)
            values = np.full(len(frame), np.nan, dtype=float)
            mask = available & np.isfinite(raw)
            if mask.any() and channel in calibrators_local:
                values[mask] = calibrators_local[channel].transform(raw[mask])
        else:
            raise ValueError(f"Unsupported fusion channel: {channel}")
        channel_values.append(values)

    score_matrix = np.column_stack(channel_values)
    available_matrix = np.isfinite(score_matrix)
    score_matrix = np.nan_to_num(score_matrix, nan=0.0)
    query_codes, query_ids = pd.factorize(
        frame["query_image_id"].astype(str), sort=False
    )
    n_queries = len(query_ids)
    labels = frame["label"].to_numpy(dtype=int) == 1
    candidate_ids = frame["candidate_individual_id"].astype(str).to_numpy()
    query_truth = (
        frame.groupby("query_image_id", sort=False)["query_individual_id"]
        .first()
        .reindex(query_ids)
        .astype(str)
        .to_numpy()
    )
    truth_identity_codes, truth_identities = pd.factorize(query_truth, sort=False)

    def _metrics_for_weights(weight_tuple: tuple[float, ...]) -> tuple[float, float]:
        weight_array = np.asarray(weight_tuple, dtype=float)
        denominators = available_matrix @ weight_array
        numerators = score_matrix @ weight_array
        fused = np.divide(
            numerators,
            denominators,
            out=np.zeros_like(numerators),
            where=denominators > 0,
        )
        truth_scores = np.full(n_queries, np.nan, dtype=float)
        truth_scores[query_codes[labels]] = fused[labels]
        valid_queries = np.isfinite(truth_scores)
        row_truth_scores = truth_scores[query_codes]
        row_truth_ids = query_truth[query_codes]
        outranks_truth = (
            (fused > row_truth_scores)
            | ((fused == row_truth_scores) & (candidate_ids < row_truth_ids))
        ) & np.isfinite(row_truth_scores)
        better_counts = np.bincount(
            query_codes,
            weights=outranks_truth.astype(np.int64),
            minlength=n_queries,
        )
        reciprocal_ranks = np.zeros(n_queries, dtype=float)
        reciprocal_ranks[valid_queries] = 1.0 / (
            better_counts[valid_queries] + 1.0
        )
        top1_hits = valid_queries & (better_counts == 0)

        identity_counts = np.bincount(
            truth_identity_codes[valid_queries],
            minlength=len(truth_identities),
        )
        rr_sums = np.bincount(
            truth_identity_codes[valid_queries],
            weights=reciprocal_ranks[valid_queries],
            minlength=len(truth_identities),
        )
        top1_sums = np.bincount(
            truth_identity_codes[valid_queries],
            weights=top1_hits[valid_queries].astype(float),
            minlength=len(truth_identities),
        )
        supported = identity_counts > 0
        return (
            float(np.mean(rr_sums[supported] / identity_counts[supported])),
            float(np.mean(top1_sums[supported] / identity_counts[supported])),
        )

    best_mrr = -1.0
    best_top1 = -1.0
    best_w = {ch: 1.0 / n_ch for ch in all_channels}
    candidate_count = 0

    for w_tuple in _grid_weight_gen(n_ch, 1.0, grid_step):
        if abs(sum(w_tuple) - 1.0) > 1e-6:
            continue
        weights = {ch: float(w) for ch, w in zip(all_channels, w_tuple)}
        # Non-negative guard.
        if any(v < -1e-9 for v in weights.values()):
            continue

        candidate_count += 1
        mrr, top1 = _metrics_for_weights(w_tuple)

        if mrr > best_mrr or (abs(mrr - best_mrr) < 1e-8 and top1 > best_top1):
            best_mrr = mrr
            best_top1 = top1
            best_w = dict(weights)

    logger.info(
        "fit_4channel_weights: %d candidates; best MRR=%.4f top1=%.4f weights=%s",
        candidate_count, best_mrr, best_top1,
        {ch: round(v, 4) for ch, v in best_w.items()},
    )

    return best_w, {
        "grid_step": grid_step,
        "n_candidates_evaluated": candidate_count,
        "best_mrr": round(best_mrr, 6),
        "best_top1": round(best_top1, 6),
        "best_weights": {ch: round(v, 6) for ch, v in best_w.items()},
    }


# ---------------------------------------------------------------------------
# Budget estimation
# ---------------------------------------------------------------------------

@dataclass
class OOFBudgetEstimate:
    n_gallery_images: int
    n_gallery_identities: int
    n_gallery_sessions: int
    frozen_k: int
    max_sessions: int
    n_pairs_estimated: int
    unique_crops_estimated: int
    estimated_cache_gb: float
    estimated_h100_hours: float
    within_budget: bool
    gate_max_h100_hours: float
    gate_max_cache_gb: float


def estimate_oof_budget(
    gallery_df: pd.DataFrame,
    crop_df: pd.DataFrame,
    frozen_k: int,
    max_sessions: int = LOCAL_IDENTITY_SCORER_MAX_SESSIONS,
    gate_max_h100_hours: float = GATE_MAX_H100_HOURS,
    gate_max_cache_gb: float = GATE_MAX_CACHE_GB,
) -> OOFBudgetEstimate:
    """
    Estimate pair count, cache size, and GPU time for the OOF local scoring run.

    Estimates:
      n_pairs ≈ n_gallery_images × frozen_k × max_sessions × 2 regions
      Each pair uses LightGlue+homography for both body and ear.
    """
    n_images = len(gallery_df)
    n_identities = gallery_df["individual_id"].nunique()
    n_sessions = gallery_df["session_id"].nunique()

    n_pairs = n_images * frozen_k * max_sessions * 2  # 2 regions (body+ear)

    accepted = crop_df[crop_df["detector_status"] == "accepted"]
    unique_crops = len(accepted["crop_id"].unique())
    cache_bytes = unique_crops * _BYTES_PER_FEATURE_BUNDLE * 2  # original + flipped
    cache_gb = cache_bytes / (1024 ** 3)

    gpu_secs = n_pairs / _H100_PAIRS_PER_SEC_LIGHTGLUE
    gpu_hours = gpu_secs / 3600.0

    return OOFBudgetEstimate(
        n_gallery_images=n_images,
        n_gallery_identities=n_identities,
        n_gallery_sessions=n_sessions,
        frozen_k=frozen_k,
        max_sessions=max_sessions,
        n_pairs_estimated=n_pairs,
        unique_crops_estimated=unique_crops,
        estimated_cache_gb=round(cache_gb, 3),
        estimated_h100_hours=round(gpu_hours, 3),
        within_budget=(gpu_hours <= gate_max_h100_hours and cache_gb <= gate_max_cache_gb),
        gate_max_h100_hours=gate_max_h100_hours,
        gate_max_cache_gb=gate_max_cache_gb,
    )


# ---------------------------------------------------------------------------
# Artifact save / load
# ---------------------------------------------------------------------------

def save_oof_artifacts(
    output_dir: Path,
    config: OOFPipelineConfig,
    calibrator_body: Optional[Calibrator],
    calibrator_ear: Optional[Calibrator],
    fusion_weights: Dict[str, float],
    oof_metrics: Dict[str, Any],
    shortlist_df: Optional[pd.DataFrame] = None,
    oof_table_df: Optional[pd.DataFrame] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist all OOF calibration artifacts to *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cal_dir = output_dir / CALIBRATORS_SUBDIR
    cal_dir.mkdir(exist_ok=True)

    if calibrator_body is not None:
        calibrator_body.save(str(cal_dir / BODY_LOCAL_CALIBRATOR_FILE))
    if calibrator_ear is not None:
        calibrator_ear.save(str(cal_dir / EAR_LOCAL_CALIBRATOR_FILE))

    with open(output_dir / WEIGHTS_JSON, "w") as fh:
        json.dump(fusion_weights, fh, indent=2)

    with open(output_dir / OOF_METRICS_JSON, "w") as fh:
        json.dump(oof_metrics, fh, indent=2)

    with open(output_dir / OOF_CONFIG_JSON, "w") as fh:
        json.dump(asdict(config), fh, indent=2)

    fp_payload = {
        "config_fingerprint": config.fingerprint(),
        "schema_version": LOCAL_SCORE_SCHEMA_VERSION,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if provenance:
        fp_payload["provenance"] = provenance
    with open(output_dir / OOF_FINGERPRINT_JSON, "w") as fh:
        json.dump(fp_payload, fh, indent=2)

    if shortlist_df is not None and not shortlist_df.empty:
        shortlist_df.to_parquet(output_dir / SHORTLIST_REGISTRATION_PARQUET, index=False)

    if oof_table_df is not None and not oof_table_df.empty:
        oof_table_df.to_parquet(output_dir / OOF_TABLE_PARQUET, index=False)

    logger.info("OOF artifacts saved to %s", output_dir)


def load_oof_artifacts(
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Load frozen OOF calibration artifacts from *output_dir*.

    Returns dict with keys:
        config, fusion_weights, oof_metrics, fingerprint,
        calibrator_body, calibrator_ear,
        shortlist_df (optional), oof_table_df (optional).
    """
    output_dir = Path(output_dir)

    def _load_json(name: str) -> dict:
        p = output_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Missing OOF artifact: {p}")
        with open(p) as fh:
            return json.load(fh)

    config_dict = _load_json(OOF_CONFIG_JSON)
    fusion_weights = _load_json(WEIGHTS_JSON)
    oof_metrics = _load_json(OOF_METRICS_JSON)
    fingerprint = _load_json(OOF_FINGERPRINT_JSON)

    cal_dir = output_dir / CALIBRATORS_SUBDIR
    cal_body = None
    cal_ear = None
    body_path = cal_dir / BODY_LOCAL_CALIBRATOR_FILE
    ear_path = cal_dir / EAR_LOCAL_CALIBRATOR_FILE
    if body_path.exists():
        cal_body = Calibrator().load(str(body_path))
    if ear_path.exists():
        cal_ear = Calibrator().load(str(ear_path))

    shortlist_df = None
    sl_path = output_dir / SHORTLIST_REGISTRATION_PARQUET
    if sl_path.exists():
        shortlist_df = pd.read_parquet(sl_path)

    oof_table_df = None
    oof_path = output_dir / OOF_TABLE_PARQUET
    if oof_path.exists():
        oof_table_df = pd.read_parquet(oof_path)

    return {
        "config": config_dict,
        "fusion_weights": fusion_weights,
        "oof_metrics": oof_metrics,
        "fingerprint": fingerprint,
        "calibrator_body": cal_body,
        "calibrator_ear": cal_ear,
        "shortlist_df": shortlist_df,
        "oof_table_df": oof_table_df,
    }


# ---------------------------------------------------------------------------
# Finalize: merge shards + calibrate (parallel-worker post-processing)
# ---------------------------------------------------------------------------

def run_finalize_calibration(
    output_dir: Path,
    config: Optional[OOFPipelineConfig] = None,
    calibrators_global: Optional[Dict[str, Calibrator]] = None,
    provenance_override: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Merge all Stage-4 shards and run Platt calibration + fusion weight fitting.

    This is the finalize step for parallel-worker runs.  It:

    * Loads the frozen ``shortlist_registration.parquet`` from *output_dir*.
    * Validates that **every** query in the shortlist has a completed,
      fingerprint-valid shard (raises ``ShardCoverageError`` if any are missing).
    * Merges all shard files into the OOF table.
    * Fits Platt calibrators for body-local and ear-local channels.
    * Fits 4-channel fusion weights via grid search (identity-macro MRR).
    * Saves final OOF calibration artifacts (calibrators, weights, metrics,
      config, fingerprint, merged OOF table).

    Guarantees
    ----------
    * Does NOT rescore any queries (Stage 4 is already done by workers).
    * Does NOT recompute global OOF rankings or shortlist (Stages 1–3).
    * Does NOT overwrite the frozen ``shortlist_registration.parquet``.
    * ``PRODUCTION_FUSION_WEIGHTS`` / ``PRODUCTION_SELECTED_CHANNELS`` are never
      mutated.

    Parameters
    ----------
    output_dir : directory that contains the frozen shortlist, config, and
        shard sub-directory written by ``run`` / ``worker`` commands.
    config : optional override; loaded from *output_dir/config.json* when None.
    calibrators_global : optional global-channel calibrators for weight
        comparison diagnostics; uses ``{}`` when None (no effect on fitting).

    Raises
    ------
    FileNotFoundError : frozen shortlist or config not found in *output_dir*
    ShardCoverageError : not all queries have completed shards
    """
    output_dir = Path(output_dir)

    # Load frozen shortlist (must already exist).
    sl_path = output_dir / SHORTLIST_REGISTRATION_PARQUET
    if not sl_path.exists():
        raise FileNotFoundError(
            f"Frozen shortlist not found: {sl_path}. "
            "Run 'run' or the full worker set before 'finalize'."
        )
    shortlist_df = pd.read_parquet(sl_path)
    existing_fingerprint_path = output_dir / OOF_FINGERPRINT_JSON
    existing_provenance = {}
    if existing_fingerprint_path.exists():
        existing_provenance = json.loads(
            existing_fingerprint_path.read_text()
        ).get("provenance", {})
    if provenance_override:
        existing_provenance.update(
            {
                key: value
                for key, value in provenance_override.items()
                if value
            }
        )

    # Load config from disk when not supplied.
    if config is None:
        config_path = output_dir / OOF_CONFIG_JSON
        if config_path.exists():
            raw = json.loads(config_path.read_text())
            known = set(OOFPipelineConfig.__dataclass_fields__.keys())
            config = OOFPipelineConfig(**{k: v for k, v in raw.items() if k in known})
        else:
            config = OOFPipelineConfig()

    # Validate shard coverage (raises if any query is missing or fingerprint stale).
    shortlist_fingerprints: Dict[str, str] = (
        shortlist_df.groupby("query_image_id")["shortlist_fingerprint"]
        .first()
        .astype(str)
        .to_dict()
    )
    covered, missing = check_shard_coverage(
        output_dir, shortlist_df, shortlist_fingerprints
    )
    if not covered:
        raise ShardCoverageError(
            f"finalize requires all {len(shortlist_fingerprints)} queries to be scored. "
            f"{len(missing)} still missing: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )

    # Merge all shards.
    oof_table_df = _merge_shards(output_dir)
    if oof_table_df.empty:
        raise ShardCoverageError(
            "finalize: merged OOF table is empty after merging all shards."
        )
    for column in ("body_local_fingerprint", "ear_local_fingerprint"):
        if column not in oof_table_df.columns:
            continue
        fingerprints = {
            value
            for value in oof_table_df[column].dropna().astype(str).unique()
            if value
        }
        if len(fingerprints) > 1:
            raise FingerprintMismatchError(
                f"Mixed {column} values across shards: {sorted(fingerprints)}"
            )
    logger.info("finalize: merged OOF table has %d rows.", len(oof_table_df))

    # Platt calibration for local channels.
    calibrator_body: Optional[Calibrator] = None
    calibrator_ear: Optional[Calibrator] = None

    try:
        calibrator_body = fit_local_platt(
            oof_table_df, CHANNEL_BODY_LOCAL,
            min_positive=config.min_positive_support_platt,
            min_negative=config.min_negative_support_platt,
            flatness_threshold=config.flatness_threshold,
        )
    except (LocalSupportError, LocalFlatnessError) as exc:
        logger.warning("finalize: body-local calibration skipped: %s", exc)

    try:
        calibrator_ear = fit_local_platt(
            oof_table_df, CHANNEL_EAR_LOCAL,
            min_positive=config.min_positive_support_platt,
            min_negative=config.min_negative_support_platt,
            flatness_threshold=config.flatness_threshold,
        )
    except (LocalSupportError, LocalFlatnessError) as exc:
        logger.warning("finalize: ear-local calibration skipped: %s", exc)

    calibrators_local: Dict[str, Calibrator] = {}
    if calibrator_body is not None:
        calibrators_local[CHANNEL_BODY_LOCAL] = calibrator_body
    if calibrator_ear is not None:
        calibrators_local[CHANNEL_EAR_LOCAL] = calibrator_ear

    # 4-channel fusion weight fitting.
    fusion_weights: Dict[str, float] = {
        ch: 1.0 / len(config.all_channels) for ch in config.all_channels
    }
    weight_diagnostics: Dict[str, Any] = {}

    if calibrators_local:
        try:
            fusion_weights, weight_diagnostics = fit_4channel_weights(
                oof_df=oof_table_df,
                calibrators_global=calibrators_global or {},
                calibrators_local=calibrators_local,
                all_channels=config.all_channels,
                grid_step=config.weight_grid_step,
            )
        except Exception as exc:
            logger.warning("finalize: weight fitting failed: %s", exc)

    # Coverage stats.
    body_cov = float(
        oof_table_df.get("body_local_available", pd.Series([False])).mean()
    )
    ear_cov = float(
        oof_table_df.get("ear_local_available", pd.Series([False])).mean()
    )
    v1_weights = {ch: PRODUCTION_FUSION_WEIGHTS.get(ch, 0.0) for ch in GLOBAL_CHANNELS}
    frozen_k = (
        int(shortlist_df["K"].iloc[0])
        if "K" in shortlist_df.columns and not shortlist_df.empty
        else 0
    )

    oof_metrics: Dict[str, Any] = {
        "frozen_k": frozen_k,
        "n_shortlist_rows": len(shortlist_df),
        "n_oof_table_rows": len(oof_table_df),
        "body_local_coverage": round(body_cov, 6),
        "ear_local_coverage": round(ear_cov, 6),
        "weight_fitting": weight_diagnostics,
        "weight_comparison": {
            "selected_v1_global_weights": v1_weights,
            "new_4channel_weights": {
                ch: round(v, 6) for ch, v in fusion_weights.items()
            },
        },
        "pipeline_version": LOCAL_SCORE_SCHEMA_VERSION,
        "finalized": True,
    }

    # Save artifacts.  Pass shortlist_df=None so the frozen shortlist is NOT overwritten.
    save_oof_artifacts(
        output_dir=output_dir,
        config=config,
        calibrator_body=calibrator_body,
        calibrator_ear=calibrator_ear,
        fusion_weights=fusion_weights,
        oof_metrics=oof_metrics,
        shortlist_df=None,
        oof_table_df=oof_table_df,
        provenance=existing_provenance,
    )
    logger.info("finalize: OOF calibration artifacts saved to %s", output_dir)

    return {
        "config": asdict(config),
        "fusion_weights": fusion_weights,
        "oof_metrics": oof_metrics,
        "calibrator_body": calibrator_body,
        "calibrator_ear": calibrator_ear,
        "oof_table_df": oof_table_df,
        "weight_diagnostics": weight_diagnostics,
    }


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------

def run_oof_calibration(
    config: OOFPipelineConfig,
    gallery_df: pd.DataFrame,
    crop_df: pd.DataFrame,
    descriptor_mappings: Dict[str, pd.DataFrame],
    embedding_matrices: Dict[str, np.ndarray],
    calibrators_global: Dict[str, Calibrator],
    local_scorer_body: Optional[LocalIdentityScorer],
    local_scorer_ear: Optional[LocalIdentityScorer],
    output_dir: Path,
    *,
    fusion_weights_global: Optional[Dict[str, float]] = None,
    override_budget: bool = False,
    source_fingerprint: str = "",
    split_fingerprint: str = "",
) -> Dict[str, Any]:
    """
    Run the full gallery-only OOF calibration pipeline.

    Parameters
    ----------
    config : OOFPipelineConfig
    gallery_df : gallery-only image DataFrame (columns: image_id, individual_id, session_id)
    crop_df : crop manifest DataFrame (columns: crop_id, image_id, crop_kind, crop_path, ...)
    descriptor_mappings : {channel: DataFrame} for global channels
    embedding_matrices : {channel: np.ndarray} L2-normalised
    calibrators_global : {channel: Calibrator} for global channels (selected-v1)
    local_scorer_body : canonical LocalIdentityScorer for body (LightGlue+homography)
    local_scorer_ear : canonical LocalIdentityScorer for ear (LightGlue+homography)
    output_dir : path to write OOF artifacts
    fusion_weights_global : selected-v1 global fusion weights; falls back to
        PRODUCTION_FUSION_WEIGHTS if None (never mutates the production constant).
    override_budget : skip budget gate if True

    Returns
    -------
    Artifact dict (same as load_oof_artifacts would return).
    """
    t_start = time.perf_counter()
    _assert_no_probe_ids(gallery_df, context="run_oof_calibration gallery_df")
    output_dir = Path(output_dir)

    # Stage 1: Global OOF rankings -------------------------------------------
    logger.info("Stage 1: global OOF rankings (%d gallery images).", len(gallery_df))
    oof_results = compute_global_oof_rankings(
        gallery_df=gallery_df,
        descriptor_mappings=descriptor_mappings,
        embedding_matrices=embedding_matrices,
        calibrators=calibrators_global,
        channels=config.global_channels,
        weights=fusion_weights_global,  # supplied weights; None → PRODUCTION_FUSION_WEIGHTS
    )
    logger.info("Global OOF: %d QueryResult objects.", len(oof_results))

    # Stage 2: K selection -----------------------------------------------------
    recalls_at_k = compute_recall_at_k(oof_results, config.k_grid)
    frozen_k = select_k_threshold(
        recalls_at_k, config.k_grid, config.k_recall_threshold, config.k_default
    )
    logger.info(
        "K selection: recalls=%s → frozen_k=%d",
        {k: round(v, 4) for k, v in recalls_at_k.items()}, frozen_k,
    )

    # Budget estimation and gate -----------------------------------------------
    budget = estimate_oof_budget(
        gallery_df=gallery_df,
        crop_df=crop_df,
        frozen_k=frozen_k,
        max_sessions=config.max_sessions,
        gate_max_h100_hours=config.gate_max_h100_hours,
        gate_max_cache_gb=config.gate_max_cache_gb,
    )
    logger.info(
        "Budget: %d pairs, %.2f H100-hours, %.3f GB cache; within_budget=%s",
        budget.n_pairs_estimated, budget.estimated_h100_hours,
        budget.estimated_cache_gb, budget.within_budget,
    )
    if not budget.within_budget and not override_budget and not config.override_budget:
        raise BudgetExceededError(
            f"Projected budget exceeds limits: "
            f"{budget.estimated_h100_hours:.2f} H100-hours "
            f"(limit {config.gate_max_h100_hours}) or "
            f"{budget.estimated_cache_gb:.3f} GB cache "
            f"(limit {config.gate_max_cache_gb}). "
            "Pass override_budget=True to proceed anyway."
        )

    # Stage 3: Shortlist registration ------------------------------------------
    logger.info("Stage 3: building shortlist registration (K=%d).", frozen_k)
    shortlist_df = build_shortlist_registration(oof_results, frozen_k, gallery_df)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_sl = output_dir / (SHORTLIST_REGISTRATION_PARQUET + ".tmp")
    shortlist_df.to_parquet(tmp_sl, index=False)
    os.replace(tmp_sl, output_dir / SHORTLIST_REGISTRATION_PARQUET)
    config_tmp = output_dir / f".{uuid.uuid4().hex}.config.tmp"
    config_tmp.write_text(json.dumps(asdict(config), indent=2))
    os.replace(config_tmp, output_dir / OOF_CONFIG_JSON)
    fingerprint_payload = {
        "config_fingerprint": config.fingerprint(),
        "schema_version": LOCAL_SCORE_SCHEMA_VERSION,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "pre_scoring",
        "provenance": {
            "source_fingerprint": source_fingerprint,
            "split_fingerprint": split_fingerprint,
        },
    }
    fingerprint_tmp = output_dir / f".{uuid.uuid4().hex}.fingerprint.tmp"
    fingerprint_tmp.write_text(json.dumps(fingerprint_payload, indent=2))
    os.replace(fingerprint_tmp, output_dir / OOF_FINGERPRINT_JSON)
    logger.info(
        "Shortlist registration: %d rows (%d unique queries).",
        len(shortlist_df),
        shortlist_df["query_image_id"].nunique() if not shortlist_df.empty else 0,
    )

    # Stage 4: Local scoring --------------------------------------------------
    logger.info("Stage 4: local scoring.")
    emb_miewid = embedding_matrices.get(CHANNEL_MIEWID)
    dm_miewid = descriptor_mappings.get(CHANNEL_MIEWID)

    oof_table_df = build_oof_table(
        shortlist_df=shortlist_df,
        gallery_df=gallery_df,
        crop_df=crop_df,
        emb_matrix_miewid=emb_miewid,
        desc_mapping_miewid=dm_miewid,
        local_scorer_body=local_scorer_body,
        local_scorer_ear=local_scorer_ear,
        output_dir=output_dir,
        resume=config.resume,
        source_fingerprint=source_fingerprint,
        split_fingerprint=split_fingerprint,
    )
    logger.info("OOF table: %d rows.", len(oof_table_df))

    # Stage 5: Local Platt calibration ----------------------------------------
    logger.info("Stage 5: fitting local Platt calibrators.")
    calibrator_body: Optional[Calibrator] = None
    calibrator_ear: Optional[Calibrator] = None

    if not oof_table_df.empty:
        try:
            calibrator_body = fit_local_platt(
                oof_table_df, CHANNEL_BODY_LOCAL,
                min_positive=config.min_positive_support_platt,
                min_negative=config.min_negative_support_platt,
                flatness_threshold=config.flatness_threshold,
            )
        except (LocalSupportError, LocalFlatnessError) as exc:
            logger.warning("Body-local calibration skipped: %s", exc)

        try:
            calibrator_ear = fit_local_platt(
                oof_table_df, CHANNEL_EAR_LOCAL,
                min_positive=config.min_positive_support_platt,
                min_negative=config.min_negative_support_platt,
                flatness_threshold=config.flatness_threshold,
            )
        except (LocalSupportError, LocalFlatnessError) as exc:
            logger.warning("Ear-local calibration skipped: %s", exc)

    calibrators_local: Dict[str, Calibrator] = {}
    if calibrator_body is not None:
        calibrators_local[CHANNEL_BODY_LOCAL] = calibrator_body
    if calibrator_ear is not None:
        calibrators_local[CHANNEL_EAR_LOCAL] = calibrator_ear

    # Stage 6: 4-channel fusion weight fitting --------------------------------
    logger.info("Stage 6: fitting 4-channel fusion weights.")
    fusion_weights: Dict[str, float] = {ch: 1.0 / len(config.all_channels) for ch in config.all_channels}
    weight_diagnostics: Dict[str, Any] = {}

    if not oof_table_df.empty and calibrators_local:
        try:
            fusion_weights, weight_diagnostics = fit_4channel_weights(
                oof_df=oof_table_df,
                calibrators_global=calibrators_global,
                calibrators_local=calibrators_local,
                all_channels=config.all_channels,
                grid_step=config.weight_grid_step,
            )
        except Exception as exc:
            logger.warning("Weight fitting failed: %s", exc)

    t_elapsed = time.perf_counter() - t_start

    # Compute coverage stats ---------------------------------------------------
    if not oof_table_df.empty:
        body_cov = float(oof_table_df.get("body_local_available", pd.Series([False])).mean())
        ear_cov = float(oof_table_df.get("ear_local_available", pd.Series([False])).mean())
    else:
        body_cov = ear_cov = 0.0

    # Compare selected-v1 weights (global only) with new 4-channel weights.
    v1_weights = {ch: PRODUCTION_FUSION_WEIGHTS.get(ch, 0.0) for ch in GLOBAL_CHANNELS}
    weight_comparison = {
        "selected_v1_global_weights": v1_weights,
        "new_4channel_weights": {ch: round(v, 6) for ch, v in fusion_weights.items()},
    }

    oof_metrics = {
        "frozen_k": frozen_k,
        "recalls_at_k": {str(k): round(v, 6) for k, v in recalls_at_k.items()},
        "n_oof_results": len(oof_results),
        "n_shortlist_rows": len(shortlist_df),
        "n_oof_table_rows": len(oof_table_df),
        "body_local_coverage": round(body_cov, 6),
        "ear_local_coverage": round(ear_cov, 6),
        "weight_fitting": weight_diagnostics,
        "weight_comparison": weight_comparison,
        "budget": asdict(budget),
        "runtime_seconds": round(t_elapsed, 2),
        "pipeline_version": LOCAL_SCORE_SCHEMA_VERSION,
    }

    # Stage 7: Save all artifacts --------------------------------------------
    logger.info("Stage 7: saving OOF artifacts.")
    save_oof_artifacts(
        output_dir=output_dir,
        config=config,
        calibrator_body=calibrator_body,
        calibrator_ear=calibrator_ear,
        fusion_weights=fusion_weights,
        oof_metrics=oof_metrics,
        shortlist_df=shortlist_df,
        oof_table_df=oof_table_df,
        provenance={
            "source_fingerprint": source_fingerprint,
            "split_fingerprint": split_fingerprint,
            "schema_version": LOCAL_SCORE_SCHEMA_VERSION,
        },
    )

    return {
        "config": asdict(config),
        "fusion_weights": fusion_weights,
        "oof_metrics": oof_metrics,
        "calibrator_body": calibrator_body,
        "calibrator_ear": calibrator_ear,
        "shortlist_df": shortlist_df,
        "oof_table_df": oof_table_df,
        "frozen_k": frozen_k,
        "recalls_at_k": recalls_at_k,
        "weight_diagnostics": weight_diagnostics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Gallery-only OOF calibration pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ---- estimate ------------------------------------------------------------
    est = sub.add_parser("estimate", help="Estimate budget without running scoring.")
    est.add_argument("--manifest", default=None)
    est.add_argument("--splits", default=None)
    est.add_argument("--crop-manifest", default=None)
    est.add_argument("--k", type=int, default=K_DEFAULT)
    est.add_argument("--max-sessions", type=int, default=LOCAL_IDENTITY_SCORER_MAX_SESSIONS)

    # ---- run -----------------------------------------------------------------
    run = sub.add_parser("run", help="Run OOF calibration.")
    run.add_argument("--manifest", default=None)
    run.add_argument("--splits", default=None)
    run.add_argument("--crop-manifest", default=None)
    run.add_argument("--embeddings-dir", default=None)
    run.add_argument("--calibration-dir", default=None)
    run.add_argument("--output-dir", default=None)
    run.add_argument("--device", default="cuda")
    run.add_argument("--max-keypoints", type=int, default=2048)
    run.add_argument("--cache-dir", default=None)
    run.add_argument("--override-budget", action="store_true")
    run.add_argument("--no-resume", action="store_true")
    run.add_argument("--disable-cudnn", action="store_true",
                     help="Disable cuDNN and cuDNN SDP (required on some H100 configurations).")
    run.add_argument("--max-sessions", type=int, default=LOCAL_IDENTITY_SCORER_MAX_SESSIONS,
                     help="Maximum reference sessions per candidate for local scoring.")

    # ---- worker --------------------------------------------------------------
    wkr = sub.add_parser(
        "worker",
        help="Score Stage-4 queries assigned to this worker (parallel mode).",
    )
    wkr.add_argument(
        "--worker-index", type=int, required=True,
        help="Zero-based worker index in [0, worker-count).",
    )
    wkr.add_argument(
        "--worker-count", type=int, required=True,
        help="Total number of parallel workers.",
    )
    wkr.add_argument("--manifest", default=None)
    wkr.add_argument("--splits", default=None)
    wkr.add_argument("--crop-manifest", default=None)
    wkr.add_argument("--embeddings-dir", default=None)
    wkr.add_argument("--output-dir", default=None,
                     help="Output dir that already contains the frozen shortlist/config.")
    wkr.add_argument("--device", default="cuda")
    wkr.add_argument("--max-keypoints", type=int, default=2048)
    wkr.add_argument("--cache-dir", default=None)
    wkr.add_argument("--disable-cudnn", action="store_true",
                     help="Disable cuDNN (required on some H100 configurations).")
    wkr.add_argument("--max-sessions", type=int, default=LOCAL_IDENTITY_SCORER_MAX_SESSIONS,
                     help="Maximum reference sessions per candidate for local scoring.")

    # ---- finalize ------------------------------------------------------------
    fin = sub.add_parser(
        "finalize",
        help="Merge all shards and fit calibrators/weights (after all workers done).",
    )
    fin.add_argument("--output-dir", default=None,
                     help="Output dir containing frozen shortlist and completed shards.")
    fin.add_argument("--calibration-dir", default=None,
                     help="Optional: production calibrator dir for weight comparison.")
    fin.add_argument("--source-fingerprint", default=None)
    fin.add_argument("--split-fingerprint", default=None)

    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ---- finalize: dispatched early – no manifest/splits/gallery needed -----
    if args.command == "finalize":
        from configs.config_elephant import PRODUCTION_SELECTED_CHANNELS as _PSC_FIN

        _output_dir_fin = (
            Path(args.output_dir) if args.output_dir
            else (EXPERIMENT_ROOT / OOF_CALIBRATION_SUBDIR)
        )
        _cal_global_fin: Dict[str, Calibrator] = {}
        if getattr(args, "calibration_dir", None):
            try:
                _cal_global_fin, _ = _load_production_calibrators(
                    calibration_dir=Path(args.calibration_dir),
                    channels=list(_PSC_FIN),
                )
            except FileNotFoundError as exc:
                logger.warning(
                    "finalize: could not load production calibrators: %s", exc
                )
        result_fin = run_finalize_calibration(
            output_dir=_output_dir_fin,
            calibrators_global=_cal_global_fin or None,
            provenance_override={
                "source_fingerprint": args.source_fingerprint,
                "split_fingerprint": args.split_fingerprint,
                "schema_version": LOCAL_SCORE_SCHEMA_VERSION,
            },
        )
        logger.info("finalize complete.  Output: %s", _output_dir_fin)
        logger.info("fusion_weights: %s", result_fin.get("fusion_weights"))
        return 0

    # ---- shared path resolution (estimate + run + worker) -------------------
    from configs.config_bteh import (
        ARTIFACT_VERSION_ROOT,
        MANIFEST_SUBDIR,
        MANIFEST_FILENAME,
        SPLITS_SUBDIR,
        SPLITS_FILENAME,
    )

    manifest_path = args.manifest or str(
        ARTIFACT_VERSION_ROOT / MANIFEST_SUBDIR / MANIFEST_FILENAME
    )
    splits_path = args.splits or str(
        ARTIFACT_VERSION_ROOT / SPLITS_SUBDIR / SPLITS_FILENAME
    )

    # ---- estimate ------------------------------------------------------------
    if args.command == "estimate":
        from configs.config_bteh import CROPS_SUBDIR
        crop_path = args.crop_manifest or str(
            ARTIFACT_VERSION_ROOT / CROPS_SUBDIR / "crop_manifest.parquet"
        )
        gallery_df, crop_df, _, _ = _load_gallery_data(
            Path(manifest_path),
            Path(splits_path),
            Path(crop_path),
        )

        budget = estimate_oof_budget(
            gallery_df=gallery_df,
            crop_df=crop_df,
            frozen_k=args.k,
            max_sessions=args.max_sessions,
        )
        print(json.dumps(asdict(budget), indent=2))
        return 0

    # ------------------------------------------------------------------
    # run / worker: both need gallery data + embeddings
    # ------------------------------------------------------------------
    from configs.config_bteh import (
        ARTIFACT_VERSION_ROOT as _AV_ROOT,
        MANIFEST_SUBDIR,
        MANIFEST_FILENAME,
        SPLITS_SUBDIR,
        SPLITS_FILENAME,
        CROPS_SUBDIR,
        EMBEDDINGS_SUBDIR_BTEH,
    )
    from configs.config_elephant import (
        PRODUCTION_SELECTED_CHANNELS,
        PRODUCTION_CALIBRATION_SUBDIR,
    )

    _manifest_path = args.manifest or str(
        _AV_ROOT / MANIFEST_SUBDIR / MANIFEST_FILENAME
    )
    _splits_path = args.splits or str(
        _AV_ROOT / SPLITS_SUBDIR / SPLITS_FILENAME
    )
    _crop_path = args.crop_manifest or str(
        _AV_ROOT / CROPS_SUBDIR / "crop_manifest.parquet"
    )
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else (EXPERIMENT_ROOT / OOF_CALIBRATION_SUBDIR)
    )
    embeddings_dir = (
        Path(args.embeddings_dir) if args.embeddings_dir
        else (_AV_ROOT / EMBEDDINGS_SUBDIR_BTEH)
    )
    cache_dir = (
        Path(args.cache_dir) if args.cache_dir
        else (output_dir / "feature_cache")
    )

    # 1. Gallery push-down: probe rows are never loaded.
    logger.info("Loading gallery data (split==gallery push-down).")
    gallery_df, crop_df, source_fp, split_fp = _load_gallery_data(
        manifest_path=_manifest_path,
        splits_path=_splits_path,
        crop_path=_crop_path,
    )
    gallery_id_set = set(gallery_df["image_id"].astype(str))
    logger.info(
        "Gallery: %d images, %d individuals, %d sessions.",
        len(gallery_df),
        gallery_df["individual_id"].nunique(),
        gallery_df["session_id"].nunique() if "session_id" in gallery_df.columns else 0,
    )

    # 2. Load selected-v1 reference descriptor matrices + mapping parquets.
    logger.info("Loading embedding matrices and descriptor mappings.")
    embedding_matrices, descriptor_mappings = _load_embedding_matrices_and_mappings(
        embeddings_dir=embeddings_dir,
        channels=list(PRODUCTION_SELECTED_CHANNELS),
        gallery_ids=gallery_id_set,
        source_fingerprint=source_fp,
        split_fingerprint=split_fp,
    )

    # ---- worker command (short-circuits before calibrator loading) -----------
    if args.command == "worker":
        # Fast-fail on invalid index before loading scorers.
        _validate_worker_index(args.worker_index, args.worker_count)

        # Frozen shortlist + config must already exist from a prior 'run'.
        sl_path = output_dir / SHORTLIST_REGISTRATION_PARQUET
        fp_path = output_dir / OOF_FINGERPRINT_JSON
        cfg_path = output_dir / OOF_CONFIG_JSON
        for required in (sl_path, fp_path, cfg_path):
            if not required.exists():
                raise FileNotFoundError(
                    f"Required frozen artifact missing: {required}. "
                    "Run 'run' to build the shortlist before launching workers."
                )

        shortlist_df_w = pd.read_parquet(sl_path)
        logger.info(
            "Loaded frozen shortlist: %d rows, %d queries.",
            len(shortlist_df_w),
            shortlist_df_w["query_image_id"].nunique() if not shortlist_df_w.empty else 0,
        )

        # Validate source/split fingerprints against frozen provenance.
        frozen_fp_data = json.loads(fp_path.read_text())
        provenance = frozen_fp_data.get("provenance", {})
        if provenance.get("source_fingerprint") and source_fp:
            if provenance["source_fingerprint"] != source_fp:
                raise FingerprintMismatchError(
                    f"Worker source_fingerprint {source_fp!r} does not match "
                    f"frozen provenance {provenance['source_fingerprint']!r}."
                )
        if provenance.get("split_fingerprint") and split_fp:
            if provenance["split_fingerprint"] != split_fp:
                raise FingerprintMismatchError(
                    f"Worker split_fingerprint {split_fp!r} does not match "
                    f"frozen provenance {provenance['split_fingerprint']!r}."
                )

        # Instantiate local scorers (identical to 'run' flow).
        logger.info(
            "Instantiating local scorers (device=%s, disable_cudnn=%s, "
            "max_keypoints=%d, worker %d/%d).",
            args.device, args.disable_cudnn, args.max_keypoints,
            args.worker_index, args.worker_count,
        )
        scorer_body_w, scorer_ear_w = _instantiate_local_scorers(
            device=args.device,
            disable_cudnn=args.disable_cudnn,
            max_keypoints=args.max_keypoints,
            max_sessions=args.max_sessions,
            cache_dir=cache_dir,
        )

        # Score only assigned queries; do NOT fit calibrators or save final artifacts.
        logger.info(
            "Worker %d/%d: scoring assigned Stage-4 queries.",
            args.worker_index, args.worker_count,
        )
        build_oof_table(
            shortlist_df=shortlist_df_w,
            gallery_df=gallery_df,
            crop_df=crop_df,
            emb_matrix_miewid=embedding_matrices.get(CHANNEL_MIEWID),
            desc_mapping_miewid=descriptor_mappings.get(CHANNEL_MIEWID),
            local_scorer_body=scorer_body_w,
            local_scorer_ear=scorer_ear_w,
            output_dir=output_dir,
            resume=True,
            source_fingerprint=source_fp,
            split_fingerprint=split_fp,
            worker_index=args.worker_index,
            worker_count=args.worker_count,
        )
        logger.info(
            "Worker %d/%d complete.  Run 'finalize' when all workers are done.",
            args.worker_index, args.worker_count,
        )
        return 0

    # ------------------------------------------------------------------
    # run: end-to-end gallery-only OOF calibration
    # ------------------------------------------------------------------
    calibration_dir = (
        Path(args.calibration_dir) if args.calibration_dir
        else (_AV_ROOT / PRODUCTION_CALIBRATION_SUBDIR)
    )

    # 3. Load Platt calibrators + fusion weights from calibration_projected.
    logger.info("Loading production calibrators and fusion weights from %s.", calibration_dir)
    calibrators_global, fusion_weights_global = _load_production_calibrators(
        calibration_dir=calibration_dir,
        channels=list(PRODUCTION_SELECTED_CHANNELS),
    )

    # 4. Instantiate StrictLocalMatcher LightGlue scorers.
    logger.info(
        "Instantiating local scorers (device=%s, disable_cudnn=%s, max_keypoints=%d).",
        args.device, args.disable_cudnn, args.max_keypoints,
    )
    scorer_body, scorer_ear = _instantiate_local_scorers(
        device=args.device,
        disable_cudnn=args.disable_cudnn,
        max_keypoints=args.max_keypoints,
        max_sessions=args.max_sessions,
        cache_dir=cache_dir,
    )

    # 5. Pipeline config
    config = OOFPipelineConfig(
        override_budget=args.override_budget,
        resume=not args.no_resume,
        max_sessions=args.max_sessions,
    )

    # 6. Budget pre-check at K=50 before committing GPU time.
    pre_budget = estimate_oof_budget(
        gallery_df=gallery_df,
        crop_df=crop_df,
        frozen_k=K_DEFAULT,
        max_sessions=args.max_sessions,
        gate_max_h100_hours=GATE_MAX_H100_HOURS,
        gate_max_cache_gb=GATE_MAX_CACHE_GB,
    )
    logger.info(
        "Pre-run budget estimate (K=%d): %d pairs, %.2f H100-hours, %.3f GB cache.",
        K_DEFAULT, pre_budget.n_pairs_estimated,
        pre_budget.estimated_h100_hours, pre_budget.estimated_cache_gb,
    )
    if not pre_budget.within_budget and not args.override_budget:
        raise BudgetExceededError(
            f"Pre-run budget estimate exceeds limits at K={K_DEFAULT}: "
            f"{pre_budget.estimated_h100_hours:.2f} H100-hours "
            f"(limit {GATE_MAX_H100_HOURS}) or "
            f"{pre_budget.estimated_cache_gb:.3f} GB cache "
            f"(limit {GATE_MAX_CACHE_GB}). "
            "Pass --override-budget to proceed."
        )

    # 7. Run full OOF calibration.
    logger.info("Starting run_oof_calibration → output: %s", output_dir)
    result = run_oof_calibration(
        config=config,
        gallery_df=gallery_df,
        crop_df=crop_df,
        descriptor_mappings=descriptor_mappings,
        embedding_matrices=embedding_matrices,
        calibrators_global=calibrators_global,
        local_scorer_body=scorer_body,
        local_scorer_ear=scorer_ear,
        output_dir=output_dir,
        fusion_weights_global=fusion_weights_global,
        override_budget=args.override_budget,
        source_fingerprint=source_fp,
        split_fingerprint=split_fp,
    )

    logger.info("OOF calibration complete.")
    logger.info("Output: %s", output_dir)
    logger.info("Frozen K: %s", result.get("frozen_k"))
    logger.info("Fusion weights: %s", result.get("fusion_weights"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
