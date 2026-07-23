#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Gallery-only local discrimination pilot (body + ear, no head).

Constructs a deterministic pseudo-query set from split=='gallery' images only,
runs the canonical StrictLocalMatcher + LocalIdentityScorer across multiple
configurations, and saves diagnostics under the experiment namespace.

Hard guarantees
---------------
* Probe and held_out_probe image IDs never enter any table; hard-fail on
  attempted access.
* selected-v1 production artifacts are never read for mutation.
* No real LightGlue / LoFTR inference is performed in this module; backends
  are injected by the caller.

Output namespace
----------------
  experiments/full_local_ensemble/local_pilot/
    config.json                – pilot configuration + fingerprint
    pilot_manifest.parquet     – pseudo-query manifest
    pair_scores/               – LocalPairScore records per configuration
    identity_scores/           – LocalIdentityScore records per configuration
    diagnostics/               – per-query stats (comparison counts, coverage)
    metrics.json               – gates + ROC/PR AUC + latency

Usage
-----
    python pipeline/local_signal_pilot.py run \\
        [--manifest PATH]          # bteh_image_manifest.parquet
        [--splits   PATH]          # bteh_splits.parquet
        [--crop-manifest PATH]     # crop manifest parquet
        [--embeddings-dir PATH]    # directory with miewid/ear_miewid_projected .npy
        [--output-dir PATH]        # default: EXPERIMENT_ROOT/local_pilot
        [--n-queries  N]           # default 120
        [--seed       N]           # default 42
        [--n-hard-neg N]           # default 3
        [--max-sessions N]         # default 3 (canonical cap)
        [--loftr-approved]         # pass to enable LoFTR config
        [--resume]                 # skip already-scored pairs

Estimation mode
---------------
    python pipeline/local_signal_pilot.py estimate \\
        [--manifest PATH] [--splits PATH] [--crop-manifest PATH]
        [--n-queries N] [--n-hard-neg N] [--max-sessions N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_bteh import EXPERIMENT_ROOT
from configs.config_elephant import (
    LOCAL_FEATURE_CACHE_MAX_LRU,
    LOCAL_IDENTITY_SCORER_MAX_SESSIONS,
    LOCAL_IDENTITY_SCORER_TOP_K,
    LOCAL_MATCHER_LOFTR_PILOT_APPROVED,
    LOCAL_SCORE_SCHEMA_VERSION,
    PRODUCTION_SELECTED_CHANNELS,
    PRODUCTION_FUSION_WEIGHTS,
)
from models.calibration import Calibrator
from models.feature_cache import FeatureCache
from models.local_matcher import (
    GEOM_HOMOGRAPHY,
    GEOM_PARTIAL_AFFINE,
    REGION_BODY,
    REGION_EAR,
    StrictLocalMatcher,
)
from models.local_score_schema import (
    LocalIdentityScore,
    LocalPairScore,
    assert_identity_score_integrity,
    assert_pair_score_integrity,
)
from models.identity_scorer import LocalIdentityScorer, QueryCrop, ReferenceImage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PILOT_SUBDIR: str = "local_pilot"
PILOT_OUTPUT_ROOT: Path = EXPERIMENT_ROOT / PILOT_SUBDIR

PILOT_CONFIG_FILENAME: str = "config.json"
PILOT_MANIFEST_FILENAME: str = "pilot_manifest.parquet"
PILOT_METRICS_FILENAME: str = "metrics.json"
PILOT_FINGERPRINT_FILENAME: str = "pilot_fingerprint.json"

PAIR_SCORES_SUBDIR: str = "pair_scores"
IDENTITY_SCORES_SUBDIR: str = "identity_scores"
DIAGNOSTICS_SUBDIR: str = "diagnostics"

DEFAULT_N_QUERIES: int = 120
DEFAULT_SEED: int = 42
DEFAULT_N_HARD_NEG: int = 3
DEFAULT_N_SESSIONS_CAP: int = LOCAL_IDENTITY_SCORER_MAX_SESSIONS
DEFAULT_TOP_K: int = LOCAL_IDENTITY_SCORER_TOP_K

# Hard probe split names — must NEVER appear in any pilot table
_PROBE_SPLITS: frozenset[str] = frozenset({"probe", "held_out_probe"})

# Configurations explored in the pilot
# Each entry: (config_id, region, geom_model, backend, exploratory)
_PILOT_CONFIGS: list[tuple[str, str, str, str, bool]] = [
    ("body_partial_affine", REGION_BODY, GEOM_PARTIAL_AFFINE, "lightglue", False),
    ("body_homography",     REGION_BODY, GEOM_HOMOGRAPHY,     "lightglue", True),
    ("ear_homography",      REGION_EAR,  GEOM_HOMOGRAPHY,     "lightglue", False),
    ("ear_loftr",           REGION_EAR,  GEOM_HOMOGRAPHY,     "loftr",     True),
]

# Coverage / quality gate thresholds
GATE_MIN_COVERAGE: float = 0.70
GATE_MIN_ROC_AUC: float = 0.60
GATE_MAX_H100_HOURS: float = 24.0
GATE_MAX_CACHE_GB: float = 50.0

# GPU throughput estimate (pairs/second on H100)
_H100_PAIRS_PER_SEC_LIGHTGLUE: float = 15.0
_H100_PAIRS_PER_SEC_LOFTR: float = 8.0
# Cache bytes per feature bundle (rough estimate for SuperPoint)
_BYTES_PER_FEATURE_BUNDLE: int = 1024 * 64  # ~64 KB


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
        probe_ids = bad["image_id"].tolist() if "image_id" in bad.columns else bad.index.tolist()
        raise RuntimeError(
            f"PROBE GUARD VIOLATION: {len(bad)} probe/held_out_probe rows "
            f"entered {context}. image_ids: {probe_ids[:10]}"
        )


def _filter_gallery_only(df: pd.DataFrame, split_col: str = "split") -> pd.DataFrame:
    """
    Return gallery rows from a table already screened for probe contamination.

    Mixed probe/gallery input is a hard error by design; callers must explicitly
    construct a gallery-only table before invoking pilot logic.
    """
    _assert_no_probe_ids(df, split_col, context="gallery_filter input")
    gallery = df[df[split_col] == "gallery"].copy()
    _assert_no_probe_ids(gallery, split_col, context="gallery_filter output")
    return gallery


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------

def _fingerprint_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _config_fingerprint(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Stratified pseudo-query sampling
# ---------------------------------------------------------------------------

def _image_count_quintile(n: int, breakpoints: list[int]) -> int:
    """Return 0–4 quintile label for image count *n* given precomputed breakpoints."""
    for i, bp in enumerate(breakpoints):
        if n <= bp:
            return i
    return 4


def sample_pseudo_queries(
    gallery_df: pd.DataFrame,
    crop_manifest_df: pd.DataFrame,
    *,
    n_queries: int = DEFAULT_N_QUERIES,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Select *n_queries* pseudo-query images from gallery_df, stratified by:
      - identity
      - session
      - crop availability (has body crop, has ear crop, has both)
      - image-count quintile (per identity)

    Returns a DataFrame with one row per pseudo-query including
    crop availability flags.

    Hard guarantee: no probe/held_out_probe images are ever included.
    """
    _assert_no_probe_ids(gallery_df, "split", context="pseudo_query sampling")

    rng = np.random.default_rng(seed)

    required_cols = {"image_id", "individual_id", "session_id", "split"}
    missing = required_cols - set(gallery_df.columns)
    if missing:
        raise ValueError(f"gallery_df is missing required columns: {missing}")

    # Accepted crops by kind
    accepted_crops = crop_manifest_df[
        crop_manifest_df["detector_status"] == "accepted"
    ].copy()
    body_ids = set(
        accepted_crops.loc[accepted_crops["crop_kind"] == "body", "image_id"].astype(str)
    )
    ear_ids = set(
        accepted_crops.loc[accepted_crops["crop_kind"] == "ear", "image_id"].astype(str)
    )

    gdf = gallery_df.copy()
    gdf["image_id"] = gdf["image_id"].astype(str)
    gdf["has_body"] = gdf["image_id"].isin(body_ids)
    gdf["has_ear"] = gdf["image_id"].isin(ear_ids)
    gdf["crop_stratum"] = gdf.apply(
        lambda r: "both" if r["has_body"] and r["has_ear"]
        else ("body_only" if r["has_body"] else ("ear_only" if r["has_ear"] else "none")),
        axis=1,
    )

    # Image-count quintile per identity
    id_counts = gdf.groupby("individual_id")["image_id"].transform("count")
    gdf["id_image_count"] = id_counts
    qbreaks = list(np.percentile(gdf["id_image_count"].values, [20, 40, 60, 80]).astype(int))
    gdf["img_count_quintile"] = gdf["id_image_count"].apply(
        lambda n: _image_count_quintile(n, qbreaks)
    )

    # Stratum key: identity × session × crop_stratum × quintile
    gdf["_stratum"] = (
        gdf["individual_id"].astype(str)
        + "|" + gdf["session_id"].astype(str)
        + "|" + gdf["crop_stratum"]
        + "|" + gdf["img_count_quintile"].astype(str)
    )

    # Keep only images that have at least one crop (body or ear)
    eligible = gdf[gdf["crop_stratum"] != "none"].copy()
    if eligible.empty:
        raise ValueError("No gallery images with accepted crops found.")

    strata = eligible["_stratum"].unique()
    n_strata = len(strata)
    per_stratum = max(1, math.ceil(n_queries / n_strata))

    selected_rows = []
    for stratum in strata:
        stratum_rows = eligible[eligible["_stratum"] == stratum]
        k = min(per_stratum, len(stratum_rows))
        idx = rng.choice(len(stratum_rows), size=k, replace=False)
        selected_rows.append(stratum_rows.iloc[idx])

    combined = pd.concat(selected_rows, ignore_index=True)

    # Trim or pad to exactly n_queries if overshooting
    if len(combined) > n_queries:
        # Deterministic trim: sort by a stable key then take first n_queries
        combined = combined.sort_values(
            ["individual_id", "session_id", "image_id"],
            kind="mergesort",
        ).head(n_queries).reset_index(drop=True)
    elif len(combined) < n_queries:
        # Sample additional rows from eligible pool not already selected
        already = set(combined["image_id"].tolist())
        pool = eligible[~eligible["image_id"].isin(already)]
        if not pool.empty:
            extra_n = min(n_queries - len(combined), len(pool))
            idx = rng.choice(len(pool), size=extra_n, replace=False)
            combined = pd.concat(
                [combined, pool.iloc[idx]], ignore_index=True
            )

    combined = combined.drop(columns=["_stratum"], errors="ignore")
    _assert_no_probe_ids(combined, "split", context="sampled pseudo-queries")
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Hard-negative selection using global channel embeddings
# ---------------------------------------------------------------------------

def select_hard_negatives(
    query_individual_id: str,
    query_image_id: str,
    gallery_df: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    embedding_index: dict[str, list[str]],
    *,
    n_hard_neg: int = DEFAULT_N_HARD_NEG,
    calibrators: Optional[dict[str, Calibrator]] = None,
    weights: Optional[dict[str, float]] = None,
) -> list[str]:
    """
    Select *n_hard_neg* hardest wrong identities for *query_individual_id* using
    the mean of selected-v1 global channels: body miewid + projected ear miewid.

    Truth is never forced: the positive identity is excluded from the ranking.
    Returns a list of individual_id strings (wrong identities).

    Parameters
    ----------
    query_individual_id:
        The correct identity — excluded from negatives.
    query_image_id:
        The query image id — used to find its embedding row.
    gallery_df:
        Gallery-only identity/image table.
    embeddings:
        Dict mapping channel name → (N, D) float32 embedding matrix.
    embedding_index:
        Dict mapping channel name → list of image_ids (row order).
    n_hard_neg:
        Number of hard negatives to return.
    """
    if not embeddings:
        # No embeddings available: fall back to deterministic identity order
        all_ids = sorted(
            set(gallery_df["individual_id"].tolist()) - {query_individual_id}
        )
        # Seed is derived from query_image_id for determinism
        rng_seed = int(hashlib.sha256(query_image_id.encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.default_rng(rng_seed)
        if len(all_ids) <= n_hard_neg:
            return all_ids
        idx = rng.choice(len(all_ids), size=n_hard_neg, replace=False)
        return [all_ids[i] for i in sorted(idx)]

    calibrators = calibrators or {}
    weights = weights or PRODUCTION_FUSION_WEIGHTS
    available_channels = [c for c in PRODUCTION_SELECTED_CHANNELS if c in embeddings]
    if not available_channels:
        logger.warning(
            "No selected-v1 channels found in embeddings for hard-neg selection; "
            "falling back to deterministic order."
        )
        all_ids = sorted(
            set(gallery_df["individual_id"].tolist()) - {query_individual_id}
        )
        rng_seed = int(hashlib.sha256(query_image_id.encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.default_rng(rng_seed)
        n = min(n_hard_neg, len(all_ids))
        idx = rng.choice(len(all_ids), size=n, replace=False)
        return [all_ids[i] for i in sorted(idx)]

    row_maps = {
        ch: {
            image_id: [
                index
                for index, candidate_image_id in enumerate(embedding_index[ch])
                if candidate_image_id == image_id
            ]
            for image_id in set(embedding_index[ch])
        }
        for ch in available_channels
    }
    query_rows = {
        ch: row_maps[ch].get(query_image_id, [])
        for ch in available_channels
    }
    if not any(query_rows.values()):
        all_ids = sorted(
            set(gallery_df["individual_id"].tolist()) - {query_individual_id}
        )
        return all_ids[: min(n_hard_neg, len(all_ids))]

    id_to_images = (
        gallery_df[gallery_df["individual_id"] != query_individual_id]
        .groupby("individual_id")["image_id"]
        .apply(list)
        .to_dict()
    )
    id_scores: list[tuple[str, float]] = []
    for ind_id, img_ids in id_to_images.items():
        weighted_score = 0.0
        available_weight = 0.0
        for ch in available_channels:
            q_rows = query_rows[ch]
            ref_rows = [
                row
                for img_id in img_ids
                for row in row_maps[ch].get(str(img_id), [])
            ]
            weight = float(weights.get(ch, 0.0))
            if not q_rows or not ref_rows or weight <= 0:
                continue
            raw_score = float(
                np.max(
                    embeddings[ch][q_rows]
                    @ embeddings[ch][ref_rows].T
                )
            )
            calibrated = (
                float(calibrators[ch].transform(np.array([raw_score]))[0])
                if ch in calibrators
                else raw_score
            )
            weighted_score += weight * calibrated
            available_weight += weight
        if available_weight > 0:
            id_scores.append((ind_id, weighted_score / available_weight))

    id_scores.sort(key=lambda item: (-item[1], item[0]))
    return [ind_id for ind_id, _ in id_scores[:n_hard_neg]]


# ---------------------------------------------------------------------------
# Reference session builder
# ---------------------------------------------------------------------------

def build_reference_sessions(
    candidate_individual_id: str,
    query_session_id: str,
    crop_manifest_df: pd.DataFrame,
    gallery_df: pd.DataFrame,
    region: str,
    *,
    max_sessions: int = DEFAULT_N_SESSIONS_CAP,
) -> list[ReferenceImage]:
    """
    Build reference images for *candidate_individual_id*, excluding
    *query_session_id*. Selects up to *max_sessions* distinct sessions,
    one image per session (first by crop_ordinal, then crop_id).

    Returns [] if no eligible crops found (not an error).
    """
    # Gallery images for this identity, not in the query session
    id_gallery = gallery_df[
        (gallery_df["individual_id"] == candidate_individual_id)
        & (gallery_df["session_id"].astype(str) != str(query_session_id))
    ]
    if id_gallery.empty:
        return []

    eligible_image_ids = set(id_gallery["image_id"].astype(str).tolist())

    # Accepted crops of the right kind
    accepted = crop_manifest_df[
        (crop_manifest_df["detector_status"] == "accepted")
        & (crop_manifest_df["crop_kind"] == region)
        & (crop_manifest_df["image_id"].astype(str).isin(eligible_image_ids))
    ].copy()
    if accepted.empty:
        return []

    # Join to get session_id
    merged = accepted.merge(
        gallery_df[["image_id", "session_id"]].drop_duplicates("image_id"),
        on="image_id",
        how="left",
    )

    # Sort: session_id, then crop_ordinal
    merged["image_id"] = merged["image_id"].astype(str)
    merged = merged.sort_values(
        ["session_id", "crop_ordinal", "crop_id"], kind="mergesort"
    )

    refs: list[ReferenceImage] = []
    seen_sessions: set[str] = set()
    for _, row in merged.iterrows():
        sess = str(row["session_id"])
        if sess in seen_sessions:
            continue
        if len(seen_sessions) >= max_sessions:
            break
        refs.append(
            ReferenceImage(
                crop_id=str(row["crop_id"]),
                crop_path=str(row["crop_path"]),
                crop_kind=region,
                session_id=sess,
                individual_id=candidate_individual_id,
            )
        )
        seen_sessions.add(sess)
    return refs


def build_query_crops(
    query_image_id: str,
    crop_manifest_df: pd.DataFrame,
    region: str,
) -> list[QueryCrop]:
    """Return accepted query crops of *region* for *query_image_id*."""
    accepted = crop_manifest_df[
        (crop_manifest_df["detector_status"] == "accepted")
        & (crop_manifest_df["crop_kind"] == region)
        & (crop_manifest_df["image_id"].astype(str) == str(query_image_id))
    ]
    return [
        QueryCrop(
            crop_id=str(row["crop_id"]),
            crop_path=str(row["crop_path"]),
            crop_kind=region,
        )
        for _, row in accepted.iterrows()
    ]


# ---------------------------------------------------------------------------
# Pair-count / budget estimation
# ---------------------------------------------------------------------------

@dataclass
class BudgetEstimate:
    n_queries: int
    n_hard_neg: int
    n_sessions_cap: int
    configs: list[str]
    total_pairs_per_config: dict[str, int]
    total_pairs: int
    unique_crops: int
    estimated_cache_bytes: int
    estimated_cache_gb: float
    estimated_h100_seconds: float
    estimated_h100_hours: float
    within_budget: bool
    projected_oof_pairs: int
    projected_fixed_probe_pairs: int


def estimate_budget(
    gallery_df: pd.DataFrame,
    crop_manifest_df: pd.DataFrame,
    configs: list[tuple[str, str, str, str, bool]],
    *,
    n_queries: int = DEFAULT_N_QUERIES,
    n_hard_neg: int = DEFAULT_N_HARD_NEG,
    n_sessions_cap: int = DEFAULT_N_SESSIONS_CAP,
    n_gallery_identities: Optional[int] = None,
) -> BudgetEstimate:
    """
    Estimate pair count, cache size, and GPU time for the pilot budget.

    OOF budget: n_gallery_identities × mean_sessions_per_id × pairs_per_session
    Fixed probe: n_queries × (n_hard_neg + 1) reference sessions × pairs
    """
    accepted = crop_manifest_df[crop_manifest_df["detector_status"] == "accepted"]
    body_per_session = 1  # one body crop per image, one image per session
    ear_per_image = accepted[accepted["crop_kind"] == "ear"].groupby("image_id").size().mean()
    ear_per_image = float(ear_per_image) if not math.isnan(float(ear_per_image if not isinstance(ear_per_image, float) else ear_per_image)) else 1.0
    ear_per_session = max(1.0, ear_per_image)

    n_candidates = n_hard_neg + 1  # pos + hard_negs

    pairs_per_config: dict[str, int] = {}
    total = 0
    for cfg_id, region, geom, backend, _ in configs:
        if region == REGION_BODY:
            pairs_q = n_queries * n_candidates * n_sessions_cap * body_per_session
        else:
            pairs_q = int(n_queries * n_candidates * n_sessions_cap * ear_per_session)
        pairs_per_config[cfg_id] = pairs_q
        total += pairs_q

    # Unique crops (body + ear per query image × sessions)
    gallery_images = len(gallery_df)
    unique_crops = int(gallery_images * 2)  # rough body+ear estimate
    cache_bytes = unique_crops * _BYTES_PER_FEATURE_BUNDLE
    cache_gb = cache_bytes / (1024**3)

    lg_pairs = sum(
        v for (cfg_id, _, _, backend, _), v in zip(configs, pairs_per_config.values())
        if backend == "lightglue"
    )
    loftr_pairs = total - lg_pairs
    gpu_secs = (
        lg_pairs / _H100_PAIRS_PER_SEC_LIGHTGLUE
        + loftr_pairs / _H100_PAIRS_PER_SEC_LOFTR
    )
    gpu_hours = gpu_secs / 3600.0

    if n_gallery_identities is None:
        n_gallery_identities = gallery_df["individual_id"].nunique()
    mean_sessions = (
        gallery_df.groupby("individual_id")["session_id"].nunique().mean()
    )
    oof_pairs = int(n_gallery_identities * float(mean_sessions) * n_sessions_cap * 2)
    fixed_probe_pairs = int(n_queries * n_candidates * n_sessions_cap * 2)

    return BudgetEstimate(
        n_queries=n_queries,
        n_hard_neg=n_hard_neg,
        n_sessions_cap=n_sessions_cap,
        configs=[c[0] for c in configs],
        total_pairs_per_config=pairs_per_config,
        total_pairs=total,
        unique_crops=unique_crops,
        estimated_cache_bytes=cache_bytes,
        estimated_cache_gb=round(cache_gb, 3),
        estimated_h100_seconds=round(gpu_secs, 1),
        estimated_h100_hours=round(gpu_hours, 3),
        within_budget=(gpu_hours <= GATE_MAX_H100_HOURS and cache_gb <= GATE_MAX_CACHE_GB),
        projected_oof_pairs=oof_pairs,
        projected_fixed_probe_pairs=fixed_probe_pairs,
    )


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _compute_roc_auc(labels: list[int], scores: list[float]) -> float:
    """Compute tie-safe ROC AUC from binary labels and scores."""
    if not labels:
        return float("nan")
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(labels, scores)), 4)


def _compute_pr_auc(labels: list[int], scores: list[float]) -> float:
    """Compute tie-safe average precision."""
    if not labels:
        return float("nan")
    pos = sum(labels)
    if pos == 0:
        return float("nan")
    from sklearn.metrics import average_precision_score
    return round(float(average_precision_score(labels, scores)), 4)


def _platt_calibration_check(
    labels: list[int], scores: list[float]
) -> dict[str, Any]:
    """
    Check Platt calibration support and flatness without scipy/sklearn.

    Returns:
        has_support: bool – at least 10 positive and 10 negative scores
        is_flat: bool – score distributions for pos/neg overlap heavily
        pos_mean, neg_mean, pos_std, neg_std
    """
    pos_scores = [s for s, l in zip(scores, labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0]
    has_support = len(pos_scores) >= 10 and len(neg_scores) >= 10
    if not pos_scores or not neg_scores:
        return {
            "has_support": has_support,
            "is_flat": True,
            "pos_mean": float("nan"),
            "neg_mean": float("nan"),
            "pos_std": float("nan"),
            "neg_std": float("nan"),
        }
    pos_mean = float(np.mean(pos_scores))
    neg_mean = float(np.mean(neg_scores))
    pos_std = float(np.std(pos_scores)) if len(pos_scores) > 1 else 0.0
    neg_std = float(np.std(neg_scores)) if len(neg_scores) > 1 else 0.0
    # Flat if separation is less than 0.5 * pooled std
    pooled_std = max(1e-8, (pos_std + neg_std) / 2)
    is_flat = abs(pos_mean - neg_mean) < 0.5 * pooled_std
    return {
        "has_support": has_support,
        "is_flat": is_flat,
        "pos_mean": round(pos_mean, 4),
        "neg_mean": round(neg_mean, 4),
        "pos_std": round(pos_std, 4),
        "neg_std": round(neg_std, 4),
    }


def compute_metrics(
    identity_scores: list[LocalIdentityScore],
    positive_flags: list[bool],
    calibration_only_flags: list[bool],
    config_id: str,
    region: str,
    backend: str,
    geom_model: str,
) -> dict[str, Any]:
    """
    Compute per-config metrics from identity scores.

    Positive rows flagged as calibration-only are excluded from candidate-ranking
    metrics elsewhere, but remain required for the positive-vs-hard-negative
    discrimination AUC computed here.
    """
    total = len(identity_scores)
    valid = [s for s in identity_scores if s.score > 0]
    coverage = len(valid) / total if total > 0 else 0.0

    discrimination_scores = [score.score for score in identity_scores]
    discrimination_labels = [1 if is_pos else 0 for is_pos in positive_flags]

    roc_auc = _compute_roc_auc(discrimination_labels, discrimination_scores)
    pr_auc = _compute_pr_auc(discrimination_labels, discrimination_scores)

    # Calibration diagnostics: include calibration-only positive rows
    calib_scores = [s.score for s, f in zip(identity_scores, positive_flags) if f]
    hard_neg_scores = [s.score for s, f in zip(identity_scores, positive_flags) if not f]
    calib_check = _platt_calibration_check(
        [1] * len(calib_scores) + [0] * len(hard_neg_scores),
        calib_scores + hard_neg_scores,
    )

    # Inlier distributions
    all_inliers = [
        ps.n_inliers
        for s in identity_scores
        for ps in s.pair_scores
    ]
    all_ratios = [
        ps.inlier_ratio
        for s in identity_scores
        for ps in s.pair_scores
    ]
    pos_inliers = [
        ps.n_inliers
        for s, is_pos in zip(identity_scores, positive_flags)
        if is_pos
        for ps in s.pair_scores
    ]
    neg_inliers = [
        ps.n_inliers
        for s, is_pos in zip(identity_scores, positive_flags)
        if not is_pos
        for ps in s.pair_scores
    ]

    latencies = [s.latency_ms for s in identity_scores]

    return {
        "config_id": config_id,
        "region": region,
        "backend": backend,
        "geom_model": geom_model,
        "n_total": total,
        "n_valid": len(valid),
        "coverage": round(coverage, 4),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "calibration": calib_check,
        "pos_score_median": round(float(np.median(calib_scores)), 4) if calib_scores else float("nan"),
        "neg_score_median": round(float(np.median(hard_neg_scores)), 4) if hard_neg_scores else float("nan"),
        "pos_inliers_median": round(float(np.median(pos_inliers)), 1) if pos_inliers else float("nan"),
        "pos_inliers_p95": round(float(np.percentile(pos_inliers, 95)), 1) if pos_inliers else float("nan"),
        "neg_inliers_median": round(float(np.median(neg_inliers)), 1) if neg_inliers else float("nan"),
        "neg_inliers_p95": round(float(np.percentile(neg_inliers, 95)), 1) if neg_inliers else float("nan"),
        "inlier_ratio_median": round(float(np.median(all_ratios)), 4) if all_ratios else float("nan"),
        "inlier_ratio_p95": round(float(np.percentile(all_ratios, 95)), 4) if all_ratios else float("nan"),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else float("nan"),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else float("nan"),
    }


def check_gates(metrics: dict[str, Any]) -> dict[str, Any]:
    """Check approval gates against precomputed metrics dict."""
    coverage = metrics.get("coverage", 0.0)
    roc_auc = metrics.get("roc_auc", float("nan"))
    calib = metrics.get("calibration", {})
    is_flat = calib.get("is_flat", True)
    has_calib_support = calib.get("has_support", False)

    gate_coverage = coverage >= GATE_MIN_COVERAGE
    gate_roc = (not math.isnan(roc_auc)) and roc_auc >= GATE_MIN_ROC_AUC
    gate_calib = has_calib_support and not is_flat
    approved = gate_coverage and gate_roc and gate_calib

    return {
        "approved": approved,
        "gate_coverage": gate_coverage,
        "gate_roc_auc": gate_roc,
        "gate_calibration": gate_calib,
        "note": "Budget gate checked separately via estimate_budget()",
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def write_pilot_config(
    output_dir: Path,
    config: dict,
    fingerprint: str,
) -> None:
    _write_json(output_dir / PILOT_CONFIG_FILENAME, config)
    _write_json(
        output_dir / PILOT_FINGERPRINT_FILENAME,
        {"fingerprint": fingerprint, "schema_version": LOCAL_SCORE_SCHEMA_VERSION},
    )


def write_pilot_manifest(output_dir: Path, manifest_df: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / PILOT_MANIFEST_FILENAME
    manifest_df.to_parquet(out, index=False)
    logger.info("Wrote pilot manifest: %s (%d rows)", out, len(manifest_df))


def write_identity_scores(
    output_dir: Path,
    config_id: str,
    scores: list[LocalIdentityScore],
    positive_flags: list[bool],
    calibration_only_flags: list[bool],
) -> None:
    subdir = output_dir / IDENTITY_SCORES_SUBDIR
    subdir.mkdir(parents=True, exist_ok=True)
    records = []
    for s, is_pos, is_calib_only in zip(scores, positive_flags, calibration_only_flags):
        records.append({
            "config_id": config_id,
            "schema_version": s.schema_version,
            "backend": s.backend,
            "model_fingerprint": s.model_fingerprint,
            "scoring_fingerprint": s.scoring_fingerprint,
            "query_crop_kind": s.query_crop_kind,
            "candidate_individual_id": s.candidate_individual_id,
            "n_pairs_attempted": s.n_pairs_attempted,
            "n_pairs_valid": s.n_pairs_valid,
            "n_pairs_missing_file": s.n_pairs_missing_file,
            "n_sessions_used": s.n_sessions_used,
            "aggregation_method": s.aggregation_method,
            "top_k": s.top_k,
            "score": s.score,
            "is_positive": is_pos,
            "is_calibration_only": is_calib_only,
            "latency_ms": s.latency_ms,
        })
    df = pd.DataFrame(records)
    out = subdir / f"{config_id}_identity_scores.parquet"
    df.to_parquet(out, index=False)
    logger.info("Wrote identity scores: %s (%d rows)", out, len(df))


def write_pair_scores(
    output_dir: Path,
    config_id: str,
    scores: list[LocalIdentityScore],
) -> None:
    subdir = output_dir / PAIR_SCORES_SUBDIR
    subdir.mkdir(parents=True, exist_ok=True)
    records = []
    for identity_score in scores:
        for ps in identity_score.pair_scores:
            records.append({
                "config_id": config_id,
                "candidate_id": identity_score.candidate_individual_id,
                "schema_version": ps.schema_version,
                "backend": ps.backend,
                "model_fingerprint": ps.model_fingerprint,
                "scoring_fingerprint": ps.scoring_fingerprint,
                "query_crop_id": ps.query_crop_id,
                "ref_crop_id": ps.ref_crop_id,
                "query_crop_kind": ps.query_crop_kind,
                "ref_crop_kind": ps.ref_crop_kind,
                "region": ps.region,
                "orientation": ps.orientation,
                "geom_model_used": ps.geom_model_used,
                "n_raw_matches": ps.n_raw_matches,
                "n_inliers": ps.n_inliers,
                "inlier_ratio": ps.inlier_ratio,
                "score": ps.score,
                "missing_file": ps.missing_file,
                "latency_ms": ps.latency_ms,
            })
    df = pd.DataFrame(records)
    out = subdir / f"{config_id}_pair_scores.parquet"
    df.to_parquet(out, index=False)
    logger.info("Wrote pair scores: %s (%d rows)", out, len(df))


def write_diagnostics(
    output_dir: Path,
    config_id: str,
    diagnostics: list[dict],
) -> None:
    subdir = output_dir / DIAGNOSTICS_SUBDIR
    subdir.mkdir(parents=True, exist_ok=True)
    out = subdir / f"{config_id}_diagnostics.parquet"
    pd.DataFrame(diagnostics).to_parquet(out, index=False)
    logger.info("Wrote diagnostics: %s (%d rows)", out, len(diagnostics))


def write_metrics(
    output_dir: Path,
    metrics_list: list[dict],
    gates_list: list[dict],
    budget: BudgetEstimate,
) -> None:
    result = {
        "config_metrics": metrics_list,
        "gates": gates_list,
        "budget": {
            "n_queries": budget.n_queries,
            "total_pairs": budget.total_pairs,
            "estimated_cache_gb": budget.estimated_cache_gb,
            "estimated_h100_hours": budget.estimated_h100_hours,
            "within_budget": budget.within_budget,
            "projected_oof_pairs": budget.projected_oof_pairs,
            "projected_fixed_probe_pairs": budget.projected_fixed_probe_pairs,
        },
    }
    _write_json(output_dir / PILOT_METRICS_FILENAME, result)
    logger.info("Wrote metrics: %s", output_dir / PILOT_METRICS_FILENAME)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _load_cached_identity_scores(
    output_dir: Path,
    config_id: str,
) -> Optional[pd.DataFrame]:
    """Load previously written identity scores for resume; None if not found."""
    path = output_dir / IDENTITY_SCORES_SUBDIR / f"{config_id}_identity_scores.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Could not read cached scores %s: %s", path, exc)
    return None


def _check_fingerprint(output_dir: Path, expected_fingerprint: str) -> bool:
    """Return True if the stored fingerprint matches the expected one."""
    path = output_dir / PILOT_FINGERPRINT_FILENAME
    if not path.exists():
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            stored = json.load(fh)
        return stored.get("fingerprint") == expected_fingerprint
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main pilot runner
# ---------------------------------------------------------------------------

@dataclass
class PilotConfig:
    n_queries: int = DEFAULT_N_QUERIES
    seed: int = DEFAULT_SEED
    n_hard_neg: int = DEFAULT_N_HARD_NEG
    max_sessions: int = DEFAULT_N_SESSIONS_CAP
    top_k: int = DEFAULT_TOP_K
    loftr_approved: bool = LOCAL_MATCHER_LOFTR_PILOT_APPROVED
    schema_version: str = LOCAL_SCORE_SCHEMA_VERSION
    production_channels: list = field(default_factory=lambda: list(PRODUCTION_SELECTED_CHANNELS))
    gate_min_coverage: float = GATE_MIN_COVERAGE
    gate_min_roc_auc: float = GATE_MIN_ROC_AUC
    gate_max_h100_hours: float = GATE_MAX_H100_HOURS
    gate_max_cache_gb: float = GATE_MAX_CACHE_GB
    config_ids: Optional[list[str]] = None
    loftr_max_edge: int = 640


def run_pilot(
    manifest_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    crop_manifest_df: pd.DataFrame,
    scorer_factory,  # callable(config_id, region, geom_model, backend) -> LocalIdentityScorer
    *,
    output_dir: Optional[Path] = None,
    config: Optional[PilotConfig] = None,
    embeddings: Optional[dict[str, np.ndarray]] = None,
    embedding_index: Optional[dict[str, list[str]]] = None,
    calibrators: Optional[dict[str, Calibrator]] = None,
    fusion_weights: Optional[dict[str, float]] = None,
    resume: bool = False,
) -> dict[str, Any]:
    """
    Run the full local discrimination pilot.

    Parameters
    ----------
    manifest_df:
        Full image manifest (all splits allowed; pilot filters to gallery).
    splits_df:
        Split assignments (must have 'image_id', 'split', 'session_id').
    crop_manifest_df:
        Accepted crop manifest (must have 'image_id', 'crop_id', 'crop_path',
        'crop_kind', 'detector_status').
    scorer_factory:
        Callable that returns a LocalIdentityScorer for the given config.
        Signature: scorer_factory(config_id, region, geom_model, backend)
    output_dir:
        Root output directory (default: PILOT_OUTPUT_ROOT).
    config:
        PilotConfig instance (default: PilotConfig()).
    embeddings:
        Optional dict channel→(N,D) float32 embedding matrix for hard-neg.
    embedding_index:
        Optional dict channel→list[image_id] row mapping.
    resume:
        If True, skip configs whose output already exists and whose fingerprint
        matches.

    Returns
    -------
    dict with keys: pilot_manifest, metrics, gates, budget, fingerprint
    """
    if config is None:
        config = PilotConfig()
    if output_dir is None:
        output_dir = PILOT_OUTPUT_ROOT
    if embeddings is None:
        embeddings = {}
    if embedding_index is None:
        embedding_index = {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hard-fail probe guard: ensure splits_df doesn't accidentally bring
    # probe images through, and ensure manifest_df is clean at the join
    _assert_no_probe_ids(
        splits_df.merge(
            manifest_df[["image_id"]], on="image_id", how="inner"
        ).pipe(lambda d: d[d["split"].isin(_PROBE_SPLITS)]) if "split" in splits_df.columns else splits_df.head(0),
        "split",
        context="splits_df probe check",
    )

    # Build gallery-only dataset
    gallery_split = splits_df[splits_df["split"] == "gallery"].copy()
    _assert_no_probe_ids(gallery_split, "split", context="gallery_split")

    # Merge in individual_id from manifest only if not already present in splits_df
    if "individual_id" not in gallery_split.columns:
        gallery_with_sessions = gallery_split.merge(
            manifest_df[["image_id", "individual_id"]].drop_duplicates("image_id"),
            on="image_id",
            how="left",
        )
    else:
        gallery_with_sessions = gallery_split.copy()

    # Drop any duplicate suffix columns from a prior join
    dup_cols = [c for c in gallery_with_sessions.columns if c.endswith("_x") or c.endswith("_y")]
    if dup_cols:
        # Keep _x variant, drop _y; rename _x → original name
        to_drop = [c for c in dup_cols if c.endswith("_y")]
        to_rename = {c: c[:-2] for c in dup_cols if c.endswith("_x") and c[:-2] not in gallery_with_sessions.columns}
        gallery_with_sessions = gallery_with_sessions.drop(columns=to_drop, errors="ignore")
        gallery_with_sessions = gallery_with_sessions.rename(columns=to_rename)

    # Validate no cross-contamination
    _assert_no_probe_ids(gallery_with_sessions, "split", context="gallery_with_sessions")

    # Pseudo-query sampling
    logger.info("Sampling %d pseudo-queries (seed=%d)...", config.n_queries, config.seed)
    pilot_manifest = sample_pseudo_queries(
        gallery_with_sessions,
        crop_manifest_df,
        n_queries=config.n_queries,
        seed=config.seed,
    )
    logger.info("Pilot manifest: %d rows", len(pilot_manifest))

    # Build config fingerprint
    config_dict = {
        "n_queries": config.n_queries,
        "seed": config.seed,
        "n_hard_neg": config.n_hard_neg,
        "max_sessions": config.max_sessions,
        "top_k": config.top_k,
        "loftr_approved": config.loftr_approved,
        "schema_version": config.schema_version,
        "production_channels": config.production_channels,
        "pilot_manifest_fingerprint": _fingerprint_str(
            "\n".join(sorted(pilot_manifest["image_id"].astype(str).tolist()))
        ),
    }
    fingerprint = _config_fingerprint(config_dict)

    # Fingerprint mismatch check for resume
    if resume and not _check_fingerprint(output_dir, fingerprint):
        logger.warning(
            "Stored fingerprint mismatch — resuming with fresh output directory."
        )
        resume = False

    write_pilot_config(output_dir, config_dict, fingerprint)
    write_pilot_manifest(output_dir, pilot_manifest)

    # Determine which configs to run
    active_configs = [
        c for c in _PILOT_CONFIGS
        if (c[3] != "loftr" or config.loftr_approved)
        and (config.config_ids is None or c[0] in config.config_ids)
    ]
    if not active_configs:
        raise ValueError("No pilot configurations are enabled")

    all_metrics: list[dict] = []
    all_gates: list[dict] = []

    for cfg_id, region, geom, backend, exploratory in active_configs:
        logger.info(
            "Config %s (region=%s, geom=%s, backend=%s, exploratory=%s)",
            cfg_id, region, geom, backend, exploratory,
        )

        # Resume check
        if resume and _load_cached_identity_scores(output_dir, cfg_id) is not None:
            logger.info("Config %s already scored; skipping (resume=True).", cfg_id)
            continue

        scorer = scorer_factory(cfg_id, region, geom, backend)

        identity_scores_list: list[LocalIdentityScore] = []
        positive_flags: list[bool] = []
        calibration_only_flags: list[bool] = []
        diagnostics_list: list[dict] = []

        for _, qrow in pilot_manifest.iterrows():
            q_image_id = str(qrow["image_id"])
            q_individual_id = str(qrow["individual_id"])
            q_session_id = str(qrow["session_id"])

            # Build query crops
            query_crops = build_query_crops(q_image_id, crop_manifest_df, region)
            if not query_crops:
                logger.debug(
                    "No %s crops for query image %s; skipping.", region, q_image_id
                )
                continue

            # Hard-negative identities from global channels
            hard_neg_ids = select_hard_negatives(
                q_individual_id,
                q_image_id,
                gallery_with_sessions,
                embeddings,
                embedding_index,
                n_hard_neg=config.n_hard_neg,
                calibrators=calibrators,
                weights=fusion_weights,
            )

            # Positive candidate: same identity in other sessions
            pos_refs = build_reference_sessions(
                q_individual_id,
                q_session_id,
                crop_manifest_df,
                gallery_with_sessions,
                region,
                max_sessions=config.max_sessions,
            )

            # Score positive (calibration-only: excluded from ranking metrics)
            if pos_refs:
                pos_score = scorer.score_identity(
                    query_crops,
                    pos_refs,
                    candidate_individual_id=q_individual_id,
                )
                identity_scores_list.append(pos_score)
                positive_flags.append(True)
                calibration_only_flags.append(True)  # positive not forced into ranking
                diagnostics_list.append({
                    "query_image_id": q_image_id,
                    "query_individual_id": q_individual_id,
                    "query_session_id": q_session_id,
                    "candidate_individual_id": q_individual_id,
                    "is_positive": True,
                    "is_calibration_only": True,
                    "n_refs": len(pos_refs),
                    "n_pairs_attempted": pos_score.n_pairs_attempted,
                    "n_pairs_valid": pos_score.n_pairs_valid,
                    "score": pos_score.score,
                    "config_id": cfg_id,
                    "region": region,
                })

            # Score hard negatives
            for neg_id in hard_neg_ids:
                neg_refs = build_reference_sessions(
                    neg_id,
                    q_session_id,  # session exclusion applied consistently
                    crop_manifest_df,
                    gallery_with_sessions,
                    region,
                    max_sessions=config.max_sessions,
                )
                if not neg_refs:
                    continue
                neg_score = scorer.score_identity(
                    query_crops,
                    neg_refs,
                    candidate_individual_id=neg_id,
                )
                identity_scores_list.append(neg_score)
                positive_flags.append(False)
                calibration_only_flags.append(False)
                diagnostics_list.append({
                    "query_image_id": q_image_id,
                    "query_individual_id": q_individual_id,
                    "query_session_id": q_session_id,
                    "candidate_individual_id": neg_id,
                    "is_positive": False,
                    "is_calibration_only": False,
                    "n_refs": len(neg_refs),
                    "n_pairs_attempted": neg_score.n_pairs_attempted,
                    "n_pairs_valid": neg_score.n_pairs_valid,
                    "score": neg_score.score,
                    "config_id": cfg_id,
                    "region": region,
                })

        if identity_scores_list:
            write_identity_scores(
                output_dir, cfg_id,
                identity_scores_list, positive_flags, calibration_only_flags,
            )
            write_pair_scores(output_dir, cfg_id, identity_scores_list)
            write_diagnostics(output_dir, cfg_id, diagnostics_list)

            metrics = compute_metrics(
                identity_scores_list,
                positive_flags,
                calibration_only_flags,
                config_id=cfg_id,
                region=region,
                backend=backend,
                geom_model=geom,
            )
            metrics["exploratory"] = exploratory
            gates = check_gates(metrics)
            gates["config_id"] = cfg_id
            all_metrics.append(metrics)
            all_gates.append(gates)
        else:
            logger.warning("Config %s: no identity scores produced.", cfg_id)

    budget = estimate_budget(
        gallery_with_sessions,
        crop_manifest_df,
        active_configs,
        n_queries=config.n_queries,
        n_hard_neg=config.n_hard_neg,
        n_sessions_cap=config.max_sessions,
    )
    write_metrics(output_dir, all_metrics, all_gates, budget)

    return {
        "pilot_manifest": pilot_manifest,
        "metrics": all_metrics,
        "gates": all_gates,
        "budget": budget,
        "fingerprint": fingerprint,
        "output_dir": str(output_dir),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gallery-only local discrimination pilot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # run subcommand
    run_p = sub.add_parser("run", help="Run the pilot.")
    run_p.add_argument("--manifest", help="Image manifest parquet path.")
    run_p.add_argument("--splits", help="Splits parquet path.")
    run_p.add_argument("--crop-manifest", help="Crop manifest parquet path.")
    run_p.add_argument("--embeddings-dir", help="Dir with .npy embedding files.")
    run_p.add_argument(
        "--output-dir",
        default=str(PILOT_OUTPUT_ROOT),
        help="Output directory (default: EXPERIMENT_ROOT/local_pilot).",
    )
    run_p.add_argument("--n-queries", type=int, default=DEFAULT_N_QUERIES)
    run_p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run_p.add_argument("--n-hard-neg", type=int, default=DEFAULT_N_HARD_NEG)
    run_p.add_argument("--max-sessions", type=int, default=DEFAULT_N_SESSIONS_CAP)
    run_p.add_argument("--loftr-approved", action="store_true")
    run_p.add_argument("--resume", action="store_true")
    run_p.add_argument("--calibration-dir", required=True)
    run_p.add_argument("--device", default="cuda")
    run_p.add_argument("--disable-cudnn", action="store_true")
    run_p.add_argument("--max-keypoints", type=int, default=2048)
    run_p.add_argument("--cache-dir", default=None)
    run_p.add_argument("--loftr-max-edge", type=int, default=640)
    run_p.add_argument(
        "--configs",
        nargs="+",
        choices=[config[0] for config in _PILOT_CONFIGS],
        default=None,
    )

    # estimate subcommand
    est_p = sub.add_parser("estimate", help="Estimate pair counts and budget.")
    est_p.add_argument("--manifest", help="Image manifest parquet path.")
    est_p.add_argument("--splits", help="Splits parquet path.")
    est_p.add_argument("--crop-manifest", help="Crop manifest parquet path.")
    est_p.add_argument("--n-queries", type=int, default=DEFAULT_N_QUERIES)
    est_p.add_argument("--n-hard-neg", type=int, default=DEFAULT_N_HARD_NEG)
    est_p.add_argument("--max-sessions", type=int, default=DEFAULT_N_SESSIONS_CAP)

    return parser


def _load_embeddings(
    embeddings_dir: str,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Load selected-v1 channel embeddings from *embeddings_dir*."""
    embeddings: dict[str, np.ndarray] = {}
    embedding_index: dict[str, list[str]] = {}
    if not embeddings_dir:
        return embeddings, embedding_index

    edir = Path(embeddings_dir)
    for ch in PRODUCTION_SELECTED_CHANNELS:
        npy_path = edir / f"{ch}.npy"
        mapping_path = edir / f"{ch}_mapping.parquet"
        if npy_path.exists() and mapping_path.exists():
            try:
                embeddings[ch] = np.load(str(npy_path))
                mapping = pd.read_parquet(mapping_path).sort_values(
                    "embedding_row", kind="mergesort"
                )
                expected = np.arange(len(mapping), dtype=np.int64)
                actual = mapping["embedding_row"].to_numpy(dtype=np.int64)
                if not np.array_equal(actual, expected):
                    raise ValueError(
                        f"{ch} mapping embedding rows are not contiguous"
                    )
                if len(mapping) != len(embeddings[ch]):
                    raise ValueError(
                        f"{ch} mapping/matrix row count mismatch"
                    )
                embedding_index[ch] = mapping["image_id"].astype(str).tolist()
                logger.info("Loaded %s embeddings: %s", ch, embeddings[ch].shape)
            except Exception as exc:
                raise RuntimeError(f"Could not load {ch} embeddings: {exc}") from exc
    return embeddings, embedding_index


def _load_selected_calibration(
    calibration_dir: Path,
) -> tuple[dict[str, Calibrator], dict[str, float]]:
    calibrators = {}
    for channel in PRODUCTION_SELECTED_CHANNELS:
        path = calibration_dir / f"{channel}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing selected calibrator: {path}")
        calibrators[channel] = Calibrator().load(str(path))
    weights_path = calibration_dir / "fusion_weights.json"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing fusion weights: {weights_path}")
    weights = json.loads(weights_path.read_text())["weights"]
    return calibrators, {key: float(value) for key, value in weights.items()}


def _make_real_scorer_factory(
    output_dir: Path,
    *,
    device: str,
    disable_cudnn: bool,
    max_keypoints: int,
    max_sessions: int,
    allow_loftr_pilot: bool,
    loftr_max_edge: int,
    cache_dir: Optional[Path] = None,
):
    matchers: dict[str, StrictLocalMatcher] = {}
    caches: dict[str, FeatureCache] = {}
    cache_root = cache_dir or (output_dir / "feature_cache")

    def factory(cfg_id: str, region: str, geom: str, backend: str):
        if backend not in matchers:
            matchers[backend] = StrictLocalMatcher(
                backend=backend,
                max_keypoints=max_keypoints,
                device=device,
                disable_cudnn=disable_cudnn,
                allow_loftr_pilot=allow_loftr_pilot,
                loftr_max_edge=loftr_max_edge,
            )
        matcher = matchers[backend]
        cache = None
        if backend == "lightglue":
            if backend not in caches:
                caches[backend] = FeatureCache(
                    cache_root / backend,
                    matcher.model_fingerprint,
                    max_lru_entries=LOCAL_FEATURE_CACHE_MAX_LRU,
                )
            cache = caches[backend]
        return LocalIdentityScorer(
            matcher=matcher,
            cache=cache,
            max_sessions=max_sessions,
            top_k=LOCAL_IDENTITY_SCORER_TOP_K,
            geom_model=geom,
        )

    return factory


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.manifest is None or args.splits is None or args.crop_manifest is None:
        print("--manifest, --splits, and --crop-manifest are required.", file=sys.stderr)
        return 2

    splits_df = pd.read_parquet(
        args.splits,
        filters=[("split", "==", "gallery")],
    )
    gallery_ids = splits_df["image_id"].astype(str).unique().tolist()
    manifest_df = pd.read_parquet(
        args.manifest,
        filters=[("image_id", "in", gallery_ids)],
    )
    crop_manifest_df = pd.read_parquet(
        args.crop_manifest,
        filters=[("image_id", "in", gallery_ids)],
    )

    if args.command == "estimate":
        gallery_split = splits_df[splits_df["split"] == "gallery"].copy()
        if "individual_id" in gallery_split.columns:
            gallery_with_sessions = gallery_split
        else:
            gallery_with_sessions = gallery_split.merge(
                manifest_df[["image_id", "individual_id"]].drop_duplicates("image_id"),
                on="image_id",
                how="left",
            )
        budget = estimate_budget(
            gallery_with_sessions,
            crop_manifest_df,
            _PILOT_CONFIGS,
            n_queries=args.n_queries,
            n_hard_neg=args.n_hard_neg,
            n_sessions_cap=args.max_sessions,
        )
        print(json.dumps(asdict(budget), indent=2))
        return 0

    # run
    embeddings, embedding_index = _load_embeddings(
        getattr(args, "embeddings_dir", None) or ""
    )
    calibrators, fusion_weights = _load_selected_calibration(
        Path(args.calibration_dir)
    )

    pilot_config = PilotConfig(
        n_queries=args.n_queries,
        seed=args.seed,
        n_hard_neg=args.n_hard_neg,
        max_sessions=args.max_sessions,
        loftr_approved=args.loftr_approved,
        config_ids=args.configs,
        loftr_max_edge=args.loftr_max_edge,
    )

    output_dir = Path(args.output_dir)
    scorer_factory = _make_real_scorer_factory(
        output_dir,
        device=args.device,
        disable_cudnn=args.disable_cudnn,
        max_keypoints=args.max_keypoints,
        max_sessions=args.max_sessions,
        allow_loftr_pilot=args.loftr_approved,
        loftr_max_edge=args.loftr_max_edge,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )

    result = run_pilot(
        manifest_df,
        splits_df,
        crop_manifest_df,
        scorer_factory=scorer_factory,
        output_dir=output_dir,
        config=pilot_config,
        embeddings=embeddings,
        embedding_index=embedding_index,
        calibrators=calibrators,
        fusion_weights=fusion_weights,
        resume=args.resume,
    )
    print(f"Pilot complete. Fingerprint: {result['fingerprint']}")
    print(f"Output: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
