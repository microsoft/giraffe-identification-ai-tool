#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Paired protocol-stratified identity-cluster bootstrap for fixed-probe
statistical inference.

Design
------
* Cluster unit: identity (not individual query row).
* Stratification: temporal identities (42) and onboarding identities (9)
  resampled separately.
* Each bootstrap replicate:
    - Draw n_temporal IDs with replacement from temporal IDs.
    - Draw n_onboarding IDs with replacement from onboarding IDs.
    - For each selected ID, include ALL of its probe queries in BOTH systems.
    - Gallery is fixed (not resampled).
* Primary contrast:
    - selected_v1 vs selected_v1_plus_both_local (identity-macro MRR).
* Secondary contrasts (Holm p-values + max-|T| simultaneous intervals):
    1. selected_v1 vs selected_v1_plus_body_local (identity-macro MRR)
    2. selected_v1 vs selected_v1_plus_ear_local  (identity-macro MRR)
    3. selected_v1 vs selected_v1_frozen_ear       (identity-macro MRR)
    4. selected_v1 vs selected_v1_plus_both_local  (identity-macro top-1)
* Sign-flip randomization by identity for null distribution.
* Seed: 20260714, 10k outer replicates.

References
----------
Westfall & Young (1993) Resampling-based multiple testing.
Holm (1979) A simple sequentially rejective multiple test procedure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pipeline.eval_metrics import (
    reciprocal_rank,
    top_k_hit,
    identity_macro_mrr,
    identity_macro_top1,
    query_weighted_mrr,
    query_weighted_top1,
    query_weighted_top5,
    aggregate_per_identity_rrs,
    aggregate_per_identity_top_k,
)
from pipeline.power_simulation import (
    SIMULATION_SEED,
    N_REPLICATES,
    ALPHA,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contrast definitions
# ---------------------------------------------------------------------------

CONTRAST_PRIMARY = "primary_mrr"
CONTRAST_BODY_LOCAL_MRR = "secondary_body_local_mrr"
CONTRAST_EAR_LOCAL_MRR = "secondary_ear_local_mrr"
CONTRAST_FROZEN_EAR_MRR = "secondary_frozen_ear_mrr"
CONTRAST_PRIMARY_TOP1 = "secondary_primary_top1"

SECONDARY_CONTRASTS = [
    CONTRAST_BODY_LOCAL_MRR,
    CONTRAST_EAR_LOCAL_MRR,
    CONTRAST_FROZEN_EAR_MRR,
    CONTRAST_PRIMARY_TOP1,
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ContrastResult:
    """Result for a single contrast (system A vs system B on one metric)."""
    contrast_name: str
    system_a: str
    system_b: str
    metric: str
    point_delta: float              # observed delta = metric_B - metric_A
    ci_lo: float                    # 95% bootstrap CI lower bound
    ci_hi: float                    # 95% bootstrap CI upper bound
    p_value: float                  # sign-flip p-value (two-sided)
    p_value_holm: float             # Holm-adjusted p-value
    simultaneous_ci_lo: float       # max-|T| simultaneous CI lower bound
    simultaneous_ci_hi: float       # max-|T| simultaneous CI upper bound
    reject_h0: bool                 # True if p_value_holm < alpha


@dataclass
class BootstrapResult:
    """Full bootstrap result for the paired evaluation."""
    primary: ContrastResult
    secondaries: List[ContrastResult]
    alpha: float
    n_replicates: int
    seed: int
    n_temporal_ids: int
    n_onboarding_ids: int
    # Per-stratum observed metrics for each system
    system_metrics: Dict[str, dict]     # system_name → metric dict


# ---------------------------------------------------------------------------
# Query record (lightweight)
# ---------------------------------------------------------------------------

@dataclass
class QueryRecord:
    """Single query result for one system, carrying only what bootstrap needs."""
    query_image_id: str
    truth_individual_id: Optional[str]
    probe_type: str                  # "temporal" or "onboarding"
    individual_id_key: str           # clustering key (= truth_individual_id if known)
    system_name: str
    ranked_ids: List[str]


def build_query_records_from_dataframe(
    rankings_df,
    system_name: str,
    probe_type_col: str = "probe_type",
    truth_col: str = "truth_individual_id",
    query_col: str = "query_image_id",
    rank_col: str = "rank",
    cand_col: str = "candidate_individual_id",
) -> List[QueryRecord]:
    """
    Build per-query QueryRecord list from a probe rankings DataFrame.

    Each query must appear with one row per rank.  Rows are sorted by rank
    and the ranked list is reconstructed.
    """
    import pandas as pd
    records: List[QueryRecord] = []
    df = rankings_df[rankings_df["system_name"] == system_name].copy()
    df = df.sort_values([query_col, rank_col])

    for qid, grp in df.groupby(query_col, sort=False):
        truth = grp[truth_col].iloc[0]
        truth = str(truth) if not isinstance(truth, float) else None
        pt = grp[probe_type_col].iloc[0] if probe_type_col in grp.columns else "temporal"
        ranked = grp[cand_col].astype(str).tolist()
        records.append(QueryRecord(
            query_image_id=str(qid),
            truth_individual_id=truth,
            probe_type=str(pt),
            individual_id_key=truth or str(qid),
            system_name=system_name,
            ranked_ids=ranked,
        ))
    return records


# ---------------------------------------------------------------------------
# Per-identity metric computation
# ---------------------------------------------------------------------------

def _per_identity_mrr_from_records(
    records: List[QueryRecord],
    identity_id_subset: Optional[List[str]] = None,
) -> Dict[str, List[float]]:
    """Build per-identity RR dict from query records, optionally filtered."""
    per_id: Dict[str, List[float]] = {}
    for rec in records:
        truth = rec.truth_individual_id
        if truth is None:
            continue
        if identity_id_subset is not None and truth not in identity_id_subset:
            continue
        rr = reciprocal_rank(rec.ranked_ids, truth)
        per_id.setdefault(truth, []).append(rr)
    return per_id


def _per_identity_top1_from_records(
    records: List[QueryRecord],
    identity_id_subset: Optional[List[str]] = None,
) -> Dict[str, List[float]]:
    per_id: Dict[str, List[float]] = {}
    for rec in records:
        truth = rec.truth_individual_id
        if truth is None:
            continue
        if identity_id_subset is not None and truth not in identity_id_subset:
            continue
        hit = top_k_hit(rec.ranked_ids, truth, k=1)
        per_id.setdefault(truth, []).append(hit)
    return per_id


def _per_identity_top5_from_records(
    records: List[QueryRecord],
    identity_id_subset: Optional[List[str]] = None,
) -> Dict[str, List[float]]:
    per_id: Dict[str, List[float]] = {}
    for rec in records:
        truth = rec.truth_individual_id
        if truth is None:
            continue
        if identity_id_subset is not None and truth not in identity_id_subset:
            continue
        hit = top_k_hit(rec.ranked_ids, truth, k=5)
        per_id.setdefault(truth, []).append(hit)
    return per_id


def _paired_identity_deltas(
    records_a: List[QueryRecord],
    records_b: List[QueryRecord],
    metric: str,
    temporal_ids: List[str],
    onboarding_ids: List[str],
) -> np.ndarray:
    """
    Compute per-identity paired differences for *metric*.

    Returns array of shape (n_identities,) where entry i =
    metric_B(identity_i) - metric_A(identity_i).
    """
    all_ids = list(temporal_ids) + list(onboarding_ids)

    def _id_metric(records: List[QueryRecord], iid: str) -> float:
        if metric == "identity_macro_mrr":
            per_id = _per_identity_mrr_from_records(records, [iid])
            return identity_macro_mrr(per_id)
        if metric == "identity_macro_top1":
            per_id = _per_identity_top1_from_records(records, [iid])
            return identity_macro_top1(per_id)
        raise ValueError(f"Unknown metric: {metric!r}")

    deltas = []
    for iid in all_ids:
        ma = _id_metric(records_a, iid)
        mb = _id_metric(records_b, iid)
        deltas.append(mb - ma)

    return np.array(deltas)


# ---------------------------------------------------------------------------
# Bootstrap resampling
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sign-flip randomization p-value
# ---------------------------------------------------------------------------

def _sign_flip_pvalue(
    diffs: np.ndarray,
    rng: np.random.Generator,
    n_flip: int = 9999,
) -> float:
    """Two-sided sign-flip p-value for the mean of per-identity differences."""
    obs = float(np.mean(diffs))
    n = len(diffs)
    if n == 0:
        return 1.0
    # Sign-flip null distribution (by identity)
    signs = rng.choice([-1.0, 1.0], size=(n_flip, n))
    null_stats = np.mean(signs * diffs[np.newaxis, :], axis=1)
    return float(np.mean(np.abs(null_stats) >= abs(obs))) + 1.0 / (n_flip + 1)


# ---------------------------------------------------------------------------
# Holm correction
# ---------------------------------------------------------------------------

def holm_correction(p_values: List[float]) -> List[float]:
    """
    Holm-Bonferroni step-down p-value adjustment.

    Returns adjusted p-values in the same order as the input.
    """
    n = len(p_values)
    if n == 0:
        return []
    # Sorted order
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    max_adj = 0.0
    for step, idx in enumerate(order):
        correction = n - step
        adj = min(1.0, p_values[idx] * correction)
        adj = max(adj, max_adj)  # enforce monotonicity
        max_adj = adj
        adjusted[idx] = adj
    return adjusted


# ---------------------------------------------------------------------------
# max-|T| simultaneous confidence intervals
# ---------------------------------------------------------------------------

def _max_t_simultaneous_intervals(
    contrast_deltas: Dict[str, np.ndarray],
    alpha: float = ALPHA,
    observed_deltas: Optional[Dict[str, float]] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute max-|T| simultaneous confidence intervals.

    For each bootstrap replicate, the standardized delta (T_j) is computed
    for each contrast j.  The max-|T| critical value is the (1-alpha)
    quantile of max_j(|T_j|).  The simultaneous CI for contrast j is:
      [delta_j - crit * se_j, delta_j + crit * se_j]

    Parameters
    ----------
    contrast_deltas : {contrast_name: np.array of n_boot bootstrap delta values}
    alpha           : family-wise error rate.

    Returns
    -------
    {contrast_name: (ci_lo, ci_hi)}
    """
    if not contrast_deltas:
        return {}

    keys = list(contrast_deltas.keys())
    n_boot = len(next(iter(contrast_deltas.values())))
    if n_boot == 0:
        return {k: (float("nan"), float("nan")) for k in keys}

    # Standardise each contrast's bootstrap distribution
    boot_array = np.stack([contrast_deltas[k] for k in keys], axis=1)  # (n_boot, n_contrasts)
    obs_means = boot_array.mean(axis=0)   # (n_contrasts,)
    obs_ses = boot_array.std(axis=0, ddof=1)   # (n_contrasts,)

    # Avoid division by zero
    safe_ses = np.where(obs_ses > 0, obs_ses, 1.0)
    t_stats = np.abs((boot_array - obs_means[np.newaxis, :]) / safe_ses[np.newaxis, :])  # (n_boot, n_contrasts)
    max_t = t_stats.max(axis=1)   # (n_boot,)

    crit = float(np.quantile(max_t, 1.0 - alpha))

    result: Dict[str, Tuple[float, float]] = {}
    for j, key in enumerate(keys):
        d_hat = float(
            observed_deltas[key]
            if observed_deltas is not None and key in observed_deltas
            else obs_means[j]
        )
        se = float(obs_ses[j])
        result[key] = (d_hat - crit * se, d_hat + crit * se)

    return result


# ---------------------------------------------------------------------------
# Main bootstrap runner
# ---------------------------------------------------------------------------

def run_paired_bootstrap(
    records_by_system: Dict[str, List[QueryRecord]],
    temporal_ids: List[str],
    onboarding_ids: List[str],
    *,
    system_a: str = "selected_v1",
    system_b_primary: str = "selected_v1_plus_both_local",
    system_b_body: str = "selected_v1_plus_body_local",
    system_b_ear: str = "selected_v1_plus_ear_local",
    system_b_frozen: str = "selected_v1_frozen_ear",
    n_replicates: int = N_REPLICATES,
    seed: int = SIMULATION_SEED,
    alpha: float = ALPHA,
) -> BootstrapResult:
    """
    Run the paired protocol-stratified identity-cluster bootstrap.

    Parameters
    ----------
    records_by_system : {system_name: list of QueryRecord}.
    temporal_ids      : list of temporal identity IDs (frozen, 42).
    onboarding_ids    : list of onboarding identity IDs (frozen, 9).
    system_a          : baseline system name.
    system_b_*        : challenger system names for each contrast.
    n_replicates      : bootstrap replicates (10k).
    seed              : fixed seed 20260714.
    alpha             : type-I error rate (0.05).

    Returns
    -------
    BootstrapResult
    """
    rng = np.random.default_rng(seed)

    def _records(name: str) -> List[QueryRecord]:
        recs = records_by_system.get(name, [])
        if not recs:
            logger.warning("No records for system '%s'.", name)
        return recs

    recs_a = _records(system_a)

    # Contrasts: (name, system_b, metric)
    contrasts = [
        (CONTRAST_PRIMARY, system_b_primary, "identity_macro_mrr"),
        (CONTRAST_BODY_LOCAL_MRR, system_b_body, "identity_macro_mrr"),
        (CONTRAST_EAR_LOCAL_MRR, system_b_ear, "identity_macro_mrr"),
        (CONTRAST_FROZEN_EAR_MRR, system_b_frozen, "identity_macro_mrr"),
        (CONTRAST_PRIMARY_TOP1, system_b_primary, "identity_macro_top1"),
    ]

    # -----------------------------------------------------------------
    # Observed metrics
    # -----------------------------------------------------------------
    system_metrics: Dict[str, dict] = {}
    for sys_name in records_by_system:
        recs = records_by_system[sys_name]
        per_id_rr = _per_identity_mrr_from_records(recs)
        per_id_t1 = _per_identity_top1_from_records(recs)
        per_id_t5 = _per_identity_top5_from_records(recs)
        all_rrs = [rr for rrs in per_id_rr.values() for rr in rrs]
        all_t1 = [h for hs in per_id_t1.values() for h in hs]
        all_t5 = [h for hs in per_id_t5.values() for h in hs]
        system_metrics[sys_name] = {
            "identity_macro_mrr": identity_macro_mrr(per_id_rr),
            "identity_macro_top1": identity_macro_top1(per_id_t1),
            "query_weighted_mrr": query_weighted_mrr(all_rrs),
            "query_weighted_top1": query_weighted_top1(all_t1),
            "query_weighted_top5": query_weighted_top5(all_t5),
            "n_queries": len(recs),
        }

    # -----------------------------------------------------------------
    # Bootstrap distributions: run n_replicates, collect deltas per contrast
    # -----------------------------------------------------------------
    logger.info("Running %d bootstrap replicates ...", n_replicates)
    boot_deltas: Dict[str, List[float]] = {cn: [] for cn, _, _ in contrasts}

    for rep in range(n_replicates):
        boot_temporal = list(rng.choice(temporal_ids, size=len(temporal_ids), replace=True))
        boot_onboarding = list(rng.choice(onboarding_ids, size=len(onboarding_ids), replace=True))
        boot_id_list = boot_temporal + boot_onboarding

        for contrast_name, sys_b_name, metric in contrasts:
            recs_b = _records(sys_b_name)

            def _boot_metric(recs: List[QueryRecord]) -> float:
                cluster_means: List[float] = []
                for iid in boot_id_list:
                    if metric == "identity_macro_mrr":
                        vals = [
                            reciprocal_rank(r.ranked_ids, r.truth_individual_id)
                            for r in recs
                            if r.truth_individual_id == iid
                        ]
                    else:  # identity_macro_top1
                        vals = [
                            top_k_hit(r.ranked_ids, r.truth_individual_id, k=1)
                            for r in recs
                            if r.truth_individual_id == iid
                        ]
                    if vals:
                        cluster_means.append(float(np.mean(vals)))
                return float(np.mean(cluster_means)) if cluster_means else 0.0

            ma = _boot_metric(recs_a)
            mb = _boot_metric(recs_b)
            boot_deltas[contrast_name].append(mb - ma)

    logger.info("Bootstrap complete.")

    # -----------------------------------------------------------------
    # Per-contrast confidence intervals (percentile)
    # -----------------------------------------------------------------
    ci_map: Dict[str, Tuple[float, float]] = {}
    for contrast_name, _, _ in contrasts:
        bd = np.array(boot_deltas[contrast_name])
        lo = float(np.percentile(bd, 100 * alpha / 2))
        hi = float(np.percentile(bd, 100 * (1 - alpha / 2)))
        ci_map[contrast_name] = (lo, hi)

    # -----------------------------------------------------------------
    # Sign-flip p-values (by identity, using observed per-identity diffs)
    # -----------------------------------------------------------------
    p_values_raw: Dict[str, float] = {}
    for contrast_name, sys_b_name, metric in contrasts:
        recs_b = _records(sys_b_name)
        diffs = _paired_identity_deltas(
            recs_a, recs_b, metric, temporal_ids, onboarding_ids
        )
        p_values_raw[contrast_name] = _sign_flip_pvalue(diffs, rng)

    # -----------------------------------------------------------------
    # Holm correction (over secondary contrasts only; primary is separate)
    # -----------------------------------------------------------------
    secondary_names = [cn for cn, _, _ in contrasts if cn != CONTRAST_PRIMARY]
    secondary_raw_p = [p_values_raw[cn] for cn in secondary_names]
    secondary_adjusted = holm_correction(secondary_raw_p)
    holm_map: Dict[str, float] = dict(zip(secondary_names, secondary_adjusted))
    holm_map[CONTRAST_PRIMARY] = p_values_raw[CONTRAST_PRIMARY]  # primary not adjusted

    # -----------------------------------------------------------------
    # max-|T| simultaneous intervals (secondary contrasts only)
    # -----------------------------------------------------------------
    secondary_boot_deltas = {
        cn: np.array(boot_deltas[cn]) for cn in secondary_names
    }
    secondary_observed = {
        contrast_name: (
            system_metrics.get(system_b, {}).get(metric, 0.0)
            - system_metrics.get(system_a, {}).get(metric, 0.0)
        )
        for contrast_name, system_b, metric in contrasts
        if contrast_name in secondary_names
    }
    simultaneous = _max_t_simultaneous_intervals(
        secondary_boot_deltas,
        alpha=alpha,
        observed_deltas=secondary_observed,
    )
    # Primary gets its own percentile CI (not simultaneous)
    simultaneous[CONTRAST_PRIMARY] = ci_map[CONTRAST_PRIMARY]

    # -----------------------------------------------------------------
    # Build ContrastResult for each contrast
    # -----------------------------------------------------------------
    def _build_result(contrast_name: str, sys_b: str, metric: str) -> ContrastResult:
        obs_a = system_metrics.get(system_a, {}).get(metric, 0.0)
        obs_b = system_metrics.get(sys_b, {}).get(metric, 0.0)
        point_delta = obs_b - obs_a
        ci_lo, ci_hi = ci_map[contrast_name]
        sim_lo, sim_hi = simultaneous.get(contrast_name, (ci_lo, ci_hi))
        raw_p = p_values_raw[contrast_name]
        adj_p = holm_map[contrast_name]
        return ContrastResult(
            contrast_name=contrast_name,
            system_a=system_a,
            system_b=sys_b,
            metric=metric,
            point_delta=round(float(point_delta), 6),
            ci_lo=round(float(ci_lo), 6),
            ci_hi=round(float(ci_hi), 6),
            p_value=round(float(raw_p), 6),
            p_value_holm=round(float(adj_p), 6),
            simultaneous_ci_lo=round(float(sim_lo), 6),
            simultaneous_ci_hi=round(float(sim_hi), 6),
            reject_h0=adj_p < alpha,
        )

    primary_result = _build_result(CONTRAST_PRIMARY, system_b_primary, "identity_macro_mrr")
    secondary_results = [
        _build_result(CONTRAST_BODY_LOCAL_MRR, system_b_body, "identity_macro_mrr"),
        _build_result(CONTRAST_EAR_LOCAL_MRR, system_b_ear, "identity_macro_mrr"),
        _build_result(CONTRAST_FROZEN_EAR_MRR, system_b_frozen, "identity_macro_mrr"),
        _build_result(CONTRAST_PRIMARY_TOP1, system_b_primary, "identity_macro_top1"),
    ]

    return BootstrapResult(
        primary=primary_result,
        secondaries=secondary_results,
        alpha=alpha,
        n_replicates=n_replicates,
        seed=seed,
        n_temporal_ids=len(temporal_ids),
        n_onboarding_ids=len(onboarding_ids),
        system_metrics=system_metrics,
    )
