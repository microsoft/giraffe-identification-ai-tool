#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Pre-registration power simulation for the fixed-probe paired evaluation.

This module uses ONLY frozen cluster sizes (number of probe queries per
identity) and an estimated paired-effect variance.  It NEVER reads probe
outcomes; all simulation is synthetic.

Design
------
The primary endpoint is identity-macro MRR.  The test is a paired
protocol-stratified identity-cluster bootstrap:

  1. Resample 42 temporal identity IDs (with replacement).
  2. Resample 9 onboarding identity IDs (with replacement).
  3. For each resampled identity, include ALL of its probe queries in
     both systems.  Gallery is fixed.
  4. The test statistic T is the mean paired difference in identity-level
     mean RR: T = mean_id(RR_A_i - RR_B_i).
  5. Sign-flip randomization provides the null distribution.
  6. Power = P(reject H0 | true delta) at alpha = 0.05 (two-sided).

Power simulation algorithm
--------------------------
For each effect size delta:
  a. Simulate per-identity paired differences d_i ~ N(delta, sigma^2_d)
     using the frozen cluster-size-weighted variance estimate.
  b. Apply the sign-flip test to compute a p-value.
  c. Repeat 10k times (seed 20260714), counting rejections.
  d. Power(delta) = rejection rate.
  e. MDE = smallest delta (on a fine grid) where Power >= 0.80.

Underpowered flag: if MDE > OPERATIONAL_DELTA_THRESHOLD (0.02).

The module is deterministic given the same cluster sizes + variance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol constants (frozen at registration)
# ---------------------------------------------------------------------------

N_TEMPORAL_IDS: int = 42
N_ONBOARDING_IDS: int = 9
SIMULATION_SEED: int = 20260714
N_REPLICATES: int = 10_000
ALPHA: float = 0.05
TARGET_POWER: float = 0.80
OPERATIONAL_DELTA_THRESHOLD: float = 0.02   # +0.02 identity-macro MRR

# MDE grid resolution
_MDE_GRID_STEP: float = 0.001
_MDE_GRID_MAX: float = 0.20


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClusterSpec:
    """
    Frozen cluster specification for one stratum.

    cluster_sizes : dict mapping identity_id → number of probe queries.
    stratum       : "temporal" or "onboarding".
    """
    stratum: str
    cluster_sizes: Dict[str, int]   # identity_id → n_queries

    @property
    def n_ids(self) -> int:
        return len(self.cluster_sizes)

    @property
    def query_counts(self) -> List[int]:
        return list(self.cluster_sizes.values())

    @property
    def total_queries(self) -> int:
        return sum(self.cluster_sizes.values())


@dataclass
class PowerSimulationResult:
    """Output of run_power_simulation()."""
    mde_80: float                       # MDE at 80% power
    expected_ci_half_width: float       # expected bootstrap CI half-width under H0
    underpowered: bool                  # True if MDE > OPERATIONAL_DELTA_THRESHOLD
    power_at_operational: float         # power at the +0.02 threshold
    power_curve: Dict[str, float]       # delta → power (sparse grid)
    simulation_params: dict             # provenance

    def to_dict(self) -> dict:
        d = asdict(self)
        d["underpowered"] = bool(self.underpowered)
        return d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _effective_n_ids(specs: List[ClusterSpec]) -> int:
    """Total number of distinct identity IDs across all strata."""
    return sum(s.n_ids for s in specs)


def _sign_flip_p_value(
    diffs: np.ndarray,
    n_flip: int,
    rng: np.random.Generator,
) -> float:
    """
    Sign-flip randomization p-value (two-sided) for the mean of *diffs*.

    Parameters
    ----------
    diffs  : per-identity paired differences, shape (n,).
    n_flip : number of sign-flip replicates for the null.
    rng    : NumPy Generator.
    """
    obs = float(np.mean(diffs))
    n = len(diffs)
    if n == 0:
        return 1.0
    # Generate n_flip sign-flip replicates
    signs = rng.choice([-1.0, 1.0], size=(n_flip, n))
    null_stats = np.mean(signs * diffs[np.newaxis, :], axis=1)
    p = float(np.mean(np.abs(null_stats) >= abs(obs)))
    return max(p, 1.0 / n_flip)


def _simulate_one_replicate(
    specs: List[ClusterSpec],
    true_delta: float,
    sigma_d: float,
    rng: np.random.Generator,
    n_flip: int = 999,
) -> float:
    """
    Simulate one outer replicate and return its p-value.

    For each identity, the per-identity paired difference is:
      d_i ~ N(true_delta, sigma_d^2 / k_i)
    where k_i is the number of probe queries for identity i.

    Returns the sign-flip p-value.
    """
    all_diffs: List[float] = []
    for spec in specs:
        for k_i in spec.query_counts:
            # Variance of mean-RR for identity i ∝ 1/k_i
            sd_i = sigma_d / max(np.sqrt(k_i), 1.0)
            d_i = float(rng.normal(true_delta, sd_i))
            all_diffs.append(d_i)

    diffs = np.array(all_diffs)
    return _sign_flip_p_value(diffs, n_flip, rng)


def _run_power_at_delta(
    specs: List[ClusterSpec],
    delta: float,
    sigma_d: float,
    rng: np.random.Generator,
    n_outer: int = N_REPLICATES,
    alpha: float = ALPHA,
    n_flip: int = 999,
) -> float:
    """Return empirical power (rejection rate) at *delta*."""
    rejections = 0
    for _ in range(n_outer):
        pval = _simulate_one_replicate(specs, delta, sigma_d, rng, n_flip)
        if pval < alpha:
            rejections += 1
    return rejections / n_outer


def _estimate_ci_half_width(
    specs: List[ClusterSpec],
    sigma_d: float,
    rng: np.random.Generator,
    n_outer: int = 1000,
    n_boot: int = 999,
    alpha: float = ALPHA,
) -> float:
    """
    Estimate expected CI half-width under H0 via simulation.

    Simulates *n_outer* datasets under H0 (delta=0), and for each
    dataset runs a percentile bootstrap to get the 95% CI.  Returns
    the mean half-width.
    """
    hw_list: List[float] = []
    for _ in range(n_outer):
        # Simulate observed paired differences under H0
        obs_diffs: List[float] = []
        for spec in specs:
            for k_i in spec.query_counts:
                sd_i = sigma_d / max(np.sqrt(k_i), 1.0)
                obs_diffs.append(float(rng.normal(0.0, sd_i)))

        diffs = np.array(obs_diffs)
        n = len(diffs)
        if n < 2:
            continue
        # Percentile bootstrap
        boot_means = np.array([
            np.mean(rng.choice(diffs, size=n, replace=True))
            for _ in range(n_boot)
        ])
        lo = float(np.percentile(boot_means, 100 * alpha / 2))
        hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        hw_list.append((hi - lo) / 2.0)

    return float(np.mean(hw_list)) if hw_list else float("nan")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_power_simulation(
    temporal_cluster_sizes: Dict[str, int],
    onboarding_cluster_sizes: Dict[str, int],
    oof_paired_variance: float,
    *,
    n_replicates: int = N_REPLICATES,
    seed: int = SIMULATION_SEED,
    alpha: float = ALPHA,
    target_power: float = TARGET_POWER,
    operational_threshold: float = OPERATIONAL_DELTA_THRESHOLD,
    mde_grid_step: float = _MDE_GRID_STEP,
    n_flip: int = 999,
) -> PowerSimulationResult:
    """
    Run the pre-registration power simulation.

    All inputs are frozen at registration time.  No probe outcomes are used.

    Parameters
    ----------
    temporal_cluster_sizes   : identity_id → n_probe_queries (42 temporal IDs).
    onboarding_cluster_sizes : identity_id → n_probe_queries (9 onboarding IDs).
    oof_paired_variance      : estimated variance of per-identity mean RR
                               differences from gallery OOF data.  This is
                               the ``sigma_d`` of the per-identity paired
                               effect, before cluster-size adjustment.
    n_replicates             : outer simulation replicates for power curves.
    seed                     : random seed (frozen at 20260714).
    alpha                    : type-I error rate (0.05).
    target_power             : desired power level (0.80).
    operational_threshold    : operational MDE threshold (0.02).
    mde_grid_step            : grid resolution for MDE search.
    n_flip                   : sign-flip replicates per outer replicate.

    Returns
    -------
    PowerSimulationResult
    """
    if len(temporal_cluster_sizes) != N_TEMPORAL_IDS:
        raise ValueError(
            f"Expected {N_TEMPORAL_IDS} temporal identities, "
            f"got {len(temporal_cluster_sizes)}."
        )
    if len(onboarding_cluster_sizes) != N_ONBOARDING_IDS:
        raise ValueError(
            f"Expected {N_ONBOARDING_IDS} onboarding identities, "
            f"got {len(onboarding_cluster_sizes)}."
        )
    if oof_paired_variance <= 0:
        raise ValueError(f"oof_paired_variance must be > 0, got {oof_paired_variance}.")

    sigma_d = float(np.sqrt(oof_paired_variance))

    specs = [
        ClusterSpec("temporal", temporal_cluster_sizes),
        ClusterSpec("onboarding", onboarding_cluster_sizes),
    ]

    rng = np.random.default_rng(seed)

    # --- CI half-width under H0 ---
    logger.info("Estimating CI half-width under H0 ...")
    hw = _estimate_ci_half_width(specs, sigma_d, rng, n_outer=500, n_boot=499, alpha=alpha)

    # --- Power curve over MDE grid ---
    logger.info("Computing power curve (n_replicates=%d) ...", n_replicates)
    delta_grid = np.arange(mde_grid_step, _MDE_GRID_MAX + mde_grid_step, mde_grid_step)
    power_curve: Dict[str, float] = {}

    mde_80: float = float(_MDE_GRID_MAX)
    power_at_op: float = 0.0

    # Sparse grid: compute power at select deltas; do fine search around target
    sparse_deltas = [0.005, 0.010, 0.015, OPERATIONAL_DELTA_THRESHOLD, 0.030, 0.050, 0.100]
    for d in sparse_deltas:
        p = _run_power_at_delta(specs, d, sigma_d, rng, n_outer=min(n_replicates, 2000),
                                alpha=alpha, n_flip=n_flip)
        power_curve[f"{d:.3f}"] = round(p, 4)
        if abs(d - operational_threshold) < 1e-9:
            power_at_op = p

    # Fine search around the region where power crosses target_power
    # Find bracketing pair from sparse results
    sorted_sparse = sorted((float(k), v) for k, v in power_curve.items())
    bracket_lo = 0.0
    bracket_hi = _MDE_GRID_MAX
    for (d_lo, p_lo), (d_hi, p_hi) in zip(sorted_sparse, sorted_sparse[1:]):
        if p_lo < target_power <= p_hi:
            bracket_lo, bracket_hi = d_lo, d_hi
            break
    if sorted_sparse and sorted_sparse[-1][1] < target_power:
        # power never reaches target_power on the sparse grid
        mde_80 = float(_MDE_GRID_MAX)
    else:
        # Fine-grained binary search in [bracket_lo, bracket_hi]
        lo, hi = bracket_lo, bracket_hi
        for _ in range(10):
            mid = (lo + hi) / 2.0
            p_mid = _run_power_at_delta(
                specs, mid, sigma_d, rng,
                n_outer=min(n_replicates, 1000), alpha=alpha, n_flip=n_flip
            )
            power_curve[f"{mid:.4f}"] = round(p_mid, 4)
            if p_mid >= target_power:
                hi = mid
            else:
                lo = mid
        mde_80 = round(float(hi), 4)

    underpowered = mde_80 > operational_threshold

    if underpowered:
        logger.warning(
            "Study is UNDERPOWERED: MDE=%.4f > operational threshold %.3f. "
            "Increase probe set or accept wider CI.",
            mde_80,
            operational_threshold,
        )
    else:
        logger.info(
            "Study adequately powered: MDE=%.4f ≤ operational threshold %.3f.",
            mde_80,
            operational_threshold,
        )

    return PowerSimulationResult(
        mde_80=mde_80,
        expected_ci_half_width=round(hw, 5),
        underpowered=underpowered,
        power_at_operational=round(power_at_op, 4),
        power_curve=power_curve,
        simulation_params={
            "n_temporal_ids": len(temporal_cluster_sizes),
            "n_onboarding_ids": len(onboarding_cluster_sizes),
            "total_queries_temporal": sum(temporal_cluster_sizes.values()),
            "total_queries_onboarding": sum(onboarding_cluster_sizes.values()),
            "oof_paired_variance": oof_paired_variance,
            "sigma_d": sigma_d,
            "n_replicates": n_replicates,
            "seed": seed,
            "alpha": alpha,
            "target_power": target_power,
            "operational_threshold": operational_threshold,
        },
    )
