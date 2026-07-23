#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Candidate/no-candidate decision report.

Applies the pre-registered decision gates to bootstrap results and writes
a structured report.  Never modifies production artifacts.

Decision gates (all must pass for candidate promotion)
-------------------------------------------------------
1. Point delta gate      : primary identity-macro MRR delta ≥ +0.02.
2. CI lower bound gate   : 95% CI lower bound > 0 (improvement is real).
3. Top-1 margin gate     : identity-macro top-1 delta ≥ -0.01.
4. Subset gate           : improvement consistent across temporal AND onboarding
                           strata (not just one stratum).
5. Runtime gate          : challenger system must not exceed baseline latency by
                           more than the registered budget (populated externally).
6. Coverage gate         : challenger channels must be available on ≥ 95% of
                           probe queries.

Additional metadata
-------------------
* Head gate: recorded as FAILED / NOT APPLICABLE (head channel was dropped).
* Covariate shift flag: raised if baseline query-RR drops > 0.05 vs OOF.
* Future-session protocol JSON: specifies how to acquire future sessions.
  - Acquired after registration hash is locked.
  - Complete sessions (no partial sessions).
  - Never previously loaded for scoring.
  - Power-calculated identity count (from registration power simulation).
  - One-touch protocol (no incremental scoring).
  - Covariate shift clause: flag if > 0.05 baseline RR drop vs OOF mean.
  - Head coverage clause: marked NOT APPLICABLE (head channel dropped).

Output files
------------
candidate_report.json       – full decision report.
future_session_protocol.json – protocol specification for future sessions.

This module NEVER writes to production artifact directories.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.paired_bootstrap import BootstrapResult, ContrastResult, CONTRAST_PRIMARY_TOP1
from pipeline.power_simulation import OPERATIONAL_DELTA_THRESHOLD
from pipeline.statistical_registration import FIXED_TOP1_MARGIN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gate thresholds (frozen at registration)
# ---------------------------------------------------------------------------

POINT_DELTA_THRESHOLD: float = OPERATIONAL_DELTA_THRESHOLD    # +0.02
TOP1_MARGIN_THRESHOLD: float = FIXED_TOP1_MARGIN               # -0.01
CI_LO_THRESHOLD: float = 0.0                                   # CI lower > 0
COVERAGE_THRESHOLD: float = 0.95                               # ≥ 95% coverage
COVARIATE_SHIFT_THRESHOLD: float = 0.05                        # > 0.05 RR drop

REPORT_FILENAME: str = "candidate_report.json"
FUTURE_PROTOCOL_FILENAME: str = "future_session_protocol.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    gate_name: str
    passed: bool
    value: Any
    threshold: Any
    note: str = ""


@dataclass
class DecisionReport:
    """Full candidate/no-candidate decision report."""
    registration_hash: str
    candidate_system: str
    baseline_system: str
    decision: str                   # "candidate" or "no_candidate"
    gates: List[GateResult]
    primary_contrast: ContrastResult
    bootstrap_n_replicates: int
    bootstrap_seed: int
    alpha: float
    # Per-stratum consistency check
    temporal_delta: Optional[float]
    onboarding_delta: Optional[float]
    # Metadata
    head_gate_status: str           # always "failed_not_applicable"
    covariate_shift_flag: bool
    underpowered_flag: bool
    report_generated_at: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gates"] = [asdict(g) for g in self.gates]
        d["primary_contrast"] = asdict(self.primary_contrast)
        return d


# ---------------------------------------------------------------------------
# Gate evaluators
# ---------------------------------------------------------------------------

def _gate_point_delta(
    delta: float,
    threshold: float = POINT_DELTA_THRESHOLD,
) -> GateResult:
    return GateResult(
        gate_name="point_delta",
        passed=delta >= threshold,
        value=round(delta, 6),
        threshold=threshold,
        note=(
            f"identity-macro MRR delta={delta:.4f} "
            f"{'≥' if delta >= threshold else '<'} {threshold}"
        ),
    )


def _gate_ci_lower(
    ci_lo: float,
    threshold: float = CI_LO_THRESHOLD,
) -> GateResult:
    return GateResult(
        gate_name="ci_lower_bound",
        passed=ci_lo > threshold,
        value=round(ci_lo, 6),
        threshold=threshold,
        note=f"95% CI lower bound={ci_lo:.4f} {'>' if ci_lo > threshold else '≤'} {threshold}",
    )


def _gate_top1_margin(
    top1_delta: Optional[float],
    threshold: float = TOP1_MARGIN_THRESHOLD,
) -> GateResult:
    if top1_delta is None:
        return GateResult(
            gate_name="top1_margin",
            passed=False,
            value=None,
            threshold=threshold,
            note="top-1 delta not available; gate failed.",
        )
    return GateResult(
        gate_name="top1_margin",
        passed=top1_delta >= threshold,
        value=round(top1_delta, 6),
        threshold=threshold,
        note=(
            f"identity-macro top-1 delta={top1_delta:.4f} "
            f"{'≥' if top1_delta >= threshold else '<'} {threshold}"
        ),
    )


def _gate_subset_consistency(
    temporal_delta: Optional[float],
    onboarding_delta: Optional[float],
) -> GateResult:
    """
    Consistency gate: improvement should be non-negative in both strata.
    A large benefit in one stratum masking a regression in the other is flagged.
    """
    if temporal_delta is None or onboarding_delta is None:
        return GateResult(
            gate_name="subset_consistency",
            passed=False,
            value=None,
            threshold=0.0,
            note="Per-stratum deltas not available.",
        )
    passed = temporal_delta >= 0.0 and onboarding_delta >= 0.0
    return GateResult(
        gate_name="subset_consistency",
        passed=passed,
        value={"temporal": round(temporal_delta, 6), "onboarding": round(onboarding_delta, 6)},
        threshold=0.0,
        note=(
            f"temporal delta={temporal_delta:.4f}, onboarding delta={onboarding_delta:.4f}; "
            f"{'both non-negative' if passed else 'regression in at least one stratum'}"
        ),
    )


def _gate_runtime(
    runtime_p95_seconds: Optional[float],
    max_seconds: float = 10.0,
) -> GateResult:
    """
    Runtime gate: p95 local rerank latency in seconds.
    """
    if runtime_p95_seconds is None:
        return GateResult(
            gate_name="runtime",
            passed=False,
            value=None,
            threshold=max_seconds,
            note=(
                "Runtime not measured in this evaluation. "
                "Operational gate cannot pass without evidence."
            ),
        )
    return GateResult(
        gate_name="runtime",
        passed=runtime_p95_seconds <= max_seconds,
        value=round(runtime_p95_seconds, 3),
        threshold=max_seconds,
        note=(
            f"p95 local rerank latency={runtime_p95_seconds:.3f}s "
            f"{'≤' if runtime_p95_seconds <= max_seconds else '>'} {max_seconds}s"
        ),
    )


def _gate_coverage(
    coverage: Optional[float],
    threshold: float = COVERAGE_THRESHOLD,
) -> GateResult:
    """
    Coverage gate: fraction of probe queries with all required channels present.
    """
    if coverage is None:
        return GateResult(
            gate_name="coverage",
            passed=False,
            value=None,
            threshold=threshold,
            note=(
                "Coverage not measured in this evaluation. "
                "Operational gate cannot pass without evidence."
            ),
        )
    return GateResult(
        gate_name="coverage",
        passed=coverage >= threshold,
        value=round(coverage, 4),
        threshold=threshold,
        note=(
            f"challenger channel coverage={coverage:.3%} "
            f"{'≥' if coverage >= threshold else '<'} {threshold:.0%}"
        ),
    )


def _gate_head(
) -> GateResult:
    """
    Head gate: always recorded as FAILED / NOT APPLICABLE.
    The head channel was dropped from the production ensemble.
    """
    return GateResult(
        gate_name="head_channel",
        passed=False,
        value="not_applicable",
        threshold="required_for_pass",
        note=(
            "Head channel was dropped from the production ensemble. "
            "Head gate is recorded as FAILED / NOT APPLICABLE and does not "
            "block candidate promotion for this evaluation."
        ),
    )


# ---------------------------------------------------------------------------
# Top-level report builder
# ---------------------------------------------------------------------------

def build_decision_report(
    bootstrap_result: BootstrapResult,
    registration: dict,
    *,
    temporal_delta: Optional[float] = None,
    onboarding_delta: Optional[float] = None,
    runtime_p95_seconds: Optional[float] = None,
    coverage: Optional[float] = None,
    covariate_shift_flag: bool = False,
    candidate_system: str = "selected_v1_plus_both_local",
    baseline_system: str = "selected_v1",
) -> DecisionReport:
    """
    Build the decision report from bootstrap results and registration.

    Parameters
    ----------
    bootstrap_result   : output of run_paired_bootstrap().
    registration       : loaded+verified registration document.
    temporal_delta     : observed MRR delta in temporal stratum only.
    onboarding_delta   : observed MRR delta in onboarding stratum only.
    runtime_p95_seconds: challenger p95 local rerank latency in seconds.
    coverage           : fraction of probes with all challenger channels present.
    covariate_shift_flag : True if baseline RR drops > 0.05 vs OOF.
    candidate_system   : name of the system being evaluated as challenger.
    baseline_system    : name of the baseline system.

    Returns
    -------
    DecisionReport
    """
    primary = bootstrap_result.primary
    alpha = bootstrap_result.alpha

    # Find top-1 secondary contrast
    top1_contrast: Optional[ContrastResult] = None
    for sc in bootstrap_result.secondaries:
        if sc.metric == "identity_macro_top1" and sc.system_b == candidate_system:
            top1_contrast = sc
            break

    top1_delta: Optional[float] = None
    if top1_contrast is not None:
        top1_delta = top1_contrast.point_delta
    else:
        # Fall back to system_metrics
        m_a = bootstrap_result.system_metrics.get(baseline_system, {})
        m_b = bootstrap_result.system_metrics.get(candidate_system, {})
        if "identity_macro_top1" in m_a and "identity_macro_top1" in m_b:
            top1_delta = m_b["identity_macro_top1"] - m_a["identity_macro_top1"]

    # --- Evaluate all gates ---
    gates = [
        _gate_point_delta(primary.point_delta),
        _gate_ci_lower(primary.ci_lo),
        _gate_top1_margin(top1_delta),
        _gate_subset_consistency(temporal_delta, onboarding_delta),
        _gate_runtime(runtime_p95_seconds),
        _gate_coverage(coverage),
        _gate_head(),
    ]

    # Decision: all mandatory gates must pass
    # Head gate is NOT mandatory (marked not applicable)
    mandatory_gates = [g for g in gates if g.gate_name != "head_channel"]
    all_passed = all(g.passed for g in mandatory_gates)
    decision = "candidate" if all_passed else "no_candidate"

    underpowered = registration.get("power_simulation", {}).get("underpowered", False)

    return DecisionReport(
        registration_hash=registration.get("registration_hash", ""),
        candidate_system=candidate_system,
        baseline_system=baseline_system,
        decision=decision,
        gates=gates,
        primary_contrast=primary,
        bootstrap_n_replicates=bootstrap_result.n_replicates,
        bootstrap_seed=bootstrap_result.seed,
        alpha=alpha,
        temporal_delta=temporal_delta,
        onboarding_delta=onboarding_delta,
        head_gate_status="failed_not_applicable",
        covariate_shift_flag=covariate_shift_flag,
        underpowered_flag=bool(underpowered),
        report_generated_at=datetime.now(tz=timezone.utc).isoformat(),
    )


def build_future_session_protocol(
    registration: dict,
    bootstrap_result: BootstrapResult,
    *,
    covariate_shift_flag: bool = False,
) -> dict:
    """
    Build the future-session confirmatory protocol specification.

    The future protocol is acquired AFTER the registration hash is locked,
    covers complete sessions only (no partial sessions), and uses the
    power-calculated identity count from the registration power simulation.

    Head coverage clause is marked NOT APPLICABLE (head channel dropped).
    """
    power_sim = registration.get("power_simulation", {})
    n_temporal_required = registration.get("n_temporal_ids", 42)
    n_onboarding_required = registration.get("n_onboarding_ids", 9)

    return {
        "protocol_version": "future-session-v1",
        "registration_hash": registration.get("registration_hash", ""),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "acquisition_rules": {
            "acquired_after_registration_lock": True,
            "complete_sessions_only": True,
            "partial_sessions_allowed": False,
            "never_previously_loaded_for_scoring": True,
            "one_touch_protocol": True,
        },
        "identity_counts": {
            "n_temporal_required": n_temporal_required,
            "n_onboarding_required": n_onboarding_required,
            "power_basis": power_sim.get("simulation_params", {}),
            "source": "registration_power_simulation",
        },
        "covariate_shift_clause": {
            "threshold": COVARIATE_SHIFT_THRESHOLD,
            "flag_raised": covariate_shift_flag,
            "description": (
                f"Flag is raised if the baseline (selected-v1) query-level RR "
                f"drops by more than {COVARIATE_SHIFT_THRESHOLD:.2f} relative to "
                f"the OOF mean across the future-session probe set."
            ),
        },
        "head_coverage_clause": {
            "status": "not_applicable",
            "reason": (
                "Head channel was dropped from the production ensemble. "
                "Head coverage is not applicable for future session evaluation."
            ),
        },
        "endpoints": {
            "primary": registration.get("primary_endpoint", "identity_macro_mrr"),
            "secondaries": registration.get("secondary_endpoints", []),
            "multiplicity_correction": registration.get("multiplicity_correction", "holm_max_t"),
        },
        "statistical_design": registration.get("statistical_design", {}),
        "mde_at_80_power": power_sim.get("mde_80", None),
        "underpowered_flag": power_sim.get("underpowered", False),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_decision_report(
    report: DecisionReport,
    output_dir: Path,
) -> Path:
    """Write the decision report JSON and return the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / REPORT_FILENAME
    with open(path, "w") as fh:
        json.dump(report.to_dict(), fh, indent=2, sort_keys=True)
    logger.info("Decision report written: %s (decision=%s)", path, report.decision)
    return path


def write_future_session_protocol(
    protocol: dict,
    output_dir: Path,
) -> Path:
    """Write the future-session protocol JSON and return the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / FUTURE_PROTOCOL_FILENAME
    with open(path, "w") as fh:
        json.dump(protocol, fh, indent=2, sort_keys=True)
    logger.info("Future session protocol written: %s", path)
    return path
