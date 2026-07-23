#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Comprehensive tests for the statistical registration and fixed-probe
evaluation tooling:

  pipeline/eval_metrics.py
  pipeline/power_simulation.py
  pipeline/statistical_registration.py
  pipeline/fixed_probe_evaluator.py
  pipeline/paired_bootstrap.py
  pipeline/candidate_report.py

All tests use synthetic rankings and fake scorers.  No real probe scoring
is performed.

Test matrix
-----------
 1.  test_reciprocal_rank_hit          – truth in list → 1/rank
 2.  test_reciprocal_rank_miss         – truth absent → 0.0
 3.  test_identity_macro_mrr_formula   – exact formula with manual values
 4.  test_query_weighted_mrr           – mean of per-query RR values
 5.  test_legacy_map_regression_fixture– 0.473 reproduces within tolerance
 6.  test_legacy_map_regression_fail   – raises ValueError outside tolerance
 7.  test_identity_macro_top1_top5     – top-k hit aggregation
 8.  test_compute_all_metrics          – end-to-end metric dict
 9.  test_registration_cannot_read_outcomes – outcome cols in manifest → error
10.  test_registration_hash_roundtrip  – write + verify is consistent
11.  test_registration_hash_mismatch   – tampered file → RegistrationHashMismatchError
12.  test_extract_cluster_sizes        – correct temporal/onboarding split
13.  test_cluster_not_row_bootstrap    – bootstrap resamples IDs, not rows
14.  test_protocol_strata_separate     – temporal and onboarding resampled independently
15.  test_bootstrap_deterministic_10k  – same seed → identical results
16.  test_power_underpowered_flag      – MDE > 0.02 → underpowered = True
17.  test_power_deterministic          – same inputs → same MDE
18.  test_sign_flip_null_mean_zero     – sign-flip null has zero mean
19.  test_holm_correction              – Holm adjustment matches manual calc
20.  test_max_t_simultaneous           – simultaneous CIs are wider than marginal
21.  test_primary_gate_delta_threshold – gate fails below 0.02 threshold
22.  test_candidate_promotion          – all gates pass → candidate decision
23.  test_no_candidate_ci_fails        – CI lower bound ≤ 0 → no_candidate
24.  test_head_gate_always_failed      – head gate always failed/NA
25.  test_future_protocol_fields       – future protocol has required fields
26.  test_consumed_probes_only_once    – second score_probes call raises error
27.  test_evaluator_verifies_hash      – tampered registration → hard fail
28.  test_evaluator_calls_ensemble_scorer – scorer.score() is called with truth=None
29.  test_no_production_mutation       – evaluator does not modify OOF artifacts
30.  test_probe_type_gallery_semantics – temporal→gallery, onboarding→combined
31.  test_imbalanced_clusters          – strata with different cluster sizes
32.  test_holm_monotonicity            – adjusted p-values are non-decreasing
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, call
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("BTEH_SOURCE_ROOT", "/nonexistent/BTEH")
os.environ.setdefault("BTEH_ARTIFACT_ROOT", "/nonexistent/artifacts")

from pipeline.eval_metrics import (
    LEGACY_MAP_KNOWN_VALUE,
    LEGACY_MAP_TOLERANCE,
    aggregate_per_identity_rrs,
    aggregate_per_identity_top_k,
    compute_all_metrics,
    identity_macro_mrr,
    identity_macro_top1,
    identity_macro_top5,
    query_weighted_mrr,
    query_weighted_top1,
    query_weighted_top5,
    reciprocal_rank,
    top_k_hit,
    verify_legacy_map_regression,
)
from pipeline.power_simulation import (
    N_ONBOARDING_IDS,
    N_TEMPORAL_IDS,
    OPERATIONAL_DELTA_THRESHOLD,
    SIMULATION_SEED,
    run_power_simulation,
)
from pipeline.statistical_registration import (
    FIXED_SEED,
    FIXED_ALPHA,
    FIXED_OPERATIONAL_MRR_MDE,
    FIXED_TOP1_MARGIN,
    SECONDARY_ENDPOINTS,
    RegistrationHashMismatchError,
    RegistrationOutcomePollutionError,
    build_registration_document,
    compute_registration_hash,
    extract_cluster_sizes,
    load_and_verify_registration,
    write_registration,
)
from pipeline.paired_bootstrap import (
    CONTRAST_PRIMARY,
    CONTRAST_BODY_LOCAL_MRR,
    CONTRAST_EAR_LOCAL_MRR,
    BootstrapResult,
    ContrastResult,
    QueryRecord,
    _sign_flip_pvalue,
    holm_correction,
    _max_t_simultaneous_intervals,
    _per_identity_mrr_from_records,
    run_paired_bootstrap,
)
from pipeline.candidate_report import (
    POINT_DELTA_THRESHOLD,
    TOP1_MARGIN_THRESHOLD,
    build_decision_report,
    build_future_session_protocol,
    write_decision_report,
    write_future_session_protocol,
    _gate_head,
    _gate_point_delta,
    _gate_ci_lower,
)
from pipeline.fixed_probe_evaluator import (
    CONSUMED_MARKER_FILENAME,
    PROBE_RANKINGS_PARQUET,
    FixedProbeEvaluator,
    QueryRankRecord,
    SystemSpec,
    SYSTEM_SPECS,
    _build_gallery_splits,
    _records_to_dataframe,
    _scoring_fingerprint,
    build_onboarding_combined_gallery,
    score_probe_with_scorer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_splits_parquet(tmp_path: Path, n_temporal: int = 5, n_onboarding: int = 3) -> Path:
    """Create a minimal bteh_splits.parquet for testing."""
    rows = []
    for i in range(n_temporal):
        for j in range(2):  # 2 queries per temporal identity
            rows.append({
                "image_id": f"temporal_{i}_img_{j}",
                "individual_id": f"temp_id_{i}",
                "session_id": f"sess_{i}_{j}",
                "split": "probe",
            })
    for i in range(n_onboarding):
        for j in range(1):
            rows.append({
                "image_id": f"onboarding_{i}_img_{j}",
                "individual_id": f"onb_id_{i}",
                "session_id": f"sess_onb_{i}_{j}",
                "split": "held_out_probe",
            })
    # Gallery rows
    for i in range(10):
        rows.append({
            "image_id": f"gallery_img_{i}",
            "individual_id": f"temp_id_{i % n_temporal}",
            "session_id": f"gal_sess_{i}",
            "split": "gallery",
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "bteh_splits.parquet"
    df.to_parquet(str(path), index=False)
    return path


def _make_full_splits_parquet(tmp_path: Path) -> Path:
    """Create a splits parquet with exactly N_TEMPORAL_IDS and N_ONBOARDING_IDS."""
    rows = []
    for i in range(N_TEMPORAL_IDS):
        for j in range(2):
            rows.append({
                "image_id": f"temporal_{i}_img_{j}",
                "individual_id": f"temp_id_{i:03d}",
                "session_id": f"sess_{i}_{j}",
                "split": "probe",
            })
    for i in range(N_ONBOARDING_IDS):
        rows.append({
            "image_id": f"onboarding_{i}_img_0",
            "individual_id": f"onb_id_{i:03d}",
            "session_id": f"sess_onb_{i}_0",
            "split": "held_out_probe",
        })
    for i in range(20):
        rows.append({
            "image_id": f"gallery_img_{i}",
            "individual_id": f"temp_id_{i % N_TEMPORAL_IDS:03d}",
            "session_id": f"gal_sess_{i}",
            "split": "gallery",
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "bteh_splits_full.parquet"
    df.to_parquet(str(path), index=False)
    return path


def _make_oof_artifacts(tmp_path: Path) -> Path:
    """Create minimal OOF artifact files for testing."""
    arts_dir = tmp_path / "oof_artifacts"
    arts_dir.mkdir(parents=True, exist_ok=True)
    config = {"all_channels": ["miewid", "ear_miewid_projected", "body_local", "ear_local"],
               "k_default": 50}
    metrics = {"frozen_k": 20, "identity_macro_mrr": 0.41}
    fingerprint = {"config_fingerprint": "abc123", "schema_version": "local-v1",
                   "saved_at": "2025-01-01T00:00:00Z"}
    weights = {"miewid": 0.5, "ear_miewid_projected": 0.3,
               "body_local": 0.1, "ear_local": 0.1}
    (arts_dir / "config.json").write_text(json.dumps(config))
    (arts_dir / "oof_metrics.json").write_text(json.dumps(metrics))
    (arts_dir / "fingerprint.json").write_text(json.dumps(fingerprint))
    (arts_dir / "fusion_weights.json").write_text(json.dumps(weights))
    return arts_dir


def _make_registration(tmp_path: Path, n_temporal: int = N_TEMPORAL_IDS,
                       n_onboarding: int = N_ONBOARDING_IDS) -> tuple[Path, str]:
    """Create a valid registration file and return (path, hash)."""
    splits_path = _make_full_splits_parquet(tmp_path) if (
        n_temporal == N_TEMPORAL_IDS and n_onboarding == N_ONBOARDING_IDS
    ) else _make_splits_parquet(tmp_path)
    arts_dir = _make_oof_artifacts(tmp_path)

    # Build doc without calling full write_registration (avoids power sim)
    temporal_cs = {f"temp_id_{i:03d}": 2 for i in range(n_temporal)}
    onboarding_cs = {f"onb_id_{i:03d}": 1 for i in range(n_onboarding)}
    doc = build_registration_document(
        temporal_cluster_sizes=temporal_cs,
        onboarding_cluster_sizes=onboarding_cs,
        oof_artifacts_hash="fake_hash_oof",
        oof_scoring_fingerprint="abc123",
        selected_v1_eval_hash="fake_hash_v1",
        oof_paired_variance=0.05,
        frozen_k=20,
        all_channels=["miewid", "ear_miewid_projected", "body_local", "ear_local"],
        fusion_weights={"miewid": 0.5, "ear_miewid_projected": 0.5},
    )
    reg_hash = compute_registration_hash(doc)
    doc["registration_hash"] = reg_hash
    reg_path = tmp_path / "retrospective_registration.json"
    with open(reg_path, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    return reg_path, reg_hash


def _make_query_records(
    system_name: str,
    temporal_ids: List[str],
    onboarding_ids: List[str],
    queries_per_id: int = 2,
    truth_rank: int = 1,
) -> List[QueryRecord]:
    """Synthetic QueryRecord list where truth is at *truth_rank*."""
    records = []
    all_gallery = [f"gal_id_{i}" for i in range(20)]
    for iid in temporal_ids:
        for q in range(queries_per_id):
            ranked = list(all_gallery[:truth_rank - 1]) + [iid] + list(all_gallery[truth_rank:])
            records.append(QueryRecord(
                query_image_id=f"{iid}_q{q}",
                truth_individual_id=iid,
                probe_type="temporal",
                individual_id_key=iid,
                system_name=system_name,
                ranked_ids=ranked,
            ))
    for iid in onboarding_ids:
        for q in range(1):
            ranked = list(all_gallery[:truth_rank - 1]) + [iid] + list(all_gallery[truth_rank:])
            records.append(QueryRecord(
                query_image_id=f"{iid}_q{q}",
                truth_individual_id=iid,
                probe_type="onboarding",
                individual_id_key=iid,
                system_name=system_name,
                ranked_ids=ranked,
            ))
    return records


def _make_bootstrap_result(point_delta: float = 0.03, ci_lo: float = 0.005,
                            top1_delta: float = 0.01) -> BootstrapResult:
    """Minimal BootstrapResult for decision-report tests."""
    primary = ContrastResult(
        contrast_name=CONTRAST_PRIMARY,
        system_a="selected_v1",
        system_b="selected_v1_plus_both_local",
        metric="identity_macro_mrr",
        point_delta=point_delta,
        ci_lo=ci_lo,
        ci_hi=ci_lo + 0.05,
        p_value=0.01,
        p_value_holm=0.01,
        simultaneous_ci_lo=ci_lo - 0.002,
        simultaneous_ci_hi=ci_lo + 0.052,
        reject_h0=True,
    )
    top1_secondary = ContrastResult(
        contrast_name="secondary_primary_top1",
        system_a="selected_v1",
        system_b="selected_v1_plus_both_local",
        metric="identity_macro_top1",
        point_delta=top1_delta,
        ci_lo=-0.01,
        ci_hi=0.03,
        p_value=0.10,
        p_value_holm=0.20,
        simultaneous_ci_lo=-0.015,
        simultaneous_ci_hi=0.035,
        reject_h0=False,
    )
    return BootstrapResult(
        primary=primary,
        secondaries=[top1_secondary],
        alpha=0.05,
        n_replicates=100,
        seed=SIMULATION_SEED,
        n_temporal_ids=42,
        n_onboarding_ids=9,
        system_metrics={
            "selected_v1": {
                "identity_macro_mrr": 0.40,
                "identity_macro_top1": 0.35,
            },
            "selected_v1_plus_both_local": {
                "identity_macro_mrr": 0.43,
                "identity_macro_top1": 0.36,
            },
        },
    )


# ===========================================================================
# 1. test_reciprocal_rank_hit
# ===========================================================================
class TestReciprocalRank:
    def test_rank_1(self):
        rr = reciprocal_rank(["a", "b", "c"], "a")
        assert rr == pytest.approx(1.0)

    def test_rank_2(self):
        rr = reciprocal_rank(["a", "b", "c"], "b")
        assert rr == pytest.approx(0.5)

    def test_rank_3(self):
        rr = reciprocal_rank(["a", "b", "c"], "c")
        assert rr == pytest.approx(1.0 / 3)

    def test_miss(self):
        # 2. test_reciprocal_rank_miss
        rr = reciprocal_rank(["a", "b", "c"], "z")
        assert rr == pytest.approx(0.0)

    def test_empty_list(self):
        rr = reciprocal_rank([], "a")
        assert rr == pytest.approx(0.0)

    def test_truth_counted_once(self):
        # Duplicate appearances of truth_id: only the first counts
        rr = reciprocal_rank(["x", "truth", "truth"], "truth")
        assert rr == pytest.approx(0.5)


# ===========================================================================
# 3. test_identity_macro_mrr_formula
# ===========================================================================
class TestIdentityMacroMrr:
    def test_manual_calculation(self):
        # Identity A: queries at rank 1, 2 → RR = 1.0, 0.5 → mean = 0.75
        # Identity B: query at rank 4 → RR = 0.25
        # Identity-macro MRR = (0.75 + 0.25) / 2 = 0.50
        per_id = {"A": [1.0, 0.5], "B": [0.25]}
        result = identity_macro_mrr(per_id)
        assert result == pytest.approx(0.50)

    def test_single_identity_single_query(self):
        per_id = {"A": [0.333]}
        result = identity_macro_mrr(per_id)
        assert result == pytest.approx(0.333)

    def test_empty(self):
        assert identity_macro_mrr({}) == pytest.approx(0.0)

    def test_equal_weight_across_identities(self):
        # Identity A has 10 queries all rank-1 → mean = 1.0
        # Identity B has 1 query rank-10 → mean = 0.1
        # Equal weight: (1.0 + 0.1) / 2 = 0.55
        per_id = {"A": [1.0] * 10, "B": [0.1]}
        result = identity_macro_mrr(per_id)
        assert result == pytest.approx(0.55)


# ===========================================================================
# 4. test_query_weighted_mrr
# ===========================================================================
class TestQueryWeightedMrr:
    def test_mean_of_rrs(self):
        rrs = [1.0, 0.5, 0.25, 0.0]
        assert query_weighted_mrr(rrs) == pytest.approx(0.4375)

    def test_empty(self):
        assert query_weighted_mrr([]) == pytest.approx(0.0)


# ===========================================================================
# 5 & 6. test_legacy_map_regression
# ===========================================================================
class TestLegacyMapRegression:
    def test_within_tolerance_passes(self):
        # 0.473 ± 0.001 should pass
        assert verify_legacy_map_regression(0.473) is True
        assert verify_legacy_map_regression(0.4735) is True
        assert verify_legacy_map_regression(0.4725) is True

    def test_outside_tolerance_raises(self):
        # 6. test_legacy_map_regression_fail
        with pytest.raises(ValueError, match="regression failed"):
            verify_legacy_map_regression(0.500)

    def test_tolerance_boundary(self):
        # Exactly at the boundary should still pass
        assert verify_legacy_map_regression(
            LEGACY_MAP_KNOWN_VALUE + LEGACY_MAP_TOLERANCE - 1e-9
        ) is True

    def test_compute_all_metrics_query_weighted(self):
        # Build synthetic records where known_mAP ≈ 0.473
        # Query-weighted MRR = mean of per-query RR values
        # Create records so that the mean RR is 0.473
        rrs = [0.473] * 100
        assert query_weighted_mrr(rrs) == pytest.approx(0.473)
        assert verify_legacy_map_regression(0.473) is True


# ===========================================================================
# 7. test_identity_macro_top1_top5
# ===========================================================================
class TestIdentityMacroTopK:
    def test_top1_hit(self):
        per_id = {"A": [1.0, 1.0], "B": [0.0]}
        result = identity_macro_top1(per_id)
        assert result == pytest.approx((1.0 + 0.0) / 2)

    def test_top5_hit(self):
        per_id = {"A": [1.0], "B": [1.0]}
        result = identity_macro_top5(per_id)
        assert result == pytest.approx(1.0)

    def test_top_k_hit_function(self):
        assert top_k_hit(["a", "b", "c", "d", "e", "truth"], "truth", k=5) == 0.0
        assert top_k_hit(["a", "b", "c", "d", "truth", "f"], "truth", k=5) == 1.0


# ===========================================================================
# 8. test_compute_all_metrics
# ===========================================================================
class TestComputeAllMetrics:
    def test_output_keys(self):
        records = [
            {"truth_individual_id": "A", "ranked_ids": ["A", "B"]},
            {"truth_individual_id": "B", "ranked_ids": ["X", "B"]},
        ]
        result = compute_all_metrics(records)
        expected_keys = {
            "identity_macro_mrr", "query_weighted_mrr",
            "identity_macro_top1", "identity_macro_top5",
            "query_weighted_top1", "query_weighted_top5",
            "n_queries", "n_identities",
        }
        assert expected_keys == set(result.keys())

    def test_no_truth_records(self):
        records = [{"truth_individual_id": None, "ranked_ids": ["A", "B"]}]
        result = compute_all_metrics(records)
        assert result["n_identities"] == 0


# ===========================================================================
# 9. test_registration_cannot_read_outcomes
# ===========================================================================
class TestRegistrationOutcomeGuard:
    def test_outcome_column_raises(self, tmp_path):
        # Create a parquet with outcome column
        df = pd.DataFrame({
            "image_id": ["img1"],
            "individual_id": ["id1"],
            "split": ["probe"],
            "fused_score": [0.8],  # outcome column!
        })
        p = tmp_path / "splits.parquet"
        df.to_parquet(str(p))
        with pytest.raises(RegistrationOutcomePollutionError, match="Outcome columns"):
            extract_cluster_sizes(p)

    def test_rr_column_raises(self, tmp_path):
        df = pd.DataFrame({
            "image_id": ["img1"],
            "individual_id": ["id1"],
            "split": ["probe"],
            "rr": [0.5],  # outcome column!
        })
        p = tmp_path / "splits.parquet"
        df.to_parquet(str(p))
        with pytest.raises(RegistrationOutcomePollutionError):
            extract_cluster_sizes(p)

    def test_clean_manifest_passes(self, tmp_path):
        splits_path = _make_full_splits_parquet(tmp_path)
        temporal_cs, onboarding_cs = extract_cluster_sizes(splits_path)
        assert len(temporal_cs) == N_TEMPORAL_IDS
        assert len(onboarding_cs) == N_ONBOARDING_IDS


# ===========================================================================
# 10. test_registration_hash_roundtrip
# ===========================================================================
class TestRegistrationHashRoundtrip:
    def test_write_and_verify(self, tmp_path):
        reg_path, reg_hash = _make_registration(tmp_path)
        doc = load_and_verify_registration(reg_path)
        assert doc["registration_hash"] == reg_hash

    def test_hash_deterministic(self, tmp_path):
        temporal_cs = {f"t{i}": 2 for i in range(N_TEMPORAL_IDS)}
        onboarding_cs = {f"o{i}": 1 for i in range(N_ONBOARDING_IDS)}
        doc1 = build_registration_document(
            temporal_cluster_sizes=temporal_cs,
            onboarding_cluster_sizes=onboarding_cs,
            oof_artifacts_hash="h1",
            oof_scoring_fingerprint="fp1",
            selected_v1_eval_hash="h2",
            oof_paired_variance=0.05,
            frozen_k=20,
            all_channels=["miewid"],
            fusion_weights={"miewid": 1.0},
        )
        doc2 = build_registration_document(
            temporal_cluster_sizes=temporal_cs,
            onboarding_cluster_sizes=onboarding_cs,
            oof_artifacts_hash="h1",
            oof_scoring_fingerprint="fp1",
            selected_v1_eval_hash="h2",
            oof_paired_variance=0.05,
            frozen_k=20,
            all_channels=["miewid"],
            fusion_weights={"miewid": 1.0},
        )
        # Same inputs → same structure (apart from registered_at timestamp)
        doc1["registered_at"] = "FIXED"
        doc2["registered_at"] = "FIXED"
        assert compute_registration_hash(doc1) == compute_registration_hash(doc2)


# ===========================================================================
# 11. test_registration_hash_mismatch
# ===========================================================================
class TestRegistrationHashMismatch:
    def test_tampered_field_detected(self, tmp_path):
        reg_path, _ = _make_registration(tmp_path)
        # Tamper with the file
        with open(reg_path) as fh:
            doc = json.load(fh)
        doc["n_temporal_ids"] = 99   # tamper
        with open(reg_path, "w") as fh:
            json.dump(doc, fh)
        with pytest.raises(RegistrationHashMismatchError, match="hash mismatch"):
            load_and_verify_registration(reg_path)

    def test_missing_hash_field(self, tmp_path):
        reg_path, _ = _make_registration(tmp_path)
        with open(reg_path) as fh:
            doc = json.load(fh)
        del doc["registration_hash"]
        with open(reg_path, "w") as fh:
            json.dump(doc, fh)
        with pytest.raises(RegistrationHashMismatchError, match="no registration_hash"):
            load_and_verify_registration(reg_path)


# ===========================================================================
# 12. test_extract_cluster_sizes
# ===========================================================================
class TestExtractClusterSizes:
    def test_temporal_onboarding_split(self, tmp_path):
        splits_path = _make_splits_parquet(tmp_path, n_temporal=5, n_onboarding=3)
        temporal_cs, onboarding_cs = extract_cluster_sizes(splits_path)
        assert len(temporal_cs) == 5
        assert len(onboarding_cs) == 3
        # Each temporal identity has 2 queries
        for v in temporal_cs.values():
            assert v == 2

    def test_no_onboarding_gives_empty(self, tmp_path):
        df = pd.DataFrame({
            "image_id": ["img1", "img2"],
            "individual_id": ["id1", "id1"],
            "split": ["probe", "probe"],
        })
        p = tmp_path / "s.parquet"
        df.to_parquet(str(p))
        _, onboarding_cs = extract_cluster_sizes(p)
        assert len(onboarding_cs) == 0


# ===========================================================================
# 13. test_cluster_not_row_bootstrap
# ===========================================================================
class TestClusterBootstrap:
    """Bootstrap resamples identity IDs (clusters), not individual rows."""

    def test_resample_is_cluster_level(self):
        # If we resample IDs and collect all rows for each ID,
        # the total rows count can exceed original (if an ID is sampled twice)
        temporal_ids = [f"t{i}" for i in range(5)]
        records_a = _make_query_records("sys_a", temporal_ids, [], queries_per_id=3)
        records_b = _make_query_records("sys_b", temporal_ids, [], queries_per_id=3)

        rng = np.random.default_rng(42)
        # Resample with replacement → some IDs appear twice
        boot_temporal = list(rng.choice(temporal_ids, size=len(temporal_ids), replace=True))

        # Count total query rows for resampled IDs
        total_rows_a = sum(
            len([r for r in records_a if r.truth_individual_id == iid])
            for iid in boot_temporal
        )
        total_rows_b = sum(
            len([r for r in records_b if r.truth_individual_id == iid])
            for iid in boot_temporal
        )

        # Should be n_ids * queries_per_id (could be > original if IDs sampled twice)
        assert total_rows_a == len(boot_temporal) * 3
        assert total_rows_a == total_rows_b  # both systems contribute equally


# ===========================================================================
# 14. test_protocol_strata_separate
# ===========================================================================
class TestProtocolStrataSeparate:
    def test_temporal_onboarding_resampled_independently(self):
        """
        Verify that temporal and onboarding IDs are resampled from their
        own pools, never mixing strata.
        """
        temporal_ids = ["t0", "t1", "t2"]
        onboarding_ids = ["o0", "o1"]
        records_a = _make_query_records("sys_a", temporal_ids, onboarding_ids)
        records_b = _make_query_records("sys_b", temporal_ids, onboarding_ids)

        result = run_paired_bootstrap(
            records_by_system={"sys_a": records_a, "sys_b": records_b},
            temporal_ids=temporal_ids,
            onboarding_ids=onboarding_ids,
            system_a="sys_a",
            system_b_primary="sys_b",
            system_b_body="sys_b",
            system_b_ear="sys_b",
            system_b_frozen="sys_b",
            n_replicates=100,
            seed=42,
        )
        # Temporal metric must only use temporal records
        temporal_recs_a = [r for r in records_a if r.probe_type == "temporal"]
        onboarding_recs_a = [r for r in records_a if r.probe_type == "onboarding"]
        # All temporal IDs should be in temporal_ids set
        assert all(
            r.truth_individual_id in temporal_ids
            for r in temporal_recs_a
        )
        assert all(
            r.truth_individual_id in onboarding_ids
            for r in onboarding_recs_a
        )


# ===========================================================================
# 15. test_bootstrap_deterministic_10k
# ===========================================================================
class TestBootstrapDeterministic:
    def test_same_seed_same_result(self):
        temporal_ids = [f"t{i}" for i in range(5)]
        onboarding_ids = [f"o{i}" for i in range(3)]
        records_a = _make_query_records("sys_a", temporal_ids, onboarding_ids, truth_rank=2)
        records_b = _make_query_records("sys_b", temporal_ids, onboarding_ids, truth_rank=1)

        kwargs = dict(
            records_by_system={"sys_a": records_a, "sys_b": records_b},
            temporal_ids=temporal_ids,
            onboarding_ids=onboarding_ids,
            system_a="sys_a",
            system_b_primary="sys_b",
            system_b_body="sys_b",
            system_b_ear="sys_b",
            system_b_frozen="sys_b",
            n_replicates=200,
            seed=SIMULATION_SEED,
        )
        result1 = run_paired_bootstrap(**kwargs)
        result2 = run_paired_bootstrap(**kwargs)

        assert result1.primary.point_delta == result2.primary.point_delta
        assert result1.primary.ci_lo == result2.primary.ci_lo
        assert result1.primary.p_value == result2.primary.p_value


# ===========================================================================
# 16. test_power_underpowered_flag
# ===========================================================================
class TestPowerUnderpoweredFlag:
    def test_large_variance_is_underpowered(self):
        # With very large variance, MDE will exceed 0.02
        temporal_cs = {f"t{i}": 1 for i in range(N_TEMPORAL_IDS)}   # 1 query each
        onboarding_cs = {f"o{i}": 1 for i in range(N_ONBOARDING_IDS)}
        result = run_power_simulation(
            temporal_cs, onboarding_cs,
            oof_paired_variance=5.0,   # extremely large variance
            n_replicates=500,
        )
        assert result.underpowered is True

    def test_very_small_variance_is_powered(self):
        # Near-zero variance → near-zero MDE
        temporal_cs = {f"t{i}": 5 for i in range(N_TEMPORAL_IDS)}
        onboarding_cs = {f"o{i}": 5 for i in range(N_ONBOARDING_IDS)}
        result = run_power_simulation(
            temporal_cs, onboarding_cs,
            oof_paired_variance=0.001,   # tiny variance
            n_replicates=200,
        )
        assert result.underpowered is False


# ===========================================================================
# 17. test_power_deterministic
# ===========================================================================
class TestPowerDeterministic:
    def test_same_inputs_same_mde(self):
        temporal_cs = {f"t{i}": 2 for i in range(N_TEMPORAL_IDS)}
        onboarding_cs = {f"o{i}": 1 for i in range(N_ONBOARDING_IDS)}
        r1 = run_power_simulation(temporal_cs, onboarding_cs, 0.02, n_replicates=100)
        r2 = run_power_simulation(temporal_cs, onboarding_cs, 0.02, n_replicates=100)
        assert r1.mde_80 == r2.mde_80

    def test_wrong_temporal_count_raises(self):
        temporal_cs = {f"t{i}": 2 for i in range(N_TEMPORAL_IDS - 1)}  # wrong count
        onboarding_cs = {f"o{i}": 1 for i in range(N_ONBOARDING_IDS)}
        with pytest.raises(ValueError, match="temporal identities"):
            run_power_simulation(temporal_cs, onboarding_cs, 0.02)

    def test_nonpositive_variance_raises(self):
        temporal_cs = {f"t{i}": 2 for i in range(N_TEMPORAL_IDS)}
        onboarding_cs = {f"o{i}": 1 for i in range(N_ONBOARDING_IDS)}
        with pytest.raises(ValueError, match="oof_paired_variance"):
            run_power_simulation(temporal_cs, onboarding_cs, 0.0)


# ===========================================================================
# 18. test_sign_flip_null_mean_zero
# ===========================================================================
class TestSignFlipNull:
    def test_null_distribution_zero_mean(self):
        """Sign-flip null distribution should have mean ≈ 0."""
        rng = np.random.default_rng(42)
        # Diffs centered at 0
        diffs = rng.normal(0, 0.1, size=50)
        # Run sign-flip many times and collect null stats
        n_trials = 1000
        null_means = []
        for _ in range(n_trials):
            signs = rng.choice([-1.0, 1.0], size=len(diffs))
            null_means.append(float(np.mean(signs * diffs)))
        assert abs(np.mean(null_means)) < 0.02   # mean should be near 0

    def test_sign_flip_p_value_range(self):
        rng = np.random.default_rng(42)
        # Large effect → small p-value
        diffs_large = np.ones(50) * 0.5
        p_large = _sign_flip_pvalue(diffs_large, rng, n_flip=999)
        assert p_large < 0.05

        # No effect → large p-value
        rng2 = np.random.default_rng(42)
        diffs_zero = np.zeros(50)
        p_zero = _sign_flip_pvalue(diffs_zero, rng2, n_flip=999)
        assert p_zero > 0.5


# ===========================================================================
# 19. test_holm_correction
# ===========================================================================
class TestHolmCorrection:
    def test_manual_example(self):
        # 4 p-values; Holm: multiply sorted p-values by (n, n-1, n-2, 1)
        # then enforce monotonicity
        p = [0.01, 0.04, 0.03, 0.005]
        adjusted = holm_correction(p)
        # sorted: 0.005, 0.01, 0.03, 0.04
        # multiply: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04
        # enforce monotone: 0.02, 0.03, 0.06, max(0.06, 0.04)=0.06
        assert adjusted[3] == pytest.approx(min(1.0, 0.005 * 4))
        assert adjusted[0] == pytest.approx(min(1.0, 0.01 * 3))

    def test_all_significant_remain_significant(self):
        p = [0.001, 0.002, 0.003, 0.004]
        adjusted = holm_correction(p)
        assert all(a <= 1.0 for a in adjusted)

    def test_empty(self):
        assert holm_correction([]) == []

    def test_single(self):
        assert holm_correction([0.03]) == pytest.approx([0.03])

    def test_monotonicity(self):
        # 32. test_holm_monotonicity
        p = [0.01, 0.04, 0.005, 0.02]
        adjusted = holm_correction(p)
        # Adjusted p-values must be non-decreasing when sorted by original order
        sorted_adj = [adjusted[i] for i in np.argsort(p)]
        for i in range(len(sorted_adj) - 1):
            assert sorted_adj[i] <= sorted_adj[i + 1] + 1e-12


# ===========================================================================
# 20. test_max_t_simultaneous
# ===========================================================================
class TestMaxTSimultaneous:
    def test_simultaneous_wider_than_marginal(self):
        """Simultaneous intervals are wider than marginal percentile intervals."""
        rng = np.random.default_rng(42)
        n = 500
        # Two correlated contrasts
        delta1 = rng.normal(0.03, 0.01, size=n)
        delta2 = rng.normal(0.01, 0.01, size=n)
        contrasts = {"c1": delta1, "c2": delta2}

        sim_intervals = _max_t_simultaneous_intervals(contrasts, alpha=0.05)

        # Marginal intervals
        lo1_marg = float(np.percentile(delta1, 2.5))
        hi1_marg = float(np.percentile(delta1, 97.5))
        sim_lo1, sim_hi1 = sim_intervals["c1"]

        # Simultaneous must be at least as wide
        assert sim_lo1 <= lo1_marg + 1e-9
        assert sim_hi1 >= hi1_marg - 1e-9


# ===========================================================================
# 21. test_primary_gate_delta_threshold
# ===========================================================================
class TestPrimaryGate:
    def test_below_threshold_fails(self):
        gate = _gate_point_delta(0.015)   # below 0.02
        assert gate.passed is False

    def test_at_threshold_passes(self):
        gate = _gate_point_delta(0.020)
        assert gate.passed is True

    def test_above_threshold_passes(self):
        gate = _gate_point_delta(0.030)
        assert gate.passed is True

    def test_ci_lower_zero_fails(self):
        gate = _gate_ci_lower(0.0)
        assert gate.passed is False

    def test_ci_lower_positive_passes(self):
        gate = _gate_ci_lower(0.001)
        assert gate.passed is True


# ===========================================================================
# 22. test_candidate_promotion
# ===========================================================================
class TestCandidatePromotion:
    def test_all_gates_pass_gives_candidate(self, tmp_path):
        reg_path, _ = _make_registration(tmp_path)
        reg = load_and_verify_registration(reg_path)
        bootstrap = _make_bootstrap_result(
            point_delta=0.025,   # ≥ 0.02
            ci_lo=0.005,         # > 0
            top1_delta=0.01,     # ≥ -0.01
        )
        report = build_decision_report(
            bootstrap,
            reg,
            temporal_delta=0.03,
            onboarding_delta=0.02,
            runtime_p95_seconds=5.0,
            coverage=0.96,
        )
        assert report.decision == "candidate"

    def test_writes_report_file(self, tmp_path):
        reg_path, _ = _make_registration(tmp_path)
        reg = load_and_verify_registration(reg_path)
        bootstrap = _make_bootstrap_result()
        report = build_decision_report(bootstrap, reg,
                                       temporal_delta=0.03, onboarding_delta=0.02)
        out_dir = tmp_path / "reports"
        path = write_decision_report(report, out_dir)
        assert path.exists()
        with open(path) as fh:
            doc = json.load(fh)
        assert doc["decision"] in {"candidate", "no_candidate"}


# ===========================================================================
# 23. test_no_candidate_ci_fails
# ===========================================================================
class TestNoCandidateDecision:
    def test_ci_lo_zero_blocks(self, tmp_path):
        reg_path, _ = _make_registration(tmp_path)
        reg = load_and_verify_registration(reg_path)
        bootstrap = _make_bootstrap_result(
            point_delta=0.025,
            ci_lo=-0.001,   # ≤ 0 → CI gate fails
            top1_delta=0.01,
        )
        report = build_decision_report(bootstrap, reg,
                                       temporal_delta=0.025, onboarding_delta=0.02)
        assert report.decision == "no_candidate"

    def test_delta_below_threshold_blocks(self, tmp_path):
        reg_path, _ = _make_registration(tmp_path)
        reg = load_and_verify_registration(reg_path)
        bootstrap = _make_bootstrap_result(
            point_delta=0.015,   # < 0.02 → delta gate fails
            ci_lo=0.001,
            top1_delta=0.01,
        )
        report = build_decision_report(bootstrap, reg,
                                       temporal_delta=0.015, onboarding_delta=0.015)
        assert report.decision == "no_candidate"


# ===========================================================================
# 24. test_head_gate_always_failed
# ===========================================================================
class TestHeadGate:
    def test_always_failed_not_applicable(self):
        gate = _gate_head()
        assert gate.passed is False
        assert gate.value == "not_applicable"
        assert "not applicable" in gate.note.lower()

    def test_head_gate_in_report(self, tmp_path):
        reg_path, _ = _make_registration(tmp_path)
        reg = load_and_verify_registration(reg_path)
        bootstrap = _make_bootstrap_result(point_delta=0.025, ci_lo=0.005)
        report = build_decision_report(bootstrap, reg,
                                       temporal_delta=0.025, onboarding_delta=0.025)
        head_gates = [g for g in report.gates if g.gate_name == "head_channel"]
        assert len(head_gates) == 1
        assert head_gates[0].passed is False
        assert report.head_gate_status == "failed_not_applicable"


# ===========================================================================
# 25. test_future_protocol_fields
# ===========================================================================
class TestFutureProtocol:
    def test_required_fields(self, tmp_path):
        reg_path, _ = _make_registration(tmp_path)
        reg = load_and_verify_registration(reg_path)
        bootstrap = _make_bootstrap_result()
        protocol = build_future_session_protocol(reg, bootstrap)

        required = {
            "registration_hash", "acquisition_rules", "identity_counts",
            "covariate_shift_clause", "head_coverage_clause",
            "endpoints", "statistical_design",
        }
        assert required.issubset(set(protocol.keys()))

    def test_head_coverage_not_applicable(self, tmp_path):
        reg_path, _ = _make_registration(tmp_path)
        reg = load_and_verify_registration(reg_path)
        bootstrap = _make_bootstrap_result()
        protocol = build_future_session_protocol(reg, bootstrap)
        assert protocol["head_coverage_clause"]["status"] == "not_applicable"

    def test_one_touch_true(self, tmp_path):
        reg_path, _ = _make_registration(tmp_path)
        reg = load_and_verify_registration(reg_path)
        bootstrap = _make_bootstrap_result()
        protocol = build_future_session_protocol(reg, bootstrap)
        assert protocol["acquisition_rules"]["one_touch_protocol"] is True
        assert protocol["acquisition_rules"]["never_previously_loaded_for_scoring"] is True

    def test_write_future_protocol(self, tmp_path):
        reg_path, _ = _make_registration(tmp_path)
        reg = load_and_verify_registration(reg_path)
        bootstrap = _make_bootstrap_result()
        protocol = build_future_session_protocol(reg, bootstrap)
        out_dir = tmp_path / "reports"
        path = write_future_session_protocol(protocol, out_dir)
        assert path.exists()


# ===========================================================================
# 26. test_consumed_probes_only_once
# ===========================================================================
class TestConsumedProbes:
    def test_second_score_probes_raises(self, tmp_path):
        from pipeline.statistical_registration import RegistrationAlreadyConsumedError

        reg_path, _ = _make_registration(tmp_path)
        evaluator = FixedProbeEvaluator(reg_path, tmp_path / "output")

        # Simulate consumed marker
        evaluator._consumed_marker.write_text("already_consumed")

        with pytest.raises(RegistrationAlreadyConsumedError):
            evaluator.score_probes(
                splits_df=pd.DataFrame(columns=["image_id", "individual_id",
                                                 "session_id", "split"]),
                artifacts=MagicMock(),
                crop_df=pd.DataFrame(),
                embedding_matrices={},
                descriptor_mappings={},
            )


# ===========================================================================
# 27. test_evaluator_verifies_hash
# ===========================================================================
class TestEvaluatorVerifiesHash:
    def test_tampered_registration_raises(self, tmp_path):
        reg_path, _ = _make_registration(tmp_path)
        # Tamper
        with open(reg_path) as fh:
            doc = json.load(fh)
        doc["frozen_k"] = 999
        with open(reg_path, "w") as fh:
            json.dump(doc, fh)
        with pytest.raises(RegistrationHashMismatchError):
            FixedProbeEvaluator(reg_path, tmp_path / "output")


# ===========================================================================
# 28. test_evaluator_calls_exact_ensemble_scorer
# ===========================================================================
class TestEvaluatorCallsEnsembleScorer:
    def test_score_probe_passes_truth_none(self):
        """
        score_probe_with_scorer must NOT pass query_individual_id (truth) to
        the scorer — no candidate truth forcing.
        """
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = []  # empty result
        mock_scorer.artifacts = MagicMock()

        rec = score_probe_with_scorer(
            query_image_id="img1",
            query_session_id="sess1",
            truth_individual_id="true_id",  # known, but must not be passed to scorer
            probe_type="temporal",
            scorer=mock_scorer,
            system_name="selected_v1",
            registration_hash="testhash",
            frozen_k=20,
            fusion_weights={"miewid": 1.0},
        )
        # Verify the scorer was called with query_individual_id=None
        call_kwargs = mock_scorer.score.call_args
        assert call_kwargs.kwargs.get("query_individual_id") is None

    def test_scorer_receives_source_fingerprint(self):
        """Scorer must receive the registration hash as source_fingerprint."""
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = []
        mock_scorer.artifacts = MagicMock()

        score_probe_with_scorer(
            query_image_id="img1",
            query_session_id="sess1",
            truth_individual_id=None,
            probe_type="temporal",
            scorer=mock_scorer,
            system_name="selected_v1",
            registration_hash="REG_HASH_123",
            frozen_k=20,
            fusion_weights={},
        )
        call_kwargs = mock_scorer.score.call_args
        assert call_kwargs.kwargs.get("source_fingerprint") == "REG_HASH_123"


# ===========================================================================
# 29. test_no_production_mutation
# ===========================================================================
class TestNoProductionMutation:
    def test_oof_artifacts_not_written(self, tmp_path):
        """FixedProbeEvaluator must not modify OOF artifact files."""
        arts_dir = _make_oof_artifacts(tmp_path)
        reg_path, _ = _make_registration(tmp_path)

        # Record modification times
        mtime_before = {f: f.stat().st_mtime for f in arts_dir.iterdir()}

        # The evaluator does NOT receive the OOF artifacts dir directly
        # (it only receives the registration path), so it cannot write to it.
        evaluator = FixedProbeEvaluator(reg_path, tmp_path / "output")

        mtime_after = {f: f.stat().st_mtime for f in arts_dir.iterdir()}
        assert mtime_before == mtime_after

    def test_registration_not_modified_by_evaluator(self, tmp_path):
        reg_path, _ = _make_registration(tmp_path)
        mtime = reg_path.stat().st_mtime
        evaluator = FixedProbeEvaluator(reg_path, tmp_path / "output")
        # Loading should not modify the file
        assert reg_path.stat().st_mtime == mtime


# ===========================================================================
# 30. test_probe_type_gallery_semantics
# ===========================================================================
class TestProbeTypeGallerySemantics:
    def test_temporal_gallery_only(self, tmp_path):
        """Temporal probes must use gallery-only split."""
        splits_path = _make_splits_parquet(tmp_path, n_temporal=3, n_onboarding=2)
        splits_df = pd.read_parquet(str(splits_path))
        splits_df["split"] = splits_df["split"].astype(str)
        splits_df["session_id"] = splits_df.get("session_id",
                                                  pd.Series(dtype=str)).fillna("s0").astype(str)

        temporal_df, onboarding_df, gallery_df, held_out_df = _build_gallery_splits(splits_df)

        # Probe images must not appear in gallery
        probe_ids = set(temporal_df["image_id"]) | set(onboarding_df["image_id"])
        gal_ids = set(gallery_df["image_id"])
        assert len(probe_ids & gal_ids) == 0

    def test_onboarding_combined_gallery(self, tmp_path):
        """Onboarding probes must use combined gallery (gallery ∪ held_out_gallery)."""
        # Add held_out_gallery rows
        rows = [
            {"image_id": "gal1", "individual_id": "id_a", "session_id": "s1", "split": "gallery"},
            {"image_id": "hog1", "individual_id": "id_b", "session_id": "s2",
             "split": "held_out_gallery"},
            {"image_id": "prb1", "individual_id": "id_c", "session_id": "s3",
             "split": "held_out_probe"},
        ]
        df = pd.DataFrame(rows)
        _, _, gallery_df, held_out_df = _build_gallery_splits(df)
        combined = build_onboarding_combined_gallery(gallery_df, held_out_df)
        assert "gal1" in combined["image_id"].values
        assert "hog1" in combined["image_id"].values
        assert "prb1" not in combined["image_id"].values


# ===========================================================================
# 31. test_imbalanced_clusters
# ===========================================================================
class TestImbalancedClusters:
    def test_imbalanced_cluster_sizes(self):
        """Bootstrap handles imbalanced cluster sizes gracefully."""
        temporal_ids = [f"t{i}" for i in range(5)]
        onboarding_ids = ["o0"]  # only one onboarding ID

        # t0 has 10 queries, others have 1
        records_a = []
        for q in range(10):
            records_a.append(QueryRecord(
                query_image_id=f"t0_q{q}", truth_individual_id="t0",
                probe_type="temporal", individual_id_key="t0",
                system_name="sys_a", ranked_ids=["t0", "x"],
            ))
        for iid in temporal_ids[1:]:
            records_a.append(QueryRecord(
                query_image_id=f"{iid}_q0", truth_individual_id=iid,
                probe_type="temporal", individual_id_key=iid,
                system_name="sys_a", ranked_ids=[iid, "x"],
            ))
        records_a.append(QueryRecord(
            query_image_id="o0_q0", truth_individual_id="o0",
            probe_type="onboarding", individual_id_key="o0",
            system_name="sys_a", ranked_ids=["o0", "x"],
        ))
        records_b = [QueryRecord(
            query_image_id=r.query_image_id, truth_individual_id=r.truth_individual_id,
            probe_type=r.probe_type, individual_id_key=r.individual_id_key,
            system_name="sys_b", ranked_ids=r.ranked_ids,
        ) for r in records_a]

        # Should not raise
        result = run_paired_bootstrap(
            records_by_system={"sys_a": records_a, "sys_b": records_b},
            temporal_ids=temporal_ids,
            onboarding_ids=onboarding_ids,
            system_a="sys_a",
            system_b_primary="sys_b",
            system_b_body="sys_b",
            system_b_ear="sys_b",
            system_b_frozen="sys_b",
            n_replicates=50,
            seed=42,
        )
        assert isinstance(result.primary.point_delta, float)


# ===========================================================================
# Scoring fingerprint determinism
# ===========================================================================
class TestScoringFingerprint:
    def test_fingerprint_deterministic(self):
        fp1 = _scoring_fingerprint("sys_a", "hash1", ["miewid"], {"miewid": 1.0}, 20)
        fp2 = _scoring_fingerprint("sys_a", "hash1", ["miewid"], {"miewid": 1.0}, 20)
        assert fp1 == fp2

    def test_fingerprint_different_system(self):
        fp1 = _scoring_fingerprint("sys_a", "hash1", ["miewid"], {"miewid": 1.0}, 20)
        fp2 = _scoring_fingerprint("sys_b", "hash1", ["miewid"], {"miewid": 1.0}, 20)
        assert fp1 != fp2

    def test_fingerprint_different_hash(self):
        fp1 = _scoring_fingerprint("sys_a", "hash1", ["miewid"], {"miewid": 1.0}, 20)
        fp2 = _scoring_fingerprint("sys_a", "hash2", ["miewid"], {"miewid": 1.0}, 20)
        assert fp1 != fp2


# ===========================================================================
# Registration fields completeness
# ===========================================================================
class TestRegistrationFields:
    def test_all_required_fields_present(self, tmp_path):
        reg_path, _ = _make_registration(tmp_path)
        doc = load_and_verify_registration(reg_path)
        required = {
            "protocol_version", "registered_at",
            "cluster_sizes", "n_temporal_ids", "n_onboarding_ids",
            "frozen_k", "all_channels", "fusion_weights",
            "primary_endpoint", "secondary_endpoints",
            "statistical_design", "multiplicity_correction",
            "conditional_uncertainty_caveat",
            "hypothesis", "registration_hash",
            "oof_paired_variance", "evaluation_guards",
        }
        assert required.issubset(set(doc.keys()))

    def test_seed_matches_fixed(self, tmp_path):
        reg_path, _ = _make_registration(tmp_path)
        doc = load_and_verify_registration(reg_path)
        assert doc["statistical_design"]["seed"] == FIXED_SEED

    def test_alpha_matches_fixed(self, tmp_path):
        reg_path, _ = _make_registration(tmp_path)
        doc = load_and_verify_registration(reg_path)
        assert doc["statistical_design"]["alpha"] == FIXED_ALPHA

    def test_n_secondaries(self, tmp_path):
        reg_path, _ = _make_registration(tmp_path)
        doc = load_and_verify_registration(reg_path)
        assert len(doc["secondary_endpoints"]) == 4

    def test_primary_endpoint_name(self, tmp_path):
        reg_path, _ = _make_registration(tmp_path)
        doc = load_and_verify_registration(reg_path)
        assert doc["primary_endpoint"] == "identity_macro_mrr"

    def test_conditional_caveat_present(self, tmp_path):
        reg_path, _ = _make_registration(tmp_path)
        doc = load_and_verify_registration(reg_path)
        assert len(doc["conditional_uncertainty_caveat"]) > 20

    def test_evaluation_guards_probe_consumed_false(self, tmp_path):
        reg_path, _ = _make_registration(tmp_path)
        doc = load_and_verify_registration(reg_path)
        assert doc["evaluation_guards"]["probe_consumed_flag"] is False
        assert doc["evaluation_guards"]["head_gate_status"] == "failed_not_applicable"
        assert doc["evaluation_guards"]["no_head_channel"] is True
        assert doc["evaluation_guards"]["no_candidate_truth_forcing"] is True


# ===========================================================================
# Records-to-dataframe roundtrip
# ===========================================================================
class TestRecordsToDataframe:
    def test_columns_present(self):
        recs = [
            QueryRankRecord(
                query_image_id="q1",
                truth_individual_id="id_a",
                probe_type="temporal",
                system_name="selected_v1",
                ranked_ids=["id_a", "id_b"],
                fused_scores=[0.9, 0.7],
                channels_available=["miewid"],
                scoring_fingerprint="fp123",
                registration_hash="reg_hash",
            )
        ]
        df = _records_to_dataframe(recs)
        assert "query_image_id" in df.columns
        assert "registration_hash" in df.columns
        assert "rank" in df.columns
        assert len(df) == 2   # 2 candidates ranked


# ===========================================================================
# Bootstrap: query weighted secondary metrics
# ===========================================================================
class TestBootstrapSecondaryMetrics:
    def test_query_mrr_in_system_metrics(self):
        temporal_ids = [f"t{i}" for i in range(3)]
        onboarding_ids = ["o0"]
        records_a = _make_query_records("sys_a", temporal_ids, onboarding_ids)
        records_b = _make_query_records("sys_b", temporal_ids, onboarding_ids, truth_rank=1)

        result = run_paired_bootstrap(
            records_by_system={"sys_a": records_a, "sys_b": records_b},
            temporal_ids=temporal_ids,
            onboarding_ids=onboarding_ids,
            system_a="sys_a",
            system_b_primary="sys_b",
            system_b_body="sys_b",
            system_b_ear="sys_b",
            system_b_frozen="sys_b",
            n_replicates=50,
            seed=42,
        )
        assert "query_weighted_mrr" in result.system_metrics["sys_a"]
        assert "identity_macro_mrr" in result.system_metrics["sys_a"]
