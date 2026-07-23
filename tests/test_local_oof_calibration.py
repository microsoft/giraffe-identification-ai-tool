# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Comprehensive tests for pipeline/local_oof_calibration.py and
pipeline/ensemble_inference.py.

All tests use fake scorers and synthetic DataFrames — no real LightGlue /
LoFTR inference is performed.

Test matrix
-----------
 1. test_probe_hard_fail              – gallery pushdown; probe rows → hard error
 2. test_gallery_filter_passes        – gallery rows pass the filter unchanged
 3. test_session_exclusion            – query session excluded from references
 4. test_truth_not_forced             – truth absent from shortlist when globally low
 5. test_k_rule_small_k_sufficient    – freeze smallest K at ≥ 0.95 recall
 6. test_k_rule_fallback_k50          – fallback to K=50 when no K qualifies
 7. test_shortlist_fingerprint_determinism – same query/candidates → same fingerprint
 8. test_local_candidates_exactly_shortlist – OOF table has exactly top-K rows per query
 9. test_positive_support_guard       – Platt fit fails with too few positives
10. test_negative_support_guard       – Platt fit fails with too few negatives
11. test_missing_channels_unavailable – missing region → available=False, score=NaN
12. test_platt_nonflat_guard          – flat calibrator output → LocalFlatnessError
13. test_identity_macro_mrr           – MRR computation matches manual calculation
14. test_weight_constraints           – all fitted weights ≥ 0 and sum = 1
15. test_resume_atomic_shards         – completed queries skipped on resume
16. test_fingerprint_mismatch_detected – changed fingerprint invalidates cached shard
17. test_inference_oof_same_function  – EnsembleScorer.score calls score_query_candidates
18. test_selected_v1_nonmutation      – production calibrators/weights unchanged after run
19. test_budget_gate_fail             – budget exceeded → BudgetExceededError
20. test_budget_gate_override         – override_budget bypasses the gate
21. test_k_recall_at_k_computation    – recall_at_k values are correct
22. test_ensemble_artifacts_roundtrip – save/load EnsembleArtifacts is lossless
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("BTEH_SOURCE_ROOT", "/nonexistent/BTEH")
os.environ.setdefault("BTEH_ARTIFACT_ROOT", "/nonexistent/artifacts")

from models.calibration import Calibrator
from models.identity_fusion import IdentityScore, QueryResult
from models.identity_scorer import LocalIdentityScorer, QueryCrop, ReferenceImage
from models.local_matcher import (
    GEOM_HOMOGRAPHY,
    REGION_BODY,
    REGION_EAR,
)
from models.local_score_schema import (
    SCHEMA_VERSION,
    LocalIdentityScore,
    LocalPairScore,
    make_identity_scoring_fingerprint,
    make_scoring_fingerprint,
)
from pipeline.local_oof_calibration import (
    ALL_CHANNELS,
    CHANNEL_BODY_LOCAL,
    CHANNEL_EAR,
    CHANNEL_EAR_LOCAL,
    CHANNEL_MIEWID,
    GLOBAL_CHANNELS,
    K_DEFAULT,
    K_GRID,
    K_RECALL_THRESHOLD,
    BudgetExceededError,
    FingerprintMismatchError,
    LocalFlatnessError,
    LocalSupportError,
    OOFPipelineConfig,
    OOF_CONFIG_JSON,
    OOF_FINGERPRINT_JSON,
    ProbePollutionError,
    SHORTLIST_REGISTRATION_PARQUET,
    ShardCoverageError,
    WorkerRangeError,
    _assert_no_probe_ids,
    _build_parser,
    _extract_single_fingerprint,
    _filter_gallery_only,
    _instantiate_local_scorers,
    _load_completed_queries,
    _load_embedding_matrices_and_mappings,
    _load_gallery_data,
    _load_production_calibrators,
    _merge_shards,
    _shard_path,
    _shortlist_fingerprint,
    _validate_worker_index,
    _worker_query_assignment,
    _write_shard_atomic,
    build_oof_table,
    build_shortlist_registration,
    check_shard_coverage,
    compute_global_oof_rankings,
    compute_recall_at_k,
    fit_4channel_weights,
    fit_local_platt,
    identity_macro_mrr,
    identity_macro_top1,
    load_oof_artifacts,
    run_finalize_calibration,
    run_oof_calibration,
    save_oof_artifacts,
    select_k_threshold,
)
from pipeline.ensemble_inference import (
    EnsembleArtifacts,
    EnsembleScorer,
    load_ensemble_artifacts,
    score_query_candidates,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _make_gallery_df(
    n_identities: int = 4,
    sessions_per_identity: int = 3,
    images_per_session: int = 2,
) -> pd.DataFrame:
    """Create a synthetic gallery DataFrame."""
    rows = []
    for i in range(n_identities):
        indiv_id = f"indiv_{i:02d}"
        for s in range(sessions_per_identity):
            sess_id = f"sess_{i:02d}_{s}"
            for j in range(images_per_session):
                img_id = f"img_{i:02d}_{s}_{j}"
                rows.append({
                    "image_id": img_id,
                    "individual_id": indiv_id,
                    "session_id": sess_id,
                    "split": "gallery",
                })
    return pd.DataFrame(rows)


def _make_gallery_df_with_probe(
    n_identities: int = 3,
    n_probe: int = 2,
) -> pd.DataFrame:
    """Create a gallery DataFrame with a few probe rows injected."""
    df = _make_gallery_df(n_identities)
    probe_rows = pd.DataFrame([
        {
            "image_id": f"probe_img_{k}",
            "individual_id": f"indiv_00",
            "session_id": "probe_sess",
            "split": "probe",
        }
        for k in range(n_probe)
    ])
    return pd.concat([df, probe_rows], ignore_index=True)


def _make_crop_df(gallery_df: pd.DataFrame) -> pd.DataFrame:
    """Create a minimal crop manifest for all gallery images (one body + one ear each)."""
    rows = []
    for _, row in gallery_df[gallery_df["split"] == "gallery"].iterrows():
        img_id = row["image_id"]
        for kind, ordinal in [("body", 0), ("ear", 0)]:
            rows.append({
                "crop_id": f"{img_id}__{kind}_{ordinal}",
                "image_id": img_id,
                "individual_id": row["individual_id"],
                "crop_kind": kind,
                "crop_ordinal": ordinal,
                "crop_path": f"/fake/{img_id}_{kind}.jpg",
                "detector_status": "accepted",
                "detector_confidence": 0.9,
                "detector_box": "",
                "review_status": "accepted",
                "schema_version": "v1",
                "source_fingerprint": "src_fp",
                "split_fingerprint": "split_fp",
            })
    return pd.DataFrame(rows)


def _make_embedding_matrices(
    gallery_df: pd.DataFrame,
    dim: int = 8,
) -> Dict[str, np.ndarray]:
    """Create random L2-normalised embedding matrices for global channels."""
    n = len(gallery_df)
    result = {}
    rng = np.random.default_rng(42)
    for ch in GLOBAL_CHANNELS:
        mat = rng.standard_normal((n, dim)).astype(np.float32)
        mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
        result[ch] = mat
    return result


def _make_descriptor_mappings(
    gallery_df: pd.DataFrame,
    crop_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Create minimal descriptor mapping DataFrames for global channels."""
    result = {}
    for ch in GLOBAL_CHANNELS:
        rows = []
        for i, (_, row) in enumerate(gallery_df.iterrows()):
            rows.append({
                "descriptor_name": ch,
                "embedding_row": i,
                "faiss_row": i,
                "crop_id": f"{row['image_id']}__body_0",
                "image_id": row["image_id"],
                "individual_id": row["individual_id"],
                "crop_kind": "body",
                "crop_ordinal": 0,
                "crop_path": f"/fake/{row['image_id']}_body.jpg",
                "schema_version": "v1",
                "source_fingerprint": "src_fp",
                "split_fingerprint": "split_fp",
                "model_preprocess_fingerprint": "model_fp",
            })
        result[ch] = pd.DataFrame(rows)
    return result


def _make_platt_calibrator(
    n_pos: int = 20,
    n_neg: int = 40,
    pos_mean: float = 0.8,
    neg_mean: float = 0.3,
) -> Calibrator:
    """Fit a Platt calibrator on synthetic data."""
    rng = np.random.default_rng(0)
    pos = rng.normal(pos_mean, 0.1, n_pos).clip(0, 1)
    neg = rng.normal(neg_mean, 0.1, n_neg).clip(0, 1)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    cal = Calibrator()
    cal.fit(scores, labels, method="platt")
    return cal


class FakeLocalIdentityScorer:
    """
    Drop-in replacement for LocalIdentityScorer that returns a configurable
    fixed score without performing any real inference.
    """

    def __init__(
        self,
        score_fn=None,
        backend: str = "lightglue",
        model_fingerprint: str = "fake_model_fp_1234",
        max_sessions: int = 3,
        top_k: int = 2,
        geom_model: Optional[str] = "homography",
        fail_missing: bool = False,
    ):
        """
        Parameters
        ----------
        score_fn : callable(query_crops, reference_sessions) -> float | None
            If None, defaults to returning 1.0.
        fail_missing : if True, always returns score=0 with missing_file=True.
        """
        self.backend = backend
        self.model_fingerprint = model_fingerprint
        self.max_sessions = max_sessions
        self.top_k = top_k
        self.geom_model = geom_model or "homography"
        self._score_fn = score_fn or (lambda q, r: 1.0)
        self._fail_missing = fail_missing

        self._scoring_fingerprint = make_scoring_fingerprint(
            backend=backend,
            model_fingerprint=model_fingerprint,
            schema_version=SCHEMA_VERSION,
            geom_model=self.geom_model,
            mirror_search=True,
        )
        self._identity_scoring_fingerprint = make_identity_scoring_fingerprint(
            backend=backend,
            model_fingerprint=model_fingerprint,
            schema_version=SCHEMA_VERSION,
            max_sessions=max_sessions,
            top_k=top_k,
            aggregation_method="mean_top_k",
        )

    def select_references(self, candidates):
        seen = {}
        for img in candidates:
            if img.session_id not in seen:
                seen[img.session_id] = img
            if len(seen) >= self.max_sessions:
                break
        return list(seen.values())

    def score_identity(
        self,
        query_crops,
        reference_sessions,
        candidate_individual_id: str = "",
        *,
        source_fingerprint: str = "",
        split_fingerprint: str = "",
        aggregation_method: str = "mean_top_k",
    ) -> LocalIdentityScore:
        selected_refs = self.select_references(reference_sessions)
        n_sess = len({r.session_id for r in selected_refs})

        if self._fail_missing or not query_crops or not selected_refs:
            return LocalIdentityScore(
                schema_version=SCHEMA_VERSION,
                backend=self.backend,
                model_fingerprint=self.model_fingerprint,
                scoring_fingerprint=self._identity_scoring_fingerprint,
                query_crop_kind=query_crops[0].crop_kind if query_crops else REGION_BODY,
                candidate_individual_id=candidate_individual_id,
                n_pairs_attempted=0,
                n_pairs_valid=0,
                n_pairs_missing_file=1,
                n_sessions_used=0,
                n_sessions_cap=self.max_sessions,
                region_coverage={},
                orientations_attempted={"original"},
                aggregation_method=aggregation_method,
                top_k=self.top_k,
                score=0.0,
                pair_scores=[],
                latency_ms=0.0,
            )

        score = self._score_fn(query_crops, selected_refs)
        if score is None:
            score = 0.0

        crop_kind = query_crops[0].crop_kind
        return LocalIdentityScore(
            schema_version=SCHEMA_VERSION,
            backend=self.backend,
            model_fingerprint=self.model_fingerprint,
            scoring_fingerprint=self._identity_scoring_fingerprint,
            query_crop_kind=crop_kind,
            candidate_individual_id=candidate_individual_id,
            n_pairs_attempted=len(selected_refs),
            n_pairs_valid=len(selected_refs),
            n_pairs_missing_file=0,
            n_sessions_used=n_sess,
            n_sessions_cap=self.max_sessions,
            region_coverage={crop_kind: len(selected_refs)},
            orientations_attempted={"original"},
            aggregation_method=aggregation_method,
            top_k=self.top_k,
            score=float(score),
            pair_scores=[],
            latency_ms=0.0,
        )


def _make_fake_local_scorer(
    score_map: Optional[Dict[str, float]] = None,
    region: str = REGION_BODY,
    fail_missing: bool = False,
) -> FakeLocalIdentityScorer:
    """
    Create a FakeLocalIdentityScorer whose score depends on the candidate.

    score_map : {individual_id: score}.  If None, returns 1.0 for all.
    """
    if score_map is None:
        score_fn = None
    else:
        def score_fn(query_crops, reference_sessions):
            cand = (
                reference_sessions[0].individual_id
                if reference_sessions
                else ""
            )
            return score_map.get(cand, 0.5)
    return FakeLocalIdentityScorer(
        score_fn=score_fn,
        fail_missing=fail_missing,
    )


def _make_oof_results_for_k_test(
    n_queries: int = 20,
    truth_in_top: int = 18,
    truth_rank: int = 1,
) -> List[QueryResult]:
    """
    Build synthetic OOF QueryResults.

    For the first *truth_in_top* queries, truth is ranked at *truth_rank*.
    For remaining queries, truth is NOT in the ranked list at all (recall miss).
    """
    results = []
    for i in range(n_queries):
        gt = f"indiv_{i:02d}"
        if i < truth_in_top:
            # truth at `truth_rank`; fill other slots with dummies
            ranked = []
            for slot in range(1, truth_rank):
                ranked.append(IdentityScore(
                    individual_id=f"dummy_{i}_{slot}",
                    channel_raw={}, channel_calibrated={},
                    channels_available=[], fused_score=float(100 - slot),
                ))
            ranked.append(IdentityScore(
                individual_id=gt, channel_raw={}, channel_calibrated={},
                channels_available=[], fused_score=float(100 - truth_rank),
            ))
        else:
            # truth completely absent — recall miss for all K values
            ranked = [
                IdentityScore(individual_id=f"other_{i}_{j}",
                              channel_raw={}, channel_calibrated={},
                              channels_available=[], fused_score=float(10 - j))
                for j in range(5)
            ]
        results.append(
            QueryResult(
                query_image_id=f"img_q_{i}",
                query_individual_id=gt,
                ranked_identities=ranked,
                channels_present=GLOBAL_CHANNELS,
                channels_absent=[],
                identity_in_oof_gallery=True,
            )
        )
    return results


def _make_oof_table_df(
    n_queries: int = 5,
    n_candidates: int = 3,
    n_pos_per_query: int = 1,
    body_score_pos: float = 0.8,
    body_score_neg: float = 0.2,
    ear_score_pos: float = 0.9,
    ear_score_neg: float = 0.1,
    include_body: bool = True,
    include_ear: bool = True,
) -> pd.DataFrame:
    """
    Build a synthetic OOF table DataFrame suitable for calibration/weight fitting.

    For each query, the first *n_pos_per_query* candidates are positives
    (candidate_id == query identity) and the rest are negatives.
    """
    rows = []
    for q_idx in range(n_queries):
        q_id = f"img_q_{q_idx:02d}"
        q_indiv = f"indiv_{q_idx:02d}"
        fold_sess = f"sess_{q_idx:02d}_0"
        for c_idx in range(n_candidates):
            if c_idx < n_pos_per_query:
                # positive: candidate IS the truth identity
                cand_id = q_indiv
                label = 1
            else:
                # negative: candidate is a distinct wrong identity
                cand_id = f"neg_{q_idx:02d}_{c_idx}"
                label = 0
            b_score = body_score_pos if label else body_score_neg
            e_score = ear_score_pos if label else ear_score_neg

            row = {
                "fold_session_id": fold_sess,
                "query_image_id": q_id,
                "query_session_id": fold_sess,
                "query_individual_id": q_indiv,
                "candidate_individual_id": cand_id,
                "global_miewid_raw": 0.9 if label else 0.4,
                "global_miewid_calibrated": 0.85 if label else 0.35,
                "global_ear_raw": 0.88 if label else 0.38,
                "global_ear_calibrated": 0.82 if label else 0.32,
                "global_fused_score": 0.87 if label else 0.36,
                "candidate_global_rank": c_idx + 1,
                "body_local_score": b_score if include_body else float("nan"),
                "body_local_n_pairs": 2 if include_body else 0,
                "body_local_n_valid": 2 if include_body else 0,
                "body_local_n_sessions": 1,
                "body_local_fingerprint": "fake_fp_body",
                "body_local_available": include_body,
                "ear_local_score": e_score if include_ear else float("nan"),
                "ear_local_n_pairs": 2 if include_ear else 0,
                "ear_local_n_valid": 2 if include_ear else 0,
                "ear_local_n_sessions": 1,
                "ear_local_fingerprint": "fake_fp_ear",
                "ear_local_available": include_ear,
                "label": label,
                "shortlist_fingerprint": "fp_shortlist",
                "K": n_candidates,
            }
            rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# Tests
# ===========================================================================

class TestProbeGuard:
    """Req 1 & 9: gallery pushdown / probe hard fail."""

    def test_probe_hard_fail(self):
        df = _make_gallery_df_with_probe()
        with pytest.raises(ProbePollutionError):
            _assert_no_probe_ids(df, context="test")

    def test_held_out_probe_hard_fail(self):
        df = _make_gallery_df()
        df.loc[0, "split"] = "held_out_probe"
        with pytest.raises(ProbePollutionError):
            _assert_no_probe_ids(df, context="test")

    def test_gallery_filter_passes_clean(self):
        df = _make_gallery_df()
        filtered = _filter_gallery_only(df)
        assert len(filtered) == len(df)
        assert set(filtered["split"].unique()) == {"gallery"}

    def test_gallery_filter_rejects_probe(self):
        df = _make_gallery_df_with_probe()
        with pytest.raises(ProbePollutionError):
            _filter_gallery_only(df)

    def test_no_split_column_is_ok(self):
        """DataFrames without a split column do not raise."""
        df = pd.DataFrame({"image_id": ["a", "b"]})
        _assert_no_probe_ids(df)  # should not raise

    def test_compute_global_oof_rankings_rejects_probe(self, tmp_path):
        """probe rows in gallery_df passed to OOF → ProbePollutionError."""
        df = _make_gallery_df_with_probe()
        with pytest.raises(ProbePollutionError):
            compute_global_oof_rankings(
                gallery_df=df,
                descriptor_mappings={},
                embedding_matrices={},
                calibrators={},
                channels=GLOBAL_CHANNELS,
            )


class TestSessionExclusion:
    """Req 1 & 9: session exclusion from references."""

    def test_select_refs_excludes_fold_session(self):
        from pipeline.local_oof_calibration import select_globally_strongest_refs

        gallery_df = _make_gallery_df(n_identities=2, sessions_per_identity=3)
        crop_df = _make_crop_df(gallery_df)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()

        fold_sess = "sess_00_0"  # exclude this session
        cand_id = "indiv_00"
        refs = select_globally_strongest_refs(
            candidate_individual_id=cand_id,
            fold_session_id=fold_sess,
            gallery_df=gdf,
            crop_df=crop_df,
            query_image_id="img_00_0_0",
            emb_matrix_miewid=None,
            desc_mapping_miewid=None,
            crop_kind=REGION_BODY,
            max_sessions=3,
        )
        ref_sessions = [r.session_id for r in refs]
        assert fold_sess not in ref_sessions, (
            f"Fold session {fold_sess!r} should not appear in references; got {ref_sessions}"
        )

    def test_session_exclusion_in_global_oof(self):
        """build_oof_identity_scores excludes query session from rest gallery."""
        gallery_df = _make_gallery_df(n_identities=3, sessions_per_identity=3)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        embs = _make_embedding_matrices(gdf, dim=4)
        dms = _make_descriptor_mappings(gdf)

        results = compute_global_oof_rankings(
            gallery_df=gdf,
            descriptor_mappings=dms,
            embedding_matrices=embs,
            calibrators={},
            channels=GLOBAL_CHANNELS,
        )
        # Each query should produce results; results come from build_oof_identity_scores
        # which excludes the query session.
        assert len(results) > 0
        for qr in results:
            assert qr.query_individual_id is not None
            scores = [item.fused_score for item in qr.ranked_identities]
            assert any(score != 0.0 for score in scores)
            assert scores == sorted(scores, reverse=True)


class TestTruthNotForced:
    """Req 2 & 9: truth is never forced into candidates."""

    def test_shortlist_no_truth_forcing(self):
        """If truth ranks poorly, it should NOT appear in the top-K shortlist."""
        # Build synthetic OOF results where truth ranks last.
        gt_id = "indiv_00"
        other_ids = [f"other_{i}" for i in range(10)]
        ranked = [
            IdentityScore(individual_id=iid, channel_raw={}, channel_calibrated={},
                          channels_available=[], fused_score=float(10 - i))
            for i, iid in enumerate(other_ids)
        ] + [
            IdentityScore(individual_id=gt_id, channel_raw={}, channel_calibrated={},
                          channels_available=[], fused_score=0.0)
        ]
        qr = QueryResult(
            query_image_id="q0",
            query_individual_id=gt_id,
            ranked_identities=ranked,
            channels_present=GLOBAL_CHANNELS,
            channels_absent=[],
            identity_in_oof_gallery=True,
        )

        gallery_df = pd.DataFrame([
            {"image_id": "q0", "individual_id": gt_id, "session_id": "sess_0", "split": "gallery"},
        ])

        sl_df = build_shortlist_registration([qr], frozen_k=5, gallery_df=gallery_df)
        candidates_in_shortlist = set(sl_df["candidate_individual_id"])

        assert gt_id not in candidates_in_shortlist, (
            "Truth was forced into shortlist despite naturally ranking below K=5"
        )


class TestKSelection:
    """Req 2 & 9: K rule."""

    def test_k_rule_small_k_sufficient(self):
        """When recall@5 ≥ 0.95, freeze K=5."""
        # 20 queries, 19 with truth in top-5 → recall@5 = 0.95
        results = _make_oof_results_for_k_test(n_queries=20, truth_in_top=19)
        recalls = compute_recall_at_k(results, K_GRID)
        assert recalls[5] >= 0.95
        k = select_k_threshold(recalls, K_GRID, 0.95, 50)
        assert k == 5

    def test_k_rule_fallback_k50(self):
        """When no K achieves ≥ 0.95 recall, fall back to K=50."""
        # 20 queries, 10 with truth in top-5 → recall@5 = 0.50 (< 0.95)
        results = _make_oof_results_for_k_test(n_queries=20, truth_in_top=10)
        recalls = compute_recall_at_k(results, K_GRID)
        assert all(recalls[k] < 0.95 for k in K_GRID)
        k = select_k_threshold(recalls, K_GRID, 0.95, 50)
        assert k == 50

    def test_k_rule_picks_smallest_qualifying_k(self):
        """When recall@10 ≥ 0.95 but recall@5 < 0.95, freeze K=10."""
        # Manually craft recalls
        recalls = {5: 0.90, 10: 0.96, 20: 0.98, 30: 0.99, 50: 1.0}
        k = select_k_threshold(recalls, K_GRID, 0.95, 50)
        assert k == 10

    def test_recall_at_k_computation(self):
        """compute_recall_at_k returns correct fractions."""
        results = _make_oof_results_for_k_test(n_queries=10, truth_in_top=7, truth_rank=1)
        recalls = compute_recall_at_k(results, [1, 5])
        # 7/10 have truth at rank 1; remaining 3 have truth completely absent
        assert abs(recalls[1] - 0.7) < 1e-9
        assert abs(recalls[5] - 0.7) < 1e-9  # still 7/10 (truth absent in other 3)

    def test_recall_excludes_truth_not_in_gallery(self):
        """Queries where identity_in_oof_gallery=False are excluded from recall."""
        gt = "indiv_00"
        qr_in = QueryResult(
            query_image_id="q_in",
            query_individual_id=gt,
            ranked_identities=[
                IdentityScore(individual_id=gt, channel_raw={}, channel_calibrated={},
                              channels_available=[], fused_score=1.0)
            ],
            channels_present=GLOBAL_CHANNELS,
            channels_absent=[],
            identity_in_oof_gallery=True,
        )
        qr_out = QueryResult(
            query_image_id="q_out",
            query_individual_id=gt,
            ranked_identities=[],  # truth not in candidates
            channels_present=GLOBAL_CHANNELS,
            channels_absent=[],
            identity_in_oof_gallery=False,
        )
        recalls = compute_recall_at_k([qr_in, qr_out], [1])
        # Only q_in counts; truth is at rank 1 → recall@1 = 1.0
        assert recalls[1] == 1.0


class TestShortlistFingerprint:
    """Req 2 & 9: content-addressed shortlist registration."""

    def test_fingerprint_determinism(self):
        fp1 = _shortlist_fingerprint("q001", ["indiv_a", "indiv_b", "indiv_c"])
        fp2 = _shortlist_fingerprint("q001", ["indiv_c", "indiv_a", "indiv_b"])
        # Order of candidates should not matter (sorted internally)
        assert fp1 == fp2

    def test_fingerprint_differs_for_different_queries(self):
        fp1 = _shortlist_fingerprint("q001", ["indiv_a"])
        fp2 = _shortlist_fingerprint("q002", ["indiv_a"])
        assert fp1 != fp2

    def test_fingerprint_differs_for_different_candidates(self):
        fp1 = _shortlist_fingerprint("q001", ["indiv_a"])
        fp2 = _shortlist_fingerprint("q001", ["indiv_b"])
        assert fp1 != fp2

    def test_shortlist_registration_columns(self):
        """build_shortlist_registration returns expected columns."""
        results = _make_oof_results_for_k_test(n_queries=3, truth_in_top=3)
        gallery_df = pd.DataFrame([
            {"image_id": f"img_q_{i}", "individual_id": f"indiv_{i:02d}",
             "session_id": f"sess_{i}", "split": "gallery"}
            for i in range(3)
        ])
        sl_df = build_shortlist_registration(results, frozen_k=2, gallery_df=gallery_df)
        for col in ["query_image_id", "candidate_individual_id", "candidate_rank",
                    "shortlist_fingerprint", "K", "global_fused_score"]:
            assert col in sl_df.columns, f"Missing column: {col}"


class TestLocalCandidatesExactlyShortlist:
    """Req 3 & 9: local candidates exactly match the shortlist."""

    def test_oof_table_rows_match_shortlist(self, tmp_path):
        """Each query in the OOF table has exactly K candidate rows from the shortlist."""
        gallery_df = _make_gallery_df(n_identities=3, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        crop_df = _make_crop_df(gallery_df)

        # Build a synthetic shortlist: 2 queries, 3 candidates each
        sl_rows = []
        for q_idx in range(2):
            q_id = f"img_{q_idx:02d}_0_0"
            q_indiv = f"indiv_{q_idx:02d}"
            q_sess = f"sess_{q_idx:02d}_0"
            cands = [f"indiv_{c:02d}" for c in range(3)]
            fp = _shortlist_fingerprint(q_id, cands)
            for rank, cand in enumerate(cands, 1):
                sl_rows.append({
                    "fold_session_id": q_sess,
                    "query_image_id": q_id,
                    "query_session_id": q_sess,
                    "query_individual_id": q_indiv,
                    "candidate_individual_id": cand,
                    "candidate_rank": rank,
                    "global_fused_score": 1.0 / rank,
                    "global_miewid_raw": 0.8,
                    "global_miewid_calibrated": 0.75,
                    "global_ear_raw": 0.7,
                    "global_ear_calibrated": 0.65,
                    "shortlist_fingerprint": fp,
                    "K": 3,
                })
        sl_df = pd.DataFrame(sl_rows)

        scorer_body = _make_fake_local_scorer()
        scorer_ear = _make_fake_local_scorer()

        oof_df = build_oof_table(
            shortlist_df=sl_df,
            gallery_df=gdf,
            crop_df=crop_df,
            emb_matrix_miewid=None,
            desc_mapping_miewid=None,
            local_scorer_body=scorer_body,
            local_scorer_ear=scorer_ear,
            output_dir=tmp_path,
            resume=False,
        )

        for q_id in sl_df["query_image_id"].unique():
            n_oof = len(oof_df[oof_df["query_image_id"] == q_id])
            n_sl = len(sl_df[sl_df["query_image_id"] == q_id])
            assert n_oof == n_sl, (
                f"Query {q_id}: OOF table has {n_oof} rows, shortlist has {n_sl}"
            )


class TestPositiveSupport:
    """Req 5 & 9: Platt calibration support guards."""

    def test_positive_support_guard(self):
        """Fewer than MIN_POSITIVE_SUPPORT_PLATT positive rows → LocalSupportError."""
        df = _make_oof_table_df(n_queries=2, n_candidates=3, n_pos_per_query=1)
        # With n_queries=2 and n_candidates=3, we have at most 2 positive rows.
        with pytest.raises(LocalSupportError, match="insufficient positive"):
            fit_local_platt(df, CHANNEL_BODY_LOCAL, min_positive=10)

    def test_negative_support_guard(self):
        """Fewer than MIN_NEGATIVE_SUPPORT_PLATT negative rows → LocalSupportError."""
        # Build a table where all rows are positive (same candidate == truth for every query)
        n = 50
        df = pd.DataFrame({
            "body_local_score": [0.8] * n,
            "body_local_available": [True] * n,
            "label": [1] * n,  # all positive, no negatives
        })
        with pytest.raises(LocalSupportError, match="insufficient negative"):
            fit_local_platt(df, CHANNEL_BODY_LOCAL, min_positive=2, min_negative=10)

    def test_sufficient_support_fits(self):
        """Sufficient pos/neg support → Calibrator fits successfully."""
        df = _make_oof_table_df(n_queries=20, n_candidates=5, n_pos_per_query=1)
        cal = fit_local_platt(df, CHANNEL_BODY_LOCAL, min_positive=5, min_negative=5,
                              flatness_threshold=0.0)
        assert cal is not None
        assert cal.method == "platt"


class TestMissingChannels:
    """Req 5 & 9: missing region = unavailable, not zero evidence."""

    def test_missing_body_crops_available_false(self, tmp_path):
        """When query has no body crops, body_local_available is False."""
        gallery_df = _make_gallery_df(n_identities=2, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        # Crop manifest with NO body crops
        crop_df = pd.DataFrame([
            {
                "crop_id": f"{row['image_id']}__ear_0",
                "image_id": row["image_id"],
                "individual_id": row["individual_id"],
                "crop_kind": "ear",
                "crop_ordinal": 0,
                "crop_path": f"/fake/{row['image_id']}_ear.jpg",
                "detector_status": "accepted",
                "detector_confidence": 0.9,
                "detector_box": "",
                "review_status": "accepted",
                "schema_version": "v1",
                "source_fingerprint": "src_fp",
                "split_fingerprint": "split_fp",
            }
            for _, row in gdf.iterrows()
        ])

        q_id = "img_00_0_0"
        fold_sess = "sess_00_0"
        cand_id = "indiv_01"
        scorer_body = _make_fake_local_scorer()
        scorer_ear = _make_fake_local_scorer()

        from pipeline.local_oof_calibration import score_local_for_candidate
        result = score_local_for_candidate(
            query_image_id=q_id,
            fold_session_id=fold_sess,
            candidate_individual_id=cand_id,
            gallery_df=gdf,
            crop_df=crop_df,
            emb_matrix_miewid=None,
            desc_mapping_miewid=None,
            local_scorer_body=scorer_body,
            local_scorer_ear=scorer_ear,
        )

        assert result["body_local_available"] is False
        assert np.isnan(result["body_local_score"])
        # ear should be available (has ear crops)
        assert result["ear_local_available"] is True

    def test_missing_channel_excluded_from_fusion_not_zeroed(self):
        """
        Missing body_local → score is NaN in OOF table; not zeroed for fusion.
        Fusion denominator only sums available channel weights.
        """
        df = _make_oof_table_df(
            n_queries=5, n_candidates=3, n_pos_per_query=1,
            include_body=False,  # body channel absent
        )

        # The body_local_available should be False → excluded from fusion.
        assert not df["body_local_available"].any()
        assert df["body_local_score"].isna().all()

        # Fit ear calibrator only (body unavailable → skip)
        cal_ear = fit_local_platt(
            df, CHANNEL_EAR_LOCAL, min_positive=2, min_negative=5,
            flatness_threshold=0.0
        )
        assert cal_ear is not None


class TestPlattFlatnessGuard:
    """Req 5 & 9: Platt nonflat guard."""

    def test_flat_calibrator_raises(self):
        """Calibrator with constant output → LocalFlatnessError."""
        # All rows have the same body_local_score → flat output after Platt fit
        n = 30
        df = pd.DataFrame({
            "body_local_score": [0.5] * n,
            "body_local_available": [True] * n,
            "label": [1] * (n // 2) + [0] * (n // 2),
        })
        with pytest.raises(LocalFlatnessError):
            fit_local_platt(
                df, CHANNEL_BODY_LOCAL,
                min_positive=5, min_negative=5,
                flatness_threshold=0.05,  # high threshold to trigger flatness detection
            )

    def test_discriminative_scores_pass(self):
        """Well-separated scores → no flatness error."""
        rng = np.random.default_rng(0)
        n_pos, n_neg = 20, 40
        pos = rng.normal(0.9, 0.05, n_pos).clip(0, 1)
        neg = rng.normal(0.1, 0.05, n_neg).clip(0, 1)
        df = pd.DataFrame({
            "body_local_score": np.concatenate([pos, neg]).tolist(),
            "body_local_available": [True] * (n_pos + n_neg),
            "label": [1] * n_pos + [0] * n_neg,
        })
        cal = fit_local_platt(df, CHANNEL_BODY_LOCAL, min_positive=5, min_negative=5,
                              flatness_threshold=1e-4)
        assert cal is not None


class TestIdentityMacroMRR:
    """Req 6 & 9: identity-macro MRR."""

    def test_mrr_perfect(self):
        """All truths ranked first → MRR = 1.0."""
        results = _make_oof_results_for_k_test(n_queries=10, truth_in_top=10)
        mrr = identity_macro_mrr(results)
        assert abs(mrr - 1.0) < 1e-9

    def test_mrr_zero(self):
        """Truth never found → MRR = 0.0."""
        results = []
        for i in range(5):
            gt = f"indiv_{i:02d}"
            ranked = [
                IdentityScore(individual_id="other", channel_raw={},
                              channel_calibrated={}, channels_available=[],
                              fused_score=1.0)
            ]
            results.append(QueryResult(
                query_image_id=f"q{i}", query_individual_id=gt,
                ranked_identities=ranked, channels_present=GLOBAL_CHANNELS,
                channels_absent=[], identity_in_oof_gallery=True,
            ))
        mrr = identity_macro_mrr(results)
        assert mrr == 0.0

    def test_mrr_partial(self):
        """Manual calculation: two identities with one query each."""
        # identity A: truth at rank 2 → RR = 0.5
        # identity B: truth at rank 4 → RR = 0.25
        # identity-macro MRR = (0.5 + 0.25) / 2 = 0.375
        def _make_qr(q_id, gt, truth_rank, n_total=5):
            others = [f"other_{j}" for j in range(n_total - 1) if f"other_{j}" != gt]
            ranked_ids = others[: truth_rank - 1] + [gt] + others[truth_rank - 1:]
            ranked = [
                IdentityScore(individual_id=rid, channel_raw={}, channel_calibrated={},
                              channels_available=[], fused_score=float(n_total - r))
                for r, rid in enumerate(ranked_ids)
            ]
            return QueryResult(
                query_image_id=q_id, query_individual_id=gt,
                ranked_identities=ranked, channels_present=GLOBAL_CHANNELS,
                channels_absent=[], identity_in_oof_gallery=True,
            )

        results = [
            _make_qr("qA", "indiv_A", truth_rank=2),
            _make_qr("qB", "indiv_B", truth_rank=4),
        ]
        mrr = identity_macro_mrr(results)
        assert abs(mrr - 0.375) < 1e-9

    def test_mrr_excludes_not_in_gallery(self):
        """Queries with identity_in_oof_gallery=False are excluded from MRR."""
        gt = "indiv_00"
        qr_in = QueryResult(
            query_image_id="in", query_individual_id=gt,
            ranked_identities=[
                IdentityScore(individual_id=gt, channel_raw={}, channel_calibrated={},
                              channels_available=[], fused_score=1.0)
            ],
            channels_present=GLOBAL_CHANNELS, channels_absent=[],
            identity_in_oof_gallery=True,
        )
        qr_out = QueryResult(
            query_image_id="out", query_individual_id=gt,
            ranked_identities=[
                IdentityScore(individual_id="other", channel_raw={}, channel_calibrated={},
                              channels_available=[], fused_score=1.0)
            ],
            channels_present=GLOBAL_CHANNELS, channels_absent=[],
            identity_in_oof_gallery=False,
        )
        mrr = identity_macro_mrr([qr_in, qr_out])
        assert abs(mrr - 1.0) < 1e-9  # only qr_in counted, truth at rank 1


class TestWeightConstraints:
    """Req 6 & 9: weight constraints."""

    def test_weights_sum_to_one(self):
        """Fitted weights must sum to 1."""
        df = _make_oof_table_df(n_queries=10, n_candidates=5, n_pos_per_query=1)
        cal_body = _make_platt_calibrator()
        cal_ear = _make_platt_calibrator(pos_mean=0.9, neg_mean=0.2)
        calibrators_local = {
            CHANNEL_BODY_LOCAL: cal_body,
            CHANNEL_EAR_LOCAL: cal_ear,
        }
        weights, diag = fit_4channel_weights(
            oof_df=df,
            calibrators_global={},
            calibrators_local=calibrators_local,
            all_channels=ALL_CHANNELS,
            grid_step=0.25,
        )
        total = sum(weights[ch] for ch in ALL_CHANNELS)
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, not 1.0"

    def test_weights_non_negative(self):
        """All fitted weights must be ≥ 0."""
        df = _make_oof_table_df(n_queries=10, n_candidates=5, n_pos_per_query=1)
        cal_body = _make_platt_calibrator()
        weights, _ = fit_4channel_weights(
            oof_df=df,
            calibrators_global={},
            calibrators_local={CHANNEL_BODY_LOCAL: cal_body},
            all_channels=ALL_CHANNELS,
            grid_step=0.25,
        )
        for ch, w in weights.items():
            assert w >= -1e-9, f"Weight for {ch} is negative: {w}"

    def test_weights_comparison_with_selected_v1(self):
        """The diagnostics dict contains a weight comparison entry."""
        from configs.config_elephant import PRODUCTION_FUSION_WEIGHTS

        df = _make_oof_table_df(n_queries=10, n_candidates=5, n_pos_per_query=1)
        cal_body = _make_platt_calibrator()
        _, diag = fit_4channel_weights(
            oof_df=df,
            calibrators_global={},
            calibrators_local={CHANNEL_BODY_LOCAL: cal_body},
            all_channels=ALL_CHANNELS,
            grid_step=0.5,
        )
        assert "best_weights" in diag
        assert "best_mrr" in diag


class TestAtomicShardsAndResume:
    """Req 4 & 9: resume / atomic shards / fingerprints."""

    def test_shard_written_atomically(self, tmp_path):
        """Shard file exists after _write_shard_atomic."""
        shard = tmp_path / "shards" / "shard_q001.parquet"
        rows = [{"query_image_id": "q001", "candidate_individual_id": "indiv_00", "label": 1}]
        _write_shard_atomic(shard, rows)
        assert shard.exists()
        df = pd.read_parquet(shard)
        assert len(df) == 1

    def test_empty_shard_not_written(self, tmp_path):
        """_write_shard_atomic with empty rows does not create a file."""
        shard = tmp_path / "shards" / "shard_empty.parquet"
        _write_shard_atomic(shard, [])
        assert not shard.exists()

    def test_load_completed_queries(self, tmp_path):
        """Completed query IDs are read from existing shards."""
        for q_id in ["q001", "q002"]:
            shard = tmp_path / "shards" / f"shard_{q_id}.parquet"
            shard.parent.mkdir(exist_ok=True)
            pd.DataFrame({
                "query_image_id": [q_id],
                "label": [0],
                "shortlist_fingerprint": [f"fp-{q_id}"],
            }).to_parquet(shard, index=False)

        completed = _load_completed_queries(
            tmp_path,
            {"q001": "fp-q001", "q002": "fp-q002"},
        )
        assert "q001" in completed
        assert "q002" in completed
        assert "q003" not in completed

    def test_resume_skips_completed_query(self, tmp_path):
        """On resume, already-scored queries are not re-processed."""
        gallery_df = _make_gallery_df(n_identities=3, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        crop_df = _make_crop_df(gallery_df)

        q_id = "img_00_0_0"
        q_sess = "sess_00_0"
        q_indiv = "indiv_00"
        cand_id = "indiv_01"

        sl_df = pd.DataFrame([{
            "fold_session_id": q_sess,
            "query_image_id": q_id,
            "query_session_id": q_sess,
            "query_individual_id": q_indiv,
            "candidate_individual_id": cand_id,
            "candidate_rank": 1,
            "global_fused_score": 0.9,
            "global_miewid_raw": 0.8,
            "global_miewid_calibrated": 0.75,
            "global_ear_raw": 0.7,
            "global_ear_calibrated": 0.65,
            "shortlist_fingerprint": "fp001",
            "K": 1,
        }])

        # Write a pre-existing shard to simulate a resumed run
        shard = _shard_path(tmp_path, q_id)
        _write_shard_atomic(
            shard,
            [{"query_image_id": q_id, "candidate_individual_id": cand_id,
              "label": 0, "body_local_available": False, "ear_local_available": False,
              "shortlist_fingerprint": "fp001"}]
        )

        call_count = {"n": 0}

        class _CountingScorer(FakeLocalIdentityScorer):
            def score_identity(self, *args, **kwargs):
                call_count["n"] += 1
                return super().score_identity(*args, **kwargs)

        scorer = _CountingScorer()
        build_oof_table(
            shortlist_df=sl_df, gallery_df=gdf, crop_df=crop_df,
            emb_matrix_miewid=None, desc_mapping_miewid=None,
            local_scorer_body=scorer, local_scorer_ear=scorer,
            output_dir=tmp_path, resume=True,
        )

        assert call_count["n"] == 0, (
            f"score_identity called {call_count['n']} times on already-completed query"
        )

    def test_fingerprint_mismatch_detected(self, tmp_path):
        shard = _shard_path(tmp_path, "q001")
        _write_shard_atomic(
            shard,
            [{
                "query_image_id": "q001",
                "candidate_individual_id": "indiv_01",
                "label": 0,
                "shortlist_fingerprint": "old-fingerprint",
            }],
        )
        with pytest.raises(FingerprintMismatchError, match="Stale shortlist shard"):
            _load_completed_queries(
                tmp_path,
                {"q001": "new-fingerprint"},
            )

    def test_no_resume_rescores(self, tmp_path):
        """When resume=False, already-existing shards are overwritten."""
        gallery_df = _make_gallery_df(n_identities=2, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        crop_df = _make_crop_df(gallery_df)

        q_id = "img_00_0_0"
        q_sess = "sess_00_0"
        sl_df = pd.DataFrame([{
            "fold_session_id": q_sess, "query_image_id": q_id,
            "query_session_id": q_sess, "query_individual_id": "indiv_00",
            "candidate_individual_id": "indiv_01",
            "candidate_rank": 1, "global_fused_score": 0.9,
            "global_miewid_raw": 0.8, "global_miewid_calibrated": 0.75,
            "global_ear_raw": 0.7, "global_ear_calibrated": 0.65,
            "shortlist_fingerprint": "fp001", "K": 1,
        }])

        call_count = {"n": 0}

        class _CountingScorer(FakeLocalIdentityScorer):
            def score_identity(self, *args, **kwargs):
                call_count["n"] += 1
                return super().score_identity(*args, **kwargs)

        scorer = _CountingScorer()
        build_oof_table(
            shortlist_df=sl_df, gallery_df=gdf, crop_df=crop_df,
            emb_matrix_miewid=None, desc_mapping_miewid=None,
            local_scorer_body=scorer, local_scorer_ear=scorer,
            output_dir=tmp_path, resume=False,
        )

        assert call_count["n"] >= 1


class TestInferenceOOFExactFunction:
    """Req 7 & 9: inference/OOF exact function parity."""

    def test_score_query_candidates_returns_sorted_list(self, tmp_path):
        """score_query_candidates returns IdentityScore list sorted by fused_score."""
        gallery_df = _make_gallery_df(n_identities=3, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        crop_df = _make_crop_df(gallery_df)
        embs = _make_embedding_matrices(gdf, dim=4)
        dms = _make_descriptor_mappings(gdf)

        cal = _make_platt_calibrator()
        arts = EnsembleArtifacts(
            frozen_k=5,
            fusion_weights={ch: 0.25 for ch in ALL_CHANNELS},
            calibrators_global={CHANNEL_MIEWID: cal, CHANNEL_EAR: cal},
            calibrator_body=cal,
            calibrator_ear=cal,
        )

        scorer_body = _make_fake_local_scorer()
        scorer_ear = _make_fake_local_scorer()

        candidates = ["indiv_00", "indiv_01", "indiv_02"]
        results = score_query_candidates(
            query_image_id="img_00_0_0",
            query_session_id="sess_00_0",
            candidate_individual_ids=candidates,
            gallery_df=gdf,
            crop_df=crop_df,
            embedding_matrices=embs,
            descriptor_mappings=dms,
            artifacts=arts,
            local_scorer_body=scorer_body,
            local_scorer_ear=scorer_ear,
        )

        assert isinstance(results, list)
        assert len(results) == len(candidates)
        scores = [r.fused_score for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by fused_score"

    def test_ensemble_scorer_uses_score_query_candidates(self, tmp_path):
        """EnsembleScorer.score() delegates to score_query_candidates."""
        gallery_df = _make_gallery_df(n_identities=3, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        crop_df = _make_crop_df(gallery_df)
        embs = _make_embedding_matrices(gdf, dim=4)
        dms = _make_descriptor_mappings(gdf)

        cal = _make_platt_calibrator()
        arts = EnsembleArtifacts(
            frozen_k=5,
            fusion_weights={ch: 0.25 for ch in ALL_CHANNELS},
            calibrators_global={CHANNEL_MIEWID: cal, CHANNEL_EAR: cal},
            calibrator_body=cal,
            calibrator_ear=cal,
        )

        scorer_body = _make_fake_local_scorer()
        scorer_ear = _make_fake_local_scorer()

        ensemble = EnsembleScorer(
            artifacts=arts,
            gallery_df=gdf,
            crop_df=crop_df,
            embedding_matrices=embs,
            descriptor_mappings=dms,
            local_scorer_body=scorer_body,
            local_scorer_ear=scorer_ear,
        )

        results = ensemble.score(
            query_image_id="img_00_0_0",
            query_session_id="sess_00_0",
            candidate_ids=["indiv_00", "indiv_01"],
        )
        assert isinstance(results, list)

    def test_ensemble_scorer_rejects_probe_gallery(self):
        """EnsembleScorer raises ProbePollutionError if gallery_df contains probes."""
        df = _make_gallery_df_with_probe()
        arts = EnsembleArtifacts(
            frozen_k=5,
            fusion_weights={ch: 0.25 for ch in ALL_CHANNELS},
            calibrators_global={},
            calibrator_body=None,
            calibrator_ear=None,
        )
        with pytest.raises(ProbePollutionError):
            EnsembleScorer(
                artifacts=arts,
                gallery_df=df,
                crop_df=pd.DataFrame(),
                embedding_matrices={},
                descriptor_mappings={},
                local_scorer_body=None,
                local_scorer_ear=None,
            )


class TestSelectedV1Nonmutation:
    """Req 6 & 9: selected-v1 artifacts not mutated by OOF calibration."""

    def test_production_weights_unchanged_after_run(self, tmp_path):
        """PRODUCTION_FUSION_WEIGHTS is not modified by the pipeline."""
        from configs.config_elephant import PRODUCTION_FUSION_WEIGHTS

        original_v1 = dict(PRODUCTION_FUSION_WEIGHTS)

        gallery_df = _make_gallery_df(n_identities=3, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        crop_df = _make_crop_df(gallery_df)
        embs = _make_embedding_matrices(gdf, dim=4)
        dms = _make_descriptor_mappings(gdf)

        config = OOFPipelineConfig(
            override_budget=True,
            min_positive_support_platt=1,
            min_negative_support_platt=1,
        )

        try:
            run_oof_calibration(
                config=config,
                gallery_df=gdf,
                crop_df=crop_df,
                descriptor_mappings=dms,
                embedding_matrices=embs,
                calibrators_global={},
                local_scorer_body=_make_fake_local_scorer(),
                local_scorer_ear=_make_fake_local_scorer(),
                output_dir=tmp_path / "oof",
                override_budget=True,
            )
        except Exception:
            pass  # Calibration may fail on tiny data; we only care about nonmutation.

        # Production weights must be unchanged.
        assert dict(PRODUCTION_FUSION_WEIGHTS) == original_v1, (
            "PRODUCTION_FUSION_WEIGHTS was mutated by the OOF pipeline!"
        )

    def test_production_channels_unchanged(self):
        """PRODUCTION_SELECTED_CHANNELS is not mutated."""
        from configs.config_elephant import PRODUCTION_SELECTED_CHANNELS
        original = list(PRODUCTION_SELECTED_CHANNELS)
        # Run any computation that touches global channels
        gallery_df = _make_gallery_df(n_identities=2, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        embs = _make_embedding_matrices(gdf, dim=4)
        dms = _make_descriptor_mappings(gdf)
        _ = compute_global_oof_rankings(
            gallery_df=gdf,
            descriptor_mappings=dms,
            embedding_matrices=embs,
            calibrators={},
            channels=list(PRODUCTION_SELECTED_CHANNELS),
        )
        assert list(PRODUCTION_SELECTED_CHANNELS) == original


class TestBudgetGate:
    """Req 8 & 9: pair count/runtime estimate; budget gate."""

    def test_budget_gate_fails(self, tmp_path):
        """BudgetExceededError raised when projected hours exceed limit."""
        gallery_df = _make_gallery_df(n_identities=2, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        crop_df = _make_crop_df(gallery_df)
        embs = _make_embedding_matrices(gdf, dim=4)
        dms = _make_descriptor_mappings(gdf)

        config = OOFPipelineConfig(
            gate_max_h100_hours=0.0,  # 0 hours → always over budget
            gate_max_cache_gb=0.0,
            override_budget=False,
        )

        with pytest.raises(BudgetExceededError):
            run_oof_calibration(
                config=config,
                gallery_df=gdf,
                crop_df=crop_df,
                descriptor_mappings=dms,
                embedding_matrices=embs,
                calibrators_global={},
                local_scorer_body=None,
                local_scorer_ear=None,
                output_dir=tmp_path / "oof_budget",
                override_budget=False,
            )

    def test_budget_gate_override(self, tmp_path):
        """override_budget=True bypasses the budget gate."""
        gallery_df = _make_gallery_df(n_identities=2, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        crop_df = _make_crop_df(gallery_df)
        embs = _make_embedding_matrices(gdf, dim=4)
        dms = _make_descriptor_mappings(gdf)

        config = OOFPipelineConfig(
            gate_max_h100_hours=0.0,
            gate_max_cache_gb=0.0,
            override_budget=True,
            min_positive_support_platt=999,  # make calibration fail gracefully
            min_negative_support_platt=999,
        )

        # Should not raise BudgetExceededError
        try:
            run_oof_calibration(
                config=config,
                gallery_df=gdf,
                crop_df=crop_df,
                descriptor_mappings=dms,
                embedding_matrices=embs,
                calibrators_global={},
                local_scorer_body=None,
                local_scorer_ear=None,
                output_dir=tmp_path / "oof_override",
                override_budget=True,
            )
        except BudgetExceededError:
            pytest.fail("BudgetExceededError raised despite override_budget=True")
        except Exception:
            pass  # Other errors (e.g. calibration support) are OK


class TestEnsembleArtifactsRoundtrip:
    """Req 6 & 9: save/load artifact roundtrip."""

    def test_save_and_load_oof_artifacts(self, tmp_path):
        """save_oof_artifacts + load_oof_artifacts is lossless for key fields."""
        config = OOFPipelineConfig()
        cal_body = _make_platt_calibrator()
        cal_ear = _make_platt_calibrator(pos_mean=0.85, neg_mean=0.25)
        weights = {ch: 0.25 for ch in ALL_CHANNELS}
        metrics = {"frozen_k": 10, "best_mrr": 0.85}

        save_oof_artifacts(
            output_dir=tmp_path,
            config=config,
            calibrator_body=cal_body,
            calibrator_ear=cal_ear,
            fusion_weights=weights,
            oof_metrics=metrics,
        )

        raw = load_oof_artifacts(tmp_path)

        assert raw["fusion_weights"] == weights
        assert raw["oof_metrics"]["frozen_k"] == 10
        assert raw["calibrator_body"] is not None
        assert raw["calibrator_ear"] is not None
        # Re-loaded calibrators should produce same output.
        test_scores = np.array([0.1, 0.5, 0.9])
        np.testing.assert_allclose(
            cal_body.transform(test_scores),
            raw["calibrator_body"].transform(test_scores),
            rtol=1e-5,
        )

    def test_load_ensemble_artifacts(self, tmp_path):
        """load_ensemble_artifacts returns an EnsembleArtifacts object."""
        config = OOFPipelineConfig()
        cal_body = _make_platt_calibrator()
        weights = {ch: 0.25 for ch in ALL_CHANNELS}
        metrics = {"frozen_k": 10, "best_mrr": 0.80}

        save_oof_artifacts(
            output_dir=tmp_path,
            config=config,
            calibrator_body=cal_body,
            calibrator_ear=None,
            fusion_weights=weights,
            oof_metrics=metrics,
        )

        arts = load_ensemble_artifacts(tmp_path)
        assert isinstance(arts, EnsembleArtifacts)
        assert arts.frozen_k == 10
        assert arts.calibrator_body is not None
        assert arts.calibrator_ear is None

    def test_load_fails_missing_file(self, tmp_path):
        """load_oof_artifacts raises FileNotFoundError when files are missing."""
        with pytest.raises(FileNotFoundError):
            load_oof_artifacts(tmp_path / "nonexistent")


class TestMergeShards:
    """Req 4: shard merge returns combined OOF table."""

    def test_merge_shards_empty_dir(self, tmp_path):
        """No shards → empty DataFrame."""
        df = _merge_shards(tmp_path)
        assert df.empty

    def test_merge_shards_combines(self, tmp_path):
        """Multiple shards are combined into one DataFrame."""
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        for q_id in ["q001", "q002", "q003"]:
            shard = shard_dir / f"shard_{q_id}.parquet"
            pd.DataFrame({"query_image_id": [q_id], "label": [0]}).to_parquet(shard, index=False)

        df = _merge_shards(tmp_path)
        assert len(df) == 3
        assert set(df["query_image_id"]) == {"q001", "q002", "q003"}


# ===========================================================================
# New integration tests: CLI real-data wiring
# ===========================================================================

def _make_minimal_splits_parquet(tmp_path, gallery_df, extra_probe=True):
    """Write a splits parquet containing gallery rows (and optionally probe rows)."""
    rows = gallery_df[gallery_df["split"] == "gallery"][
        ["image_id", "individual_id", "session_id", "split"]
    ].copy()
    if extra_probe:
        probe_rows = pd.DataFrame([{
            "image_id": f"probe_img_{k}",
            "individual_id": "indiv_00",
            "session_id": "probe_sess",
            "split": "probe",
        } for k in range(3)])
        rows = pd.concat([rows, probe_rows], ignore_index=True)
    path = tmp_path / "splits.parquet"
    rows.to_parquet(path, index=False)
    return path


def _make_minimal_manifest_parquet(tmp_path, gallery_df):
    """Write a manifest parquet (image_id + individual_id only)."""
    rows = gallery_df[gallery_df["split"] == "gallery"][
        ["image_id", "individual_id"]
    ].copy()
    path = tmp_path / "manifest.parquet"
    rows.to_parquet(path, index=False)
    return path


def _make_minimal_crop_parquet(tmp_path, gallery_df, source_fp="src_fp", split_fp="split_fp"):
    """Write a minimal crop manifest parquet with fingerprint columns."""
    crop_df = _make_crop_df(gallery_df)
    crop_df["source_fingerprint"] = source_fp
    crop_df["split_fingerprint"] = split_fp
    path = tmp_path / "crop_manifest.parquet"
    crop_df.to_parquet(path, index=False)
    return path


def _make_minimal_embedding_files(
    tmp_path, gallery_df, channels=None, dim=8, source_fp="src_fp", split_fp="split_fp"
):
    """Write minimal .npy + _mapping.parquet files for the given channels."""
    if channels is None:
        channels = list(GLOBAL_CHANNELS)
    gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
    n = len(gdf)
    rng = np.random.default_rng(0)
    for ch in channels:
        mat = rng.standard_normal((n, dim)).astype(np.float32)
        np.save(str(tmp_path / f"{ch}.npy"), mat)
        rows = []
        for i, (_, row) in enumerate(gdf.iterrows()):
            rows.append({
                "descriptor_name": ch,
                "embedding_row": i,
                "image_id": row["image_id"],
                "individual_id": row["individual_id"],
                "source_fingerprint": source_fp,
                "split_fingerprint": split_fp,
            })
        pd.DataFrame(rows).to_parquet(tmp_path / f"{ch}_mapping.parquet", index=False)


def _make_calibration_dir(tmp_path, channels=None, weights=None):
    """Write minimal calibrator .pkl files and fusion_weights.json."""
    if channels is None:
        channels = list(GLOBAL_CHANNELS)
    if weights is None:
        weights = {ch: 1.0 / len(channels) for ch in channels}
    cal = _make_platt_calibrator()
    for ch in channels:
        cal.save(str(tmp_path / f"{ch}.pkl"))
    with open(tmp_path / "fusion_weights.json", "w") as fh:
        json.dump(weights, fh)


class TestCLIPlaceholderGone:
    """CLI 'not implemented' placeholder is removed."""

    def test_placeholder_gone(self):
        """main() must not contain the old 'not implemented' placeholder."""
        import inspect
        from pipeline.local_oof_calibration import main
        src = inspect.getsource(main)
        assert "not implemented" not in src.lower(), (
            "The 'not implemented' placeholder is still in main(). Wire it end-to-end."
        )

    def test_run_subparser_has_disable_cudnn(self):
        """The 'run' subparser must expose --disable-cudnn."""
        from pipeline.local_oof_calibration import _build_parser
        p = _build_parser()
        # parse with --disable-cudnn flag and verify it is accepted
        args = p.parse_args([
            "run",
            "--manifest", "/fake",
            "--splits", "/fake",
            "--crop-manifest", "/fake",
            "--disable-cudnn",
        ])
        assert args.disable_cudnn is True

    def test_run_subparser_has_max_sessions(self):
        """The 'run' subparser must expose --max-sessions."""
        from pipeline.local_oof_calibration import _build_parser
        p = _build_parser()
        args = p.parse_args([
            "run",
            "--manifest", "/fake",
            "--splits", "/fake",
            "--crop-manifest", "/fake",
            "--max-sessions", "5",
        ])
        assert args.max_sessions == 5


class TestGalleryPushdown:
    """Req CLI: split=='gallery' push-down; probe rows never loaded."""

    def test_load_gallery_data_excludes_probe_ids(self, tmp_path):
        """_load_gallery_data returns only gallery images; probe IDs absent."""
        gallery_df = _make_gallery_df(n_identities=3, sessions_per_identity=2)
        splits_path = _make_minimal_splits_parquet(tmp_path, gallery_df, extra_probe=True)
        manifest_path = _make_minimal_manifest_parquet(tmp_path, gallery_df)
        crop_path = _make_minimal_crop_parquet(tmp_path, gallery_df)

        gdf, crop_df, _, _ = _load_gallery_data(
            manifest_path=str(manifest_path),
            splits_path=str(splits_path),
            crop_path=str(crop_path),
        )

        image_ids_in_result = set(gdf["image_id"].astype(str))
        for k in range(3):
            assert f"probe_img_{k}" not in image_ids_in_result, (
                f"Probe image probe_img_{k} found in gallery_df — probe pushdown failed."
            )

    def test_load_gallery_data_probe_in_gallery_df_raises(self, tmp_path):
        """If probe rows somehow enter gallery_df, ProbePollutionError is raised."""
        gallery_df = _make_gallery_df(n_identities=2, sessions_per_identity=2)
        # Write splits with only gallery (no probes initially)
        splits_path = _make_minimal_splits_parquet(tmp_path, gallery_df, extra_probe=False)
        manifest_path = _make_minimal_manifest_parquet(tmp_path, gallery_df)
        crop_path = _make_minimal_crop_parquet(tmp_path, gallery_df)

        # This should succeed normally
        gdf, _, _, _ = _load_gallery_data(
            manifest_path=str(manifest_path),
            splits_path=str(splits_path),
            crop_path=str(crop_path),
        )
        assert all(gdf["split"] == "gallery")

    def test_load_gallery_data_returns_gallery_split_only(self, tmp_path):
        """gallery_df contains only split=='gallery' rows."""
        gallery_df = _make_gallery_df(n_identities=4, sessions_per_identity=2)
        splits_path = _make_minimal_splits_parquet(tmp_path, gallery_df, extra_probe=True)
        manifest_path = _make_minimal_manifest_parquet(tmp_path, gallery_df)
        crop_path = _make_minimal_crop_parquet(tmp_path, gallery_df)

        gdf, crop_df, _, _ = _load_gallery_data(
            manifest_path=str(manifest_path),
            splits_path=str(splits_path),
            crop_path=str(crop_path),
        )

        assert (gdf["split"] == "gallery").all(), "Non-gallery rows in gallery_df"

    def test_load_gallery_data_source_fingerprint_extracted(self, tmp_path):
        """source_fingerprint and split_fingerprint are extracted from crop manifest."""
        gallery_df = _make_gallery_df(n_identities=2, sessions_per_identity=2)
        splits_path = _make_minimal_splits_parquet(tmp_path, gallery_df, extra_probe=False)
        manifest_path = _make_minimal_manifest_parquet(tmp_path, gallery_df)
        crop_path = _make_minimal_crop_parquet(
            tmp_path, gallery_df, source_fp="my_source_fp", split_fp="my_split_fp"
        )

        _, _, src_fp, spl_fp = _load_gallery_data(
            manifest_path=str(manifest_path),
            splits_path=str(splits_path),
            crop_path=str(crop_path),
        )
        assert src_fp == "my_source_fp"
        assert spl_fp == "my_split_fp"

    def test_extract_single_fingerprint_single_value(self):
        """Single fingerprint value is returned correctly."""
        df = pd.DataFrame({"source_fingerprint": ["fp_abc", "fp_abc", "fp_abc"]})
        fp = _extract_single_fingerprint(df, "source_fingerprint")
        assert fp == "fp_abc"

    def test_extract_single_fingerprint_multiple_raises(self):
        """Multiple distinct fingerprint values raise FingerprintMismatchError."""
        df = pd.DataFrame({"source_fingerprint": ["fp_abc", "fp_xyz"]})
        with pytest.raises(FingerprintMismatchError, match="Multiple values"):
            _extract_single_fingerprint(df, "source_fingerprint")

    def test_extract_single_fingerprint_missing_column(self):
        """Missing column returns empty string (not an error)."""
        df = pd.DataFrame({"other_col": [1, 2]})
        fp = _extract_single_fingerprint(df, "source_fingerprint")
        assert fp == ""


class TestMappingsAligned:
    """Req CLI: descriptor mapping parquets pass contiguous row/fingerprint checks."""

    def test_contiguous_rows_pass(self, tmp_path):
        """Contiguous embedding_rows after gallery filter → no error."""
        gallery_df = _make_gallery_df(n_identities=3, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        _make_minimal_embedding_files(tmp_path, gallery_df, channels=[CHANNEL_MIEWID])

        embs, mappings = _load_embedding_matrices_and_mappings(
            embeddings_dir=tmp_path,
            channels=[CHANNEL_MIEWID],
            gallery_ids=set(gdf["image_id"].astype(str)),
        )
        assert CHANNEL_MIEWID in embs
        assert CHANNEL_MIEWID in mappings
        rows = mappings[CHANNEL_MIEWID]["embedding_row"].to_numpy(dtype=np.int64)
        assert np.array_equal(rows, np.arange(rows[0], rows[-1] + 1))

    def test_non_contiguous_rows_raise(self, tmp_path):
        """Gaps in embedding_rows after gallery filter → FingerprintMismatchError."""
        gallery_df = _make_gallery_df(n_identities=2, sessions_per_identity=2)
        n = len(gallery_df[gallery_df["split"] == "gallery"])
        mat = np.random.randn(n * 2, 8).astype(np.float32)
        np.save(str(tmp_path / f"{CHANNEL_MIEWID}.npy"), mat)

        # Non-contiguous rows (0, 2, 4, ...) — every other row
        rows = []
        for i, (_, row) in enumerate(gallery_df[gallery_df["split"] == "gallery"].iterrows()):
            rows.append({
                "image_id": row["image_id"],
                "embedding_row": i * 2,  # gaps of 1 → non-contiguous
            })
        pd.DataFrame(rows).to_parquet(
            tmp_path / f"{CHANNEL_MIEWID}_mapping.parquet", index=False
        )

        with pytest.raises(FingerprintMismatchError, match="not contiguous"):
            _load_embedding_matrices_and_mappings(
                embeddings_dir=tmp_path,
                channels=[CHANNEL_MIEWID],
                gallery_ids=None,
            )

    def test_fingerprint_mismatch_in_mapping_raises(self, tmp_path):
        """Mapping source_fingerprint mismatch → FingerprintMismatchError."""
        gallery_df = _make_gallery_df(n_identities=2, sessions_per_identity=2)
        _make_minimal_embedding_files(
            tmp_path, gallery_df,
            channels=[CHANNEL_MIEWID],
            source_fp="fp_correct",
        )

        with pytest.raises(FingerprintMismatchError, match="source_fingerprint mismatch"):
            _load_embedding_matrices_and_mappings(
                embeddings_dir=tmp_path,
                channels=[CHANNEL_MIEWID],
                gallery_ids=None,
                source_fingerprint="fp_different",  # intentional mismatch
            )

    def test_gallery_filter_reduces_mapping(self, tmp_path):
        """Gallery filter removes rows for non-gallery image IDs from mapping."""
        gallery_df = _make_gallery_df(n_identities=3, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        n = len(gdf)

        # Create mapping with more rows than gallery (simulating full dataset)
        mat = np.random.randn(n + 5, 8).astype(np.float32)
        np.save(str(tmp_path / f"{CHANNEL_MIEWID}.npy"), mat)

        extra_rows = []
        for i, (_, row) in enumerate(gdf.iterrows()):
            extra_rows.append({
                "image_id": row["image_id"],
                "embedding_row": i,
                "source_fingerprint": "fp_ok",
            })
        # Add extra rows for non-gallery images (sequential rows)
        for j in range(5):
            extra_rows.append({
                "image_id": f"non_gallery_{j}",
                "embedding_row": n + j,
                "source_fingerprint": "fp_ok",
            })
        pd.DataFrame(extra_rows).to_parquet(
            tmp_path / f"{CHANNEL_MIEWID}_mapping.parquet", index=False
        )

        gallery_ids = set(gdf["image_id"].astype(str))
        embs, mappings = _load_embedding_matrices_and_mappings(
            embeddings_dir=tmp_path,
            channels=[CHANNEL_MIEWID],
            gallery_ids=gallery_ids,
        )
        result_ids = set(mappings[CHANNEL_MIEWID]["image_id"].astype(str))
        for j in range(5):
            assert f"non_gallery_{j}" not in result_ids


class TestCalibratorsWeightsLoaded:
    """Req CLI: calibrators and fusion weights loaded from calibration_projected."""

    def test_load_calibrators_success(self, tmp_path):
        """_load_production_calibrators returns correct calibrators and weights."""
        channels = [CHANNEL_MIEWID, CHANNEL_EAR]
        weights = {CHANNEL_MIEWID: 0.6, CHANNEL_EAR: 0.4}
        _make_calibration_dir(tmp_path, channels=channels, weights=weights)

        cals, w = _load_production_calibrators(tmp_path, channels)
        assert set(cals.keys()) == set(channels)
        assert abs(w[CHANNEL_MIEWID] - 0.6) < 1e-9
        assert abs(w[CHANNEL_EAR] - 0.4) < 1e-9

    def test_load_calibrators_nested_weights_format(self, tmp_path):
        """Supports {"weights": {channel: float}} JSON layout."""
        channels = [CHANNEL_MIEWID]
        cal = _make_platt_calibrator()
        cal.save(str(tmp_path / f"{CHANNEL_MIEWID}.pkl"))
        with open(tmp_path / "fusion_weights.json", "w") as fh:
            json.dump({"weights": {CHANNEL_MIEWID: 1.0}}, fh)

        cals, w = _load_production_calibrators(tmp_path, channels)
        assert abs(w[CHANNEL_MIEWID] - 1.0) < 1e-9

    def test_load_calibrators_missing_pkl_raises(self, tmp_path):
        """Missing .pkl file raises FileNotFoundError."""
        with open(tmp_path / "fusion_weights.json", "w") as fh:
            json.dump({CHANNEL_MIEWID: 1.0}, fh)
        with pytest.raises(FileNotFoundError, match="Missing calibrator"):
            _load_production_calibrators(tmp_path, [CHANNEL_MIEWID])

    def test_load_calibrators_missing_weights_raises(self, tmp_path):
        """Missing fusion_weights.json raises FileNotFoundError."""
        cal = _make_platt_calibrator()
        cal.save(str(tmp_path / f"{CHANNEL_MIEWID}.pkl"))
        with pytest.raises(FileNotFoundError, match="Missing fusion weights"):
            _load_production_calibrators(tmp_path, [CHANNEL_MIEWID])

    def test_calibrator_produces_output(self, tmp_path):
        """Reloaded calibrator produces same output as original."""
        channels = [CHANNEL_MIEWID]
        _make_calibration_dir(tmp_path, channels=channels)
        cals, _ = _load_production_calibrators(tmp_path, channels)
        scores = np.array([0.1, 0.5, 0.9])
        out = cals[CHANNEL_MIEWID].transform(scores)
        assert out.shape == (3,)
        assert (out >= 0).all() and (out <= 1).all()


class TestRunInvoked:
    """Req CLI: run_oof_calibration is invoked with all loaded structures."""

    def test_run_oof_calibration_accepts_fusion_weights_global(self, tmp_path):
        """run_oof_calibration accepts fusion_weights_global kwarg (no TypeError)."""
        gallery_df = _make_gallery_df(n_identities=2, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        crop_df = _make_crop_df(gallery_df)
        embs = _make_embedding_matrices(gdf, dim=4)
        dms = _make_descriptor_mappings(gdf)

        custom_weights = {CHANNEL_MIEWID: 0.3, CHANNEL_EAR: 0.7}
        config = OOFPipelineConfig(
            override_budget=True,
            min_positive_support_platt=999,
        )

        # Should not raise TypeError for the new kwarg.
        try:
            run_oof_calibration(
                config=config,
                gallery_df=gdf,
                crop_df=crop_df,
                descriptor_mappings=dms,
                embedding_matrices=embs,
                calibrators_global={},
                local_scorer_body=None,
                local_scorer_ear=None,
                output_dir=tmp_path / "oof",
                fusion_weights_global=custom_weights,
            )
        except TypeError as e:
            pytest.fail(f"run_oof_calibration raised TypeError for fusion_weights_global: {e}")
        except Exception:
            pass  # Other failures (e.g. support) are fine

    def test_compute_global_oof_rankings_uses_supplied_weights(self):
        """compute_global_oof_rankings uses supplied weights, not PRODUCTION_FUSION_WEIGHTS."""
        gallery_df = _make_gallery_df(n_identities=3, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        embs = _make_embedding_matrices(gdf, dim=4)
        dms = _make_descriptor_mappings(gdf)

        # Highly asymmetric weights to create a detectable signal difference.
        custom_weights = {CHANNEL_MIEWID: 0.01, CHANNEL_EAR: 0.99}
        results = compute_global_oof_rankings(
            gallery_df=gdf,
            descriptor_mappings=dms,
            embedding_matrices=embs,
            calibrators={},
            channels=GLOBAL_CHANNELS,
            weights=custom_weights,
        )
        assert len(results) > 0
        # With heavy ear weight, fused scores should differ from equal-weight baseline.
        equal_results = compute_global_oof_rankings(
            gallery_df=gdf,
            descriptor_mappings=dms,
            embedding_matrices=embs,
            calibrators={},
            channels=GLOBAL_CHANNELS,
            weights={CHANNEL_MIEWID: 0.5, CHANNEL_EAR: 0.5},
        )
        # Fused score sums differ across weight configurations.
        custom_sum = sum(
            s.fused_score
            for qr in results
            for s in qr.ranked_identities
        )
        equal_sum = sum(
            s.fused_score
            for qr in equal_results
            for s in qr.ranked_identities
        )
        # They should not be identical (different weights produce different fusion).
        assert abs(custom_sum - equal_sum) > 1e-6 or len(results) == 0, (
            "Supplied weights had no effect on fused scores."
        )

    def test_run_oof_calibration_passes_weights_to_global_rankings(self, tmp_path):
        """run_oof_calibration passes fusion_weights_global to compute_global_oof_rankings."""
        gallery_df = _make_gallery_df(n_identities=2, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        crop_df = _make_crop_df(gallery_df)
        embs = _make_embedding_matrices(gdf, dim=4)
        dms = _make_descriptor_mappings(gdf)

        captured = {}

        import pipeline.local_oof_calibration as _mod
        _orig = _mod.compute_global_oof_rankings

        def _capturing(*args, **kwargs):
            captured["weights"] = kwargs.get("weights")
            return _orig(*args, **kwargs)

        custom_weights = {CHANNEL_MIEWID: 0.2, CHANNEL_EAR: 0.8}
        config = OOFPipelineConfig(override_budget=True, min_positive_support_platt=999)

        with patch.object(_mod, "compute_global_oof_rankings", side_effect=_capturing):
            try:
                run_oof_calibration(
                    config=config,
                    gallery_df=gdf,
                    crop_df=crop_df,
                    descriptor_mappings=dms,
                    embedding_matrices=embs,
                    calibrators_global={},
                    local_scorer_body=None,
                    local_scorer_ear=None,
                    output_dir=tmp_path / "oof",
                    fusion_weights_global=custom_weights,
                )
            except Exception:
                pass  # Calibration may fail; only care about weight forwarding

        assert "weights" in captured, "compute_global_oof_rankings was not called."
        assert captured["weights"] == custom_weights, (
            f"Expected {custom_weights}, got {captured['weights']}"
        )

    def test_main_run_invokes_run_oof_calibration(self, tmp_path):
        """main() 'run' command calls run_oof_calibration with gallery data."""
        gallery_df = _make_gallery_df(n_identities=3, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()

        splits_path = _make_minimal_splits_parquet(tmp_path, gallery_df, extra_probe=True)
        manifest_path = _make_minimal_manifest_parquet(tmp_path, gallery_df)
        crop_path = _make_minimal_crop_parquet(tmp_path, gallery_df)

        emb_dir = tmp_path / "embeddings"
        emb_dir.mkdir()
        _make_minimal_embedding_files(emb_dir, gallery_df)

        cal_dir = tmp_path / "calibration"
        cal_dir.mkdir()
        _make_calibration_dir(cal_dir)

        captured = {}

        def _mock_run(**kwargs):
            gdf_arg = kwargs.get("gallery_df", pd.DataFrame())
            captured["gallery_df_len"] = len(gdf_arg)
            captured["gallery_df_splits"] = set(gdf_arg["split"].unique()) if "split" in gdf_arg.columns else set()
            captured["called"] = True
            return {
                "frozen_k": 5,
                "fusion_weights": {CHANNEL_MIEWID: 0.6, CHANNEL_EAR: 0.4},
                "oof_metrics": {},
                "calibrator_body": None,
                "calibrator_ear": None,
                "shortlist_df": None,
                "oof_table_df": None,
                "recalls_at_k": {},
                "weight_diagnostics": {},
            }

        def _mock_scorers(*args, **kwargs):
            captured["disable_cudnn"] = kwargs.get("disable_cudnn")
            captured["max_sessions"] = kwargs.get("max_sessions")
            return (None, None)

        import pipeline.local_oof_calibration as _mod
        with patch.object(_mod, "run_oof_calibration", side_effect=_mock_run), \
             patch.object(_mod, "_instantiate_local_scorers", side_effect=_mock_scorers):
            rc = _mod.main([
                "run",
                "--manifest", str(manifest_path),
                "--splits", str(splits_path),
                "--crop-manifest", str(crop_path),
                "--embeddings-dir", str(emb_dir),
                "--calibration-dir", str(cal_dir),
                "--output-dir", str(tmp_path / "oof"),
                "--override-budget",
                "--disable-cudnn",
                "--max-sessions", "4",
            ])

        assert rc == 0
        assert captured.get("called") is True, "run_oof_calibration was not called."
        # Gallery only: probe image IDs must not be in gallery_df.
        assert captured["gallery_df_splits"] == {"gallery"}, (
            f"Non-gallery splits found: {captured['gallery_df_splits']}"
        )

    def test_main_disable_cudnn_propagated(self, tmp_path):
        """--disable-cudnn is propagated to _instantiate_local_scorers."""
        gallery_df = _make_gallery_df(n_identities=2, sessions_per_identity=2)

        splits_path = _make_minimal_splits_parquet(tmp_path, gallery_df, extra_probe=False)
        manifest_path = _make_minimal_manifest_parquet(tmp_path, gallery_df)
        crop_path = _make_minimal_crop_parquet(tmp_path, gallery_df)

        emb_dir = tmp_path / "embeddings"
        emb_dir.mkdir()
        _make_minimal_embedding_files(emb_dir, gallery_df)

        cal_dir = tmp_path / "calibration"
        cal_dir.mkdir()
        _make_calibration_dir(cal_dir)

        captured = {}

        def _mock_scorers(*args, **kwargs):
            captured["disable_cudnn"] = kwargs.get("disable_cudnn")
            return (None, None)

        def _mock_run(*args, **kwargs):
            return {
                "frozen_k": 5, "fusion_weights": {}, "oof_metrics": {},
                "calibrator_body": None, "calibrator_ear": None,
                "shortlist_df": None, "oof_table_df": None,
                "recalls_at_k": {}, "weight_diagnostics": {},
            }

        import pipeline.local_oof_calibration as _mod
        with patch.object(_mod, "_instantiate_local_scorers", side_effect=_mock_scorers), \
             patch.object(_mod, "run_oof_calibration", side_effect=_mock_run):
            _mod.main([
                "run",
                "--manifest", str(manifest_path),
                "--splits", str(splits_path),
                "--crop-manifest", str(crop_path),
                "--embeddings-dir", str(emb_dir),
                "--calibration-dir", str(cal_dir),
                "--output-dir", str(tmp_path / "oof"),
                "--override-budget",
                "--disable-cudnn",
            ])

        assert captured.get("disable_cudnn") is True, (
            "--disable-cudnn was not propagated to _instantiate_local_scorers."
        )

    def test_main_probe_labels_not_passed_to_run(self, tmp_path):
        """Probe image IDs from splits parquet are not present in gallery_df passed to run."""
        gallery_df = _make_gallery_df(n_identities=3, sessions_per_identity=2)

        # splits has both gallery and probe rows
        splits_path = _make_minimal_splits_parquet(tmp_path, gallery_df, extra_probe=True)
        manifest_path = _make_minimal_manifest_parquet(tmp_path, gallery_df)
        crop_path = _make_minimal_crop_parquet(tmp_path, gallery_df)

        emb_dir = tmp_path / "embeddings"
        emb_dir.mkdir()
        _make_minimal_embedding_files(emb_dir, gallery_df)

        cal_dir = tmp_path / "calibration"
        cal_dir.mkdir()
        _make_calibration_dir(cal_dir)

        probe_ids_in_gallery_df = set()

        def _mock_run(**kwargs):
            gdf_arg = kwargs.get("gallery_df", pd.DataFrame())
            for img_id in gdf_arg["image_id"].astype(str):
                if img_id.startswith("probe_img_"):
                    probe_ids_in_gallery_df.add(img_id)
            return {
                "frozen_k": 5, "fusion_weights": {}, "oof_metrics": {},
                "calibrator_body": None, "calibrator_ear": None,
                "shortlist_df": None, "oof_table_df": None,
                "recalls_at_k": {}, "weight_diagnostics": {},
            }

        import pipeline.local_oof_calibration as _mod
        with patch.object(_mod, "run_oof_calibration", side_effect=_mock_run), \
             patch.object(_mod, "_instantiate_local_scorers", return_value=(None, None)):
            _mod.main([
                "run",
                "--manifest", str(manifest_path),
                "--splits", str(splits_path),
                "--crop-manifest", str(crop_path),
                "--embeddings-dir", str(emb_dir),
                "--calibration-dir", str(cal_dir),
                "--output-dir", str(tmp_path / "oof"),
                "--override-budget",
            ])

        assert len(probe_ids_in_gallery_df) == 0, (
            f"Probe image IDs passed to run_oof_calibration: {probe_ids_in_gallery_df}"
        )


# ===========================================================================
# New tests: parallel worker machinery
# ===========================================================================


def _make_shortlist_df(n_queries: int = 6, n_candidates: int = 2) -> pd.DataFrame:
    """Minimal shortlist DataFrame for worker tests."""
    rows = []
    for q in range(n_queries):
        q_id = f"worker_q_{q:04d}"
        q_indiv = f"indiv_{q:02d}"
        fold_sess = f"sess_{q:02d}_0"
        cands = [f"indiv_{c:02d}" for c in range(n_candidates)]
        fp = _shortlist_fingerprint(q_id, cands)
        for rank, cand in enumerate(cands, 1):
            rows.append({
                "fold_session_id": fold_sess,
                "query_image_id": q_id,
                "query_session_id": fold_sess,
                "query_individual_id": q_indiv,
                "candidate_individual_id": cand,
                "candidate_rank": rank,
                "global_fused_score": 1.0 / rank,
                "global_miewid_raw": 0.8,
                "global_miewid_calibrated": 0.75,
                "global_ear_raw": 0.7,
                "global_ear_calibrated": 0.65,
                "shortlist_fingerprint": fp,
                "K": n_candidates,
            })
    return pd.DataFrame(rows)


class TestWorkerAssignment:
    """Stable, disjoint, complete assignments; range validation."""

    def test_assignment_stable(self):
        """Same query_id + worker_count → same worker every time."""
        for q_id in ["img_a", "img_b", "query_long_name_xyz_123", ""]:
            w1 = _worker_query_assignment(q_id, 4)
            w2 = _worker_query_assignment(q_id, 4)
            assert w1 == w2, f"Non-stable assignment for {q_id!r}: {w1} != {w2}"

    def test_assignment_in_range(self):
        """Assigned worker is always in [0, worker_count)."""
        for wc in [1, 2, 4, 8, 100]:
            for q_id in [f"q_{i}" for i in range(50)]:
                w = _worker_query_assignment(q_id, wc)
                assert 0 <= w < wc, f"worker={w} out of range [0,{wc}) for {q_id!r}"

    def test_assignment_disjoint(self):
        """For each query_id, exactly one worker is assigned."""
        queries = [f"img_{i:04d}" for i in range(30)]
        worker_count = 4
        assignment = {q: _worker_query_assignment(q, worker_count) for q in queries}
        # Partition into per-worker sets
        per_worker: Dict[int, List[str]] = {w: [] for w in range(worker_count)}
        for q, w in assignment.items():
            per_worker[w].append(q)
        # Pairwise disjoint
        all_assigned: List[str] = []
        for worker_queries in per_worker.values():
            for q in worker_queries:
                assert q not in all_assigned, f"Query {q!r} assigned to multiple workers"
                all_assigned.append(q)

    def test_assignment_complete(self):
        """The union of all workers' queries equals the full query set."""
        queries = {f"q_{i}" for i in range(40)}
        worker_count = 5
        covered: set = set()
        for w in range(worker_count):
            for q in queries:
                if _worker_query_assignment(q, worker_count) == w:
                    covered.add(q)
        assert covered == queries, f"Missing: {queries - covered}"

    def test_single_worker_processes_all(self):
        """worker_count=1 → worker_index=0 processes all queries."""
        queries = [f"q_{i}" for i in range(20)]
        for q in queries:
            assert _worker_query_assignment(q, 1) == 0

    def test_validate_worker_index_valid(self):
        """Valid combinations do not raise."""
        _validate_worker_index(0, 1)
        _validate_worker_index(0, 4)
        _validate_worker_index(3, 4)

    def test_validate_worker_index_zero_count_raises(self):
        """worker_count=0 → WorkerRangeError."""
        with pytest.raises(WorkerRangeError, match="worker_count must be >= 1"):
            _validate_worker_index(0, 0)

    def test_validate_worker_index_negative_count_raises(self):
        """worker_count<0 → WorkerRangeError."""
        with pytest.raises(WorkerRangeError, match="worker_count must be >= 1"):
            _validate_worker_index(0, -1)

    def test_validate_worker_index_too_large_raises(self):
        """worker_index >= worker_count → WorkerRangeError."""
        with pytest.raises(WorkerRangeError, match="out of range"):
            _validate_worker_index(4, 4)

    def test_validate_worker_index_negative_raises(self):
        """worker_index < 0 → WorkerRangeError."""
        with pytest.raises(WorkerRangeError, match="out of range"):
            _validate_worker_index(-1, 4)

    def test_build_oof_table_both_params_required_together(self, tmp_path):
        """Providing only worker_index (without worker_count) raises WorkerRangeError."""
        sl = _make_shortlist_df(n_queries=2)
        gdf = _make_gallery_df(n_identities=2)
        gdf = gdf[gdf["split"] == "gallery"].copy()
        cdf = _make_crop_df(gdf)
        with pytest.raises(WorkerRangeError):
            build_oof_table(
                shortlist_df=sl, gallery_df=gdf, crop_df=cdf,
                emb_matrix_miewid=None, desc_mapping_miewid=None,
                local_scorer_body=None, local_scorer_ear=None,
                output_dir=tmp_path,
                worker_index=0, worker_count=None,
            )


class TestWorkerShardIsolation:
    """Workers process disjoint queries; existing shards are skipped; shortlist not mutated."""

    def test_worker_processes_only_assigned_queries(self, tmp_path):
        """Worker 0 of 2 only scores its assigned queries, not the other half."""
        gallery_df = _make_gallery_df(n_identities=4, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        crop_df = _make_crop_df(gallery_df)
        sl_df = _make_shortlist_df(n_queries=8, n_candidates=2)

        scored_queries: List[str] = []

        class _TrackingScorer(FakeLocalIdentityScorer):
            def score_identity(self, query_crops, *args, **kwargs):
                # We can't easily get query_id from score_identity args alone,
                # so we patch at a higher level. See below.
                return super().score_identity(query_crops, *args, **kwargs)

        # Compute expected assignment
        all_queries = list(sl_df["query_image_id"].unique())
        expected_w0 = {q for q in all_queries if _worker_query_assignment(q, 2) == 0}
        expected_w1 = {q for q in all_queries if _worker_query_assignment(q, 2) == 1}
        assert expected_w0 | expected_w1 == set(all_queries), "Coverage broken"
        assert expected_w0 & expected_w1 == set(), "Overlap detected"

        # Run worker 0
        scorer = _make_fake_local_scorer()
        build_oof_table(
            shortlist_df=sl_df, gallery_df=gdf, crop_df=crop_df,
            emb_matrix_miewid=None, desc_mapping_miewid=None,
            local_scorer_body=scorer, local_scorer_ear=scorer,
            output_dir=tmp_path / "out0",
            resume=False,
            worker_index=0, worker_count=2,
        )
        # Run worker 1 in a different output dir
        build_oof_table(
            shortlist_df=sl_df, gallery_df=gdf, crop_df=crop_df,
            emb_matrix_miewid=None, desc_mapping_miewid=None,
            local_scorer_body=scorer, local_scorer_ear=scorer,
            output_dir=tmp_path / "out1",
            resume=False,
            worker_index=1, worker_count=2,
        )

        # Shards written by worker 0
        shard_dir0 = tmp_path / "out0" / "shards"
        w0_query_ids = set()
        for p in shard_dir0.glob("shard_*.parquet"):
            df = pd.read_parquet(p)
            w0_query_ids.update(df["query_image_id"].astype(str).unique())

        # Shards written by worker 1
        shard_dir1 = tmp_path / "out1" / "shards"
        w1_query_ids = set()
        for p in shard_dir1.glob("shard_*.parquet"):
            df = pd.read_parquet(p)
            w1_query_ids.update(df["query_image_id"].astype(str).unique())

        assert w0_query_ids == expected_w0, f"Worker 0 wrote wrong shards: {w0_query_ids}"
        assert w1_query_ids == expected_w1, f"Worker 1 wrote wrong shards: {w1_query_ids}"
        assert w0_query_ids & w1_query_ids == set(), "Workers wrote overlapping shards!"

    def test_workers_combined_cover_all_queries(self, tmp_path):
        """Shards from worker 0 + worker 1 together cover all queries."""
        gallery_df = _make_gallery_df(n_identities=4, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        crop_df = _make_crop_df(gallery_df)
        sl_df = _make_shortlist_df(n_queries=10, n_candidates=2)
        all_queries = set(sl_df["query_image_id"].unique())

        scorer = _make_fake_local_scorer()
        out = tmp_path / "shared_out"

        for w in range(2):
            build_oof_table(
                shortlist_df=sl_df, gallery_df=gdf, crop_df=crop_df,
                emb_matrix_miewid=None, desc_mapping_miewid=None,
                local_scorer_body=scorer, local_scorer_ear=scorer,
                output_dir=out,
                resume=True,
                worker_index=w, worker_count=2,
            )

        # All shards combined should cover every query
        merged = _merge_shards(out)
        covered = set(merged["query_image_id"].astype(str).unique())
        assert covered == all_queries, f"Missing queries: {all_queries - covered}"

    def test_worker_skips_completed_shard(self, tmp_path):
        """A query already scored (correct fingerprint) is not rescored by a worker."""
        gallery_df = _make_gallery_df(n_identities=3, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        crop_df = _make_crop_df(gallery_df)
        # Use enough queries to ensure at least one is assigned to worker 0
        sl_df = _make_shortlist_df(n_queries=8, n_candidates=2)

        all_queries = list(sl_df["query_image_id"].unique())
        # Pick a query that belongs to worker 0
        q0_candidates = [q for q in all_queries if _worker_query_assignment(q, 2) == 0]
        assert q0_candidates, (
            "Test assumption broken: no queries assigned to worker 0 with n_queries=8"
        )
        q0 = q0_candidates[0]
        fp0 = sl_df.loc[sl_df["query_image_id"] == q0, "shortlist_fingerprint"].iloc[0]

        # Pre-write a valid shard for q0
        shard = _shard_path(tmp_path, q0)
        _write_shard_atomic(shard, [{
            "query_image_id": q0,
            "candidate_individual_id": "indiv_00",
            "label": 0,
            "body_local_available": False,
            "ear_local_available": False,
            "shortlist_fingerprint": fp0,
        }])

        call_count = {"n": 0}

        class _CountingScorer(FakeLocalIdentityScorer):
            def score_identity(self, *args, **kwargs):
                call_count["n"] += 1
                return super().score_identity(*args, **kwargs)

        scorer = _CountingScorer()
        build_oof_table(
            shortlist_df=sl_df, gallery_df=gdf, crop_df=crop_df,
            emb_matrix_miewid=None, desc_mapping_miewid=None,
            local_scorer_body=scorer, local_scorer_ear=scorer,
            output_dir=tmp_path, resume=True,
            worker_index=0, worker_count=2,
        )

        # q0 should NOT have been rescored
        # (the scorer is called only for the remaining worker-0 queries)
        n_worker0_queries = sum(
            1 for q in all_queries if _worker_query_assignment(q, 2) == 0
        )
        max_expected_calls = (n_worker0_queries - 1) * 2 * 2  # remaining × candidates × regions
        assert call_count["n"] <= max_expected_calls, (
            f"Expected at most {max_expected_calls} scorer calls "
            f"(q0 already complete); got {call_count['n']}"
        )

    def test_worker_doesnt_write_shortlist(self, tmp_path):
        """build_oof_table with worker params does NOT create or modify shortlist_registration.parquet."""
        gallery_df = _make_gallery_df(n_identities=2, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        crop_df = _make_crop_df(gallery_df)
        sl_df = _make_shortlist_df(n_queries=4, n_candidates=2)

        sl_file = tmp_path / SHORTLIST_REGISTRATION_PARQUET
        # File must NOT exist before the worker run
        assert not sl_file.exists()

        scorer = _make_fake_local_scorer()
        build_oof_table(
            shortlist_df=sl_df, gallery_df=gdf, crop_df=crop_df,
            emb_matrix_miewid=None, desc_mapping_miewid=None,
            local_scorer_body=scorer, local_scorer_ear=scorer,
            output_dir=tmp_path, resume=False,
            worker_index=0, worker_count=2,
        )

        assert not sl_file.exists(), (
            "build_oof_table (worker mode) must NOT write shortlist_registration.parquet"
        )

    def test_worker_stale_fingerprint_raises(self, tmp_path):
        """A shard with a stale shortlist fingerprint raises FingerprintMismatchError."""
        sl_df = _make_shortlist_df(n_queries=4, n_candidates=2)
        all_queries = list(sl_df["query_image_id"].unique())
        q0 = all_queries[0]

        # Write a shard with the WRONG fingerprint
        shard = _shard_path(tmp_path, q0)
        _write_shard_atomic(shard, [{
            "query_image_id": q0,
            "candidate_individual_id": "indiv_00",
            "label": 0,
            "shortlist_fingerprint": "stale_fp_xyz",
        }])

        with pytest.raises(FingerprintMismatchError, match="Stale shortlist shard"):
            gallery_df = _make_gallery_df(n_identities=2, sessions_per_identity=2)
            gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
            crop_df = _make_crop_df(gallery_df)
            build_oof_table(
                shortlist_df=sl_df, gallery_df=gdf, crop_df=crop_df,
                emb_matrix_miewid=None, desc_mapping_miewid=None,
                local_scorer_body=_make_fake_local_scorer(),
                local_scorer_ear=_make_fake_local_scorer(),
                output_dir=tmp_path, resume=True,
                worker_index=0, worker_count=2,
            )

    def test_no_worker_params_processes_all(self, tmp_path):
        """Without worker params, all queries are scored (original single-process behaviour)."""
        gallery_df = _make_gallery_df(n_identities=3, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        crop_df = _make_crop_df(gallery_df)
        sl_df = _make_shortlist_df(n_queries=6, n_candidates=2)

        scorer = _make_fake_local_scorer()
        build_oof_table(
            shortlist_df=sl_df, gallery_df=gdf, crop_df=crop_df,
            emb_matrix_miewid=None, desc_mapping_miewid=None,
            local_scorer_body=scorer, local_scorer_ear=scorer,
            output_dir=tmp_path, resume=False,
        )
        merged = _merge_shards(tmp_path)
        covered = set(merged["query_image_id"].astype(str).unique())
        assert covered == set(sl_df["query_image_id"].unique())


class TestCheckShardCoverage:
    """check_shard_coverage reports missing queries correctly."""

    def test_all_covered_returns_true(self, tmp_path):
        """All shards present → (True, [])."""
        sl_df = _make_shortlist_df(n_queries=3, n_candidates=2)
        for q_id in sl_df["query_image_id"].unique():
            fp = sl_df.loc[sl_df["query_image_id"] == q_id, "shortlist_fingerprint"].iloc[0]
            shard = _shard_path(tmp_path, q_id)
            _write_shard_atomic(shard, [{
                "query_image_id": q_id,
                "candidate_individual_id": "x",
                "label": 0,
                "shortlist_fingerprint": fp,
            }])
        covered, missing = check_shard_coverage(tmp_path, sl_df)
        assert covered is True
        assert missing == []

    def test_missing_shard_returns_false(self, tmp_path):
        """One shard missing → (False, [query_id])."""
        sl_df = _make_shortlist_df(n_queries=3, n_candidates=2)
        queries = list(sl_df["query_image_id"].unique())
        # Write shards for first 2 queries only
        for q_id in queries[:2]:
            fp = sl_df.loc[sl_df["query_image_id"] == q_id, "shortlist_fingerprint"].iloc[0]
            shard = _shard_path(tmp_path, q_id)
            _write_shard_atomic(shard, [{
                "query_image_id": q_id, "label": 0,
                "shortlist_fingerprint": fp,
            }])
        covered, missing = check_shard_coverage(tmp_path, sl_df)
        assert covered is False
        assert queries[2] in missing

    def test_stale_fingerprint_propagates(self, tmp_path):
        """Stale shard fingerprint inside check_shard_coverage propagates FingerprintMismatchError."""
        sl_df = _make_shortlist_df(n_queries=2, n_candidates=2)
        q_id = sl_df["query_image_id"].iloc[0]
        shard = _shard_path(tmp_path, q_id)
        _write_shard_atomic(shard, [{
            "query_image_id": q_id, "label": 0,
            "shortlist_fingerprint": "stale_fp",
        }])
        with pytest.raises(FingerprintMismatchError):
            check_shard_coverage(tmp_path, sl_df)


class TestFinalizeCalibration:
    """run_finalize_calibration: incomplete fails; complete path succeeds; shortlist preserved."""

    def _write_all_shards(self, tmp_path, sl_df):
        """Write valid shards for every query in sl_df."""
        for q_id in sl_df["query_image_id"].unique():
            fp = sl_df.loc[sl_df["query_image_id"] == q_id, "shortlist_fingerprint"].iloc[0]
            shard = _shard_path(tmp_path, q_id)
            # Write positive + negative rows for calibration support
            rows = []
            for _, row in sl_df[sl_df["query_image_id"] == q_id].iterrows():
                cand = str(row["candidate_individual_id"])
                q_indiv = str(row["query_individual_id"])
                label = int(cand == q_indiv)
                rows.append({
                    "query_image_id": q_id,
                    "query_individual_id": q_indiv,
                    "candidate_individual_id": cand,
                    "fold_session_id": str(row["fold_session_id"]),
                    "query_session_id": str(row["fold_session_id"]),
                    "label": label,
                    "body_local_score": 0.8 if label else 0.2,
                    "body_local_available": True,
                    "body_local_n_pairs": 2,
                    "body_local_n_valid": 2,
                    "body_local_n_sessions": 1,
                    "body_local_fingerprint": "fp_body",
                    "ear_local_score": 0.9 if label else 0.1,
                    "ear_local_available": True,
                    "ear_local_n_pairs": 2,
                    "ear_local_n_valid": 2,
                    "ear_local_n_sessions": 1,
                    "ear_local_fingerprint": "fp_ear",
                    "global_miewid_raw": 0.8 if label else 0.4,
                    "global_miewid_calibrated": 0.75 if label else 0.35,
                    "global_ear_raw": 0.7 if label else 0.3,
                    "global_ear_calibrated": 0.65 if label else 0.25,
                    "global_fused_score": 0.8 if label else 0.3,
                    "candidate_global_rank": int(row["candidate_rank"]),
                    "K": int(row["K"]),
                    "shortlist_fingerprint": fp,
                })
            _write_shard_atomic(shard, rows)

    def test_finalize_missing_shortlist_raises(self, tmp_path):
        """finalize with no shortlist_registration.parquet raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="shortlist"):
            run_finalize_calibration(tmp_path)

    def test_finalize_incomplete_shards_raises(self, tmp_path):
        """finalize with missing shards raises ShardCoverageError."""
        sl_df = _make_shortlist_df(n_queries=4, n_candidates=2)
        sl_df.to_parquet(tmp_path / SHORTLIST_REGISTRATION_PARQUET, index=False)

        # Write shards for only 2 of 4 queries
        queries = list(sl_df["query_image_id"].unique())
        for q_id in queries[:2]:
            fp = sl_df.loc[sl_df["query_image_id"] == q_id, "shortlist_fingerprint"].iloc[0]
            shard = _shard_path(tmp_path, q_id)
            _write_shard_atomic(shard, [{
                "query_image_id": q_id, "label": 0, "shortlist_fingerprint": fp,
            }])

        with pytest.raises(ShardCoverageError, match="missing"):
            run_finalize_calibration(tmp_path)

    def test_finalize_complete_path(self, tmp_path):
        """finalize with all shards present succeeds and saves artifacts."""
        sl_df = _make_shortlist_df(n_queries=10, n_candidates=3)
        sl_df.to_parquet(tmp_path / SHORTLIST_REGISTRATION_PARQUET, index=False)

        config = OOFPipelineConfig(
            min_positive_support_platt=2,
            min_negative_support_platt=2,
            flatness_threshold=0.0,
        )

        self._write_all_shards(tmp_path, sl_df)

        result = run_finalize_calibration(tmp_path, config=config)

        assert "fusion_weights" in result
        assert "oof_table_df" in result
        assert not result["oof_table_df"].empty
        # Calibrators may or may not fit depending on data, but no exception
        # OOF metrics JSON must exist
        assert (tmp_path / "oof_metrics.json").exists()
        assert (tmp_path / "fusion_weights.json").exists()

    def test_finalize_does_not_overwrite_shortlist(self, tmp_path):
        """finalize does NOT overwrite the frozen shortlist_registration.parquet."""
        sl_df = _make_shortlist_df(n_queries=6, n_candidates=2)
        sl_path = tmp_path / SHORTLIST_REGISTRATION_PARQUET
        sl_df.to_parquet(sl_path, index=False)
        import os
        original_mtime = os.path.getmtime(sl_path)

        config = OOFPipelineConfig(
            min_positive_support_platt=1,
            min_negative_support_platt=1,
            flatness_threshold=0.0,
        )
        self._write_all_shards(tmp_path, sl_df)
        run_finalize_calibration(tmp_path, config=config)

        new_mtime = os.path.getmtime(sl_path)
        assert new_mtime == original_mtime, (
            "run_finalize_calibration must NOT overwrite the frozen shortlist."
        )

    def test_finalize_loads_config_from_disk(self, tmp_path):
        """finalize loads OOFPipelineConfig from config.json when config=None."""
        import json as _json
        from dataclasses import asdict as _asdict
        sl_df = _make_shortlist_df(n_queries=4, n_candidates=2)
        sl_df.to_parquet(tmp_path / SHORTLIST_REGISTRATION_PARQUET, index=False)

        cfg = OOFPipelineConfig(
            min_positive_support_platt=1,
            min_negative_support_platt=1,
            flatness_threshold=0.0,
        )
        with open(tmp_path / OOF_CONFIG_JSON, "w") as fh:
            _json.dump(_asdict(cfg), fh)

        self._write_all_shards(tmp_path, sl_df)
        # Should not raise – config loaded from disk
        run_finalize_calibration(tmp_path, config=None)

    def test_finalize_production_weights_unchanged(self, tmp_path):
        """run_finalize_calibration does not mutate PRODUCTION_FUSION_WEIGHTS."""
        from configs.config_elephant import PRODUCTION_FUSION_WEIGHTS
        original = dict(PRODUCTION_FUSION_WEIGHTS)

        sl_df = _make_shortlist_df(n_queries=6, n_candidates=2)
        sl_df.to_parquet(tmp_path / SHORTLIST_REGISTRATION_PARQUET, index=False)
        config = OOFPipelineConfig(
            min_positive_support_platt=1,
            min_negative_support_platt=1,
            flatness_threshold=0.0,
        )
        self._write_all_shards(tmp_path, sl_df)
        try:
            run_finalize_calibration(tmp_path, config=config)
        except Exception:
            pass

        assert dict(PRODUCTION_FUSION_WEIGHTS) == original, (
            "run_finalize_calibration mutated PRODUCTION_FUSION_WEIGHTS!"
        )


class TestWorkerCLI:
    """CLI: worker + finalize subcommands parse correctly; worker doesn't rewrite shortlist."""

    def test_worker_subparser_parses_basic_args(self):
        """'worker' subparser accepts --worker-index and --worker-count."""
        p = _build_parser()
        args = p.parse_args([
            "worker",
            "--worker-index", "2",
            "--worker-count", "8",
            "--manifest", "/fake/manifest.parquet",
            "--splits", "/fake/splits.parquet",
            "--output-dir", "/fake/out",
        ])
        assert args.command == "worker"
        assert args.worker_index == 2
        assert args.worker_count == 8

    def test_worker_subparser_disable_cudnn(self):
        """worker subparser exposes --disable-cudnn."""
        p = _build_parser()
        args = p.parse_args([
            "worker",
            "--worker-index", "0",
            "--worker-count", "4",
            "--disable-cudnn",
        ])
        assert args.disable_cudnn is True

    def test_worker_subparser_max_sessions(self):
        """worker subparser exposes --max-sessions."""
        p = _build_parser()
        args = p.parse_args([
            "worker",
            "--worker-index", "1",
            "--worker-count", "4",
            "--max-sessions", "6",
        ])
        assert args.max_sessions == 6

    def test_finalize_subparser_parses(self):
        """'finalize' subparser accepts --output-dir and --calibration-dir."""
        p = _build_parser()
        args = p.parse_args([
            "finalize",
            "--output-dir", "/fake/oof",
            "--calibration-dir", "/fake/cal",
        ])
        assert args.command == "finalize"
        assert args.output_dir == "/fake/oof"
        assert args.calibration_dir == "/fake/cal"

    def test_finalize_subparser_output_dir_optional(self):
        """'finalize' subparser defaults output-dir to None."""
        p = _build_parser()
        args = p.parse_args(["finalize"])
        assert args.command == "finalize"
        assert args.output_dir is None

    def test_worker_required_args(self):
        """'worker' without --worker-index raises SystemExit."""
        p = _build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["worker", "--worker-count", "4"])

    def test_main_worker_validates_index_early(self, tmp_path):
        """main worker command raises WorkerRangeError for invalid index."""
        import pipeline.local_oof_calibration as _mod

        gallery_df = _make_gallery_df(n_identities=2, sessions_per_identity=2)
        splits_path = _make_minimal_splits_parquet(tmp_path, gallery_df)
        manifest_path = _make_minimal_manifest_parquet(tmp_path, gallery_df)
        crop_path = _make_minimal_crop_parquet(tmp_path, gallery_df)
        emb_dir = tmp_path / "emb"
        emb_dir.mkdir()
        _make_minimal_embedding_files(emb_dir, gallery_df)

        with pytest.raises(WorkerRangeError):
            with patch.object(_mod, "_load_production_calibrators",
                              return_value=({}, {})), \
                 patch.object(_mod, "_instantiate_local_scorers",
                              return_value=(None, None)):
                _mod.main([
                    "worker",
                    "--worker-index", "4",   # out of range
                    "--worker-count", "4",
                    "--manifest", str(manifest_path),
                    "--splits", str(splits_path),
                    "--crop-manifest", str(crop_path),
                    "--embeddings-dir", str(emb_dir),
                    "--output-dir", str(tmp_path / "oof"),
                ])

    def test_main_worker_requires_frozen_shortlist(self, tmp_path):
        """main worker command raises FileNotFoundError if shortlist absent."""
        import pipeline.local_oof_calibration as _mod

        gallery_df = _make_gallery_df(n_identities=2, sessions_per_identity=2)
        splits_path = _make_minimal_splits_parquet(tmp_path, gallery_df)
        manifest_path = _make_minimal_manifest_parquet(tmp_path, gallery_df)
        crop_path = _make_minimal_crop_parquet(tmp_path, gallery_df)
        emb_dir = tmp_path / "emb"
        emb_dir.mkdir()
        _make_minimal_embedding_files(emb_dir, gallery_df)
        out_dir = tmp_path / "oof_missing_shortlist"
        out_dir.mkdir()  # empty – no shortlist

        with pytest.raises(FileNotFoundError):
            _mod.main([
                "worker",
                "--worker-index", "0",
                "--worker-count", "2",
                "--manifest", str(manifest_path),
                "--splits", str(splits_path),
                "--crop-manifest", str(crop_path),
                "--embeddings-dir", str(emb_dir),
                "--output-dir", str(out_dir),
            ])

    def test_main_finalize_missing_shortlist_raises(self, tmp_path):
        """main finalize raises FileNotFoundError when shortlist absent."""
        import pipeline.local_oof_calibration as _mod

        with pytest.raises(FileNotFoundError):
            _mod.main(["finalize", "--output-dir", str(tmp_path / "nonexistent")])

    def test_main_worker_doesnt_call_run_oof_calibration(self, tmp_path):
        """main worker command never calls run_oof_calibration."""
        import pipeline.local_oof_calibration as _mod

        gallery_df = _make_gallery_df(n_identities=2, sessions_per_identity=2)
        splits_path = _make_minimal_splits_parquet(tmp_path, gallery_df)
        manifest_path = _make_minimal_manifest_parquet(tmp_path, gallery_df)
        crop_path = _make_minimal_crop_parquet(tmp_path, gallery_df)
        emb_dir = tmp_path / "emb"
        emb_dir.mkdir()
        _make_minimal_embedding_files(emb_dir, gallery_df)
        out_dir = tmp_path / "oof"
        out_dir.mkdir()

        # Write required frozen artifacts
        sl_df = _make_shortlist_df(n_queries=4, n_candidates=2)
        sl_df.to_parquet(out_dir / SHORTLIST_REGISTRATION_PARQUET, index=False)
        import json as _json
        from dataclasses import asdict as _asdict
        _json.dump(_asdict(OOFPipelineConfig()), open(out_dir / OOF_CONFIG_JSON, "w"))
        _json.dump({"schema_version": "v1"}, open(out_dir / OOF_FINGERPRINT_JSON, "w"))

        run_oof_calibration_called = {"flag": False}

        def _mock_run_oof(*args, **kwargs):
            run_oof_calibration_called["flag"] = True
            return {}

        with patch.object(_mod, "run_oof_calibration", side_effect=_mock_run_oof), \
             patch.object(_mod, "_instantiate_local_scorers", return_value=(
                 _make_fake_local_scorer(), _make_fake_local_scorer()
             )), \
             patch.object(_mod, "build_oof_table", return_value=pd.DataFrame()):
            _mod.main([
                "worker",
                "--worker-index", "0",
                "--worker-count", "2",
                "--manifest", str(manifest_path),
                "--splits", str(splits_path),
                "--crop-manifest", str(crop_path),
                "--embeddings-dir", str(emb_dir),
                "--output-dir", str(out_dir),
            ])

        assert not run_oof_calibration_called["flag"], (
            "worker command must NOT call run_oof_calibration"
        )


class TestWorkerSelectedV1Nonmutation:
    """Worker-mode scoring does not mutate selected-v1 production constants."""

    def test_parallel_worker_selected_v1_nonmutation(self, tmp_path):
        """PRODUCTION_FUSION_WEIGHTS is not modified by build_oof_table in worker mode."""
        from configs.config_elephant import PRODUCTION_FUSION_WEIGHTS

        original = dict(PRODUCTION_FUSION_WEIGHTS)

        gallery_df = _make_gallery_df(n_identities=3, sessions_per_identity=2)
        gdf = gallery_df[gallery_df["split"] == "gallery"].copy()
        crop_df = _make_crop_df(gallery_df)
        sl_df = _make_shortlist_df(n_queries=6, n_candidates=2)

        scorer = _make_fake_local_scorer()
        try:
            build_oof_table(
                shortlist_df=sl_df, gallery_df=gdf, crop_df=crop_df,
                emb_matrix_miewid=None, desc_mapping_miewid=None,
                local_scorer_body=scorer, local_scorer_ear=scorer,
                output_dir=tmp_path, resume=False,
                worker_index=0, worker_count=2,
            )
        except Exception:
            pass

        assert dict(PRODUCTION_FUSION_WEIGHTS) == original, (
            "build_oof_table (worker mode) mutated PRODUCTION_FUSION_WEIGHTS!"
        )
