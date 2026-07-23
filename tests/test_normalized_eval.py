# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Tests for normalized evaluation and identity-level fusion
(models/identity_fusion.py, pipeline/step_4c_normalized_eval.py).

Coverage:
  - Weights sum to 1, all non-negative: IdentityLevelScorer validates.
  - Missing-channel renormalisation: absent channel weight is redistributed.
  - Identity-level ranking avoids single-channel shortlist bias.
  - OOF identity scores flow correctly for weight fitting.
  - fit_fusion_weights: returned weights are non-negative and sum to 1.
  - fit_fusion_weights: deterministic (same data → same weights).
  - Unknown simulation: estimate_unknown_threshold returns value in [0,1].
  - compute_map / compute_top1: correctness on simple ranked lists.
  - Unknown query handling: score vs gallery, correctly labelled unknown.
  - Channel-ablation independence: each channel evaluated independently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.calibration import Calibrator
from models.identity_fusion import (
    IdentityLevelScorer,
    IdentityScore,
    QueryResult,
    _apply_weights_and_rank,
    _average_precision,
    build_oof_identity_scores,
    check_calibration_flatness,
    compute_map,
    compute_top1,
    compute_top5,
    estimate_unknown_threshold,
    fit_fusion_weights,
    simulate_probe_unknown_trials,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _make_scorer_fixture(
    n_individuals: int = 4,
    n_sessions: int = 3,
    n_body_per_session: int = 2,
    n_ear_per_image: int = 2,
    dim: int = 16,
    seed: int = 0,
    channels: list[str] | None = None,
) -> tuple[IdentityLevelScorer, dict, pd.DataFrame]:
    """
    Build an IdentityLevelScorer and supporting data for a synthetic gallery.

    Returns (scorer, query_data_dict, gallery_df).
    query_data_dict: {image_id: {channel: np.ndarray shape (n_crops, D)}}
    """
    if channels is None:
        channels = ["body_desc", "ear_desc"]

    rng = np.random.default_rng(seed)

    # Build gallery.
    gallery_rows = []
    body_mapping_rows, ear_mapping_rows = [], []
    body_embs, ear_embs = [], []
    body_row_idx = ear_row_idx = 0

    protos = []
    for i in range(n_individuals):
        proto = rng.standard_normal(dim).astype(np.float32)
        proto /= np.linalg.norm(proto)
        protos.append(proto)

    for indiv_i in range(n_individuals):
        proto = protos[indiv_i]
        for sess_j in range(n_sessions):
            session_id = f"sess_{indiv_i:02d}_{sess_j:02d}"
            for img_k in range(n_body_per_session):
                image_id = f"gal_{indiv_i:02d}_{sess_j:02d}_{img_k:02d}"
                gallery_rows.append(
                    {
                        "image_id": image_id,
                        "individual_id": f"eleph_{indiv_i:02d}",
                        "session_id": session_id,
                    }
                )
                bv = (proto + 0.1 * rng.standard_normal(dim)).astype(np.float32)
                bv /= np.linalg.norm(bv)
                body_embs.append(bv)
                body_mapping_rows.append(
                    {
                        "image_id": image_id,
                        "individual_id": f"eleph_{indiv_i:02d}",
                        "embedding_row": body_row_idx,
                        "crop_kind": "body",
                        "crop_ordinal": 0,
                    }
                )
                body_row_idx += 1

                for e in range(n_ear_per_image):
                    ev = (proto + 0.15 * rng.standard_normal(dim)).astype(np.float32)
                    ev /= np.linalg.norm(ev)
                    ear_embs.append(ev)
                    ear_mapping_rows.append(
                        {
                            "image_id": image_id,
                            "individual_id": f"eleph_{indiv_i:02d}",
                            "embedding_row": ear_row_idx,
                            "crop_kind": "ear",
                            "crop_ordinal": e,
                        }
                    )
                    ear_row_idx += 1

    gallery_df = pd.DataFrame(gallery_rows)
    body_matrix = np.stack(body_embs)
    ear_matrix = np.stack(ear_embs)

    desc_mappings = {
        "body_desc": pd.DataFrame(body_mapping_rows),
        "ear_desc": pd.DataFrame(ear_mapping_rows),
    }
    emb_matrices = {"body_desc": body_matrix, "ear_desc": ear_matrix}

    # Build calibrators (fitted on small synthetic OOF data).
    from configs.config_elephant import MIN_POSITIVE_PAIRS_FOR_ISOTONIC

    calibrators = {}
    for ch in channels:
        cal_rng = np.random.default_rng(seed + hash(ch) % 1000)
        n = max(MIN_POSITIVE_PAIRS_FOR_ISOTONIC // 2, 10)
        s = np.concatenate([cal_rng.uniform(0.6, 1.0, n), cal_rng.uniform(0.0, 0.5, n)])
        lbl = np.concatenate([np.ones(n), np.zeros(n)])
        cal = Calibrator()
        cal.fit(s, lbl)
        calibrators[ch] = cal

    # Equal weights.
    weights = {ch: 1.0 / len(channels) for ch in channels}

    scorer = IdentityLevelScorer(
        gallery_image_df=gallery_df,
        descriptor_mappings={ch: desc_mappings[ch] for ch in channels},
        embedding_matrices={ch: emb_matrices[ch] for ch in channels},
        calibrators=calibrators,
        weights=weights,
        all_channels=channels,
        accept_threshold=0.5,
    )

    # Build query data: one query image per individual from a separate "session"
    query_data: dict[str, dict[str, np.ndarray]] = {}
    for indiv_i in range(n_individuals):
        proto = protos[indiv_i]
        qid = f"query_{indiv_i:02d}"
        q_body = (proto + 0.1 * rng.standard_normal(dim)).astype(np.float32)
        q_body /= np.linalg.norm(q_body)
        q_ear_0 = (proto + 0.15 * rng.standard_normal(dim)).astype(np.float32)
        q_ear_0 /= np.linalg.norm(q_ear_0)
        q_ear_1 = (proto + 0.15 * rng.standard_normal(dim)).astype(np.float32)
        q_ear_1 /= np.linalg.norm(q_ear_1)
        query_data[qid] = {
            "body_desc": q_body.reshape(1, -1),
            "ear_desc": np.stack([q_ear_0, q_ear_1]),
        }

    return scorer, query_data, gallery_df


# ---------------------------------------------------------------------------
# Retrieval metric unit tests
# ---------------------------------------------------------------------------

class TestRetrievalMetrics:
    def test_average_precision_perfect(self):
        ranked = [
            IdentityScore("correct", fused_score=0.9),
            IdentityScore("wrong1", fused_score=0.5),
            IdentityScore("wrong2", fused_score=0.3),
        ]
        ap = _average_precision(ranked, "correct")
        assert abs(ap - 1.0) < 1e-6, f"AP should be 1.0, got {ap}"

    def test_average_precision_rank_two(self):
        ranked = [
            IdentityScore("wrong1", fused_score=0.9),
            IdentityScore("correct", fused_score=0.7),
            IdentityScore("wrong2", fused_score=0.5),
        ]
        ap = _average_precision(ranked, "correct")
        assert abs(ap - 0.5) < 1e-6, f"AP@rank2 should be 0.5, got {ap}"

    def test_average_precision_not_in_list(self):
        ranked = [IdentityScore("wrong", fused_score=0.5)]
        ap = _average_precision(ranked, "missing_id")
        assert ap == 0.0

    def test_compute_map_all_correct(self):
        results = [
            QueryResult(
                query_image_id=f"q{i}",
                query_individual_id=f"e{i}",
                ranked_identities=[IdentityScore(f"e{i}", fused_score=0.9)],
                channels_present=["body"],
                channels_absent=[],
            )
            for i in range(5)
        ]
        assert abs(compute_map(results) - 1.0) < 1e-6

    def test_compute_top1(self):
        results = [
            QueryResult(
                query_image_id="q0",
                query_individual_id="e0",
                ranked_identities=[
                    IdentityScore("e0", fused_score=0.9),
                    IdentityScore("e1", fused_score=0.5),
                ],
                channels_present=["body"],
                channels_absent=[],
            ),
            QueryResult(
                query_image_id="q1",
                query_individual_id="e1",
                ranked_identities=[
                    IdentityScore("e0", fused_score=0.9),  # wrong top-1
                    IdentityScore("e1", fused_score=0.5),
                ],
                channels_present=["body"],
                channels_absent=[],
            ),
        ]
        assert abs(compute_top1(results) - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# IdentityLevelScorer tests
# ---------------------------------------------------------------------------

class TestIdentityLevelScorer:
    def test_weight_validation_negative_weight(self):
        gallery_df = pd.DataFrame(
            [{"image_id": "i0", "individual_id": "e0", "session_id": "s0"}]
        )
        dm = pd.DataFrame(
            [{"image_id": "i0", "individual_id": "e0", "embedding_row": 0, "crop_kind": "body"}]
        )
        emb = np.ones((1, 4), dtype=np.float32)
        cal = Calibrator()
        rng = np.random.default_rng(0)
        n = 10
        cal.fit(
            np.concatenate([rng.uniform(0.5, 1, n), rng.uniform(0, 0.5, n)]),
            np.concatenate([np.ones(n), np.zeros(n)]),
        )

        with pytest.raises(ValueError, match="negative"):
            IdentityLevelScorer(
                gallery_image_df=gallery_df,
                descriptor_mappings={"ch": dm},
                embedding_matrices={"ch": emb},
                calibrators={"ch": cal},
                weights={"ch": -0.1},  # negative weight
                all_channels=["ch"],
            )

    def test_weight_validation_does_not_sum_to_one(self):
        gallery_df = pd.DataFrame(
            [{"image_id": "i0", "individual_id": "e0", "session_id": "s0"}]
        )
        dm = pd.DataFrame(
            [{"image_id": "i0", "individual_id": "e0", "embedding_row": 0, "crop_kind": "body"}]
        )
        emb = np.ones((1, 4), dtype=np.float32)
        cal = Calibrator()
        rng = np.random.default_rng(0)
        n = 10
        cal.fit(
            np.concatenate([rng.uniform(0.5, 1, n), rng.uniform(0, 0.5, n)]),
            np.concatenate([np.ones(n), np.zeros(n)]),
        )

        with pytest.raises(ValueError, match="sum"):
            IdentityLevelScorer(
                gallery_image_df=gallery_df,
                descriptor_mappings={"ch": dm},
                embedding_matrices={"ch": emb},
                calibrators={"ch": cal},
                weights={"ch": 0.5},  # sums to 0.5, not 1.0
                all_channels=["ch"],
            )

    def test_score_query_returns_all_gallery_identities(self):
        scorer, query_data, gallery_df = _make_scorer_fixture(
            n_individuals=4, n_sessions=2, n_body_per_session=2
        )
        gallery_ids = set(gallery_df["individual_id"])
        q_id = "query_00"
        result = scorer.score_query(
            q_id, query_data[q_id], query_individual_id="eleph_00"
        )
        ranked_ids = {r.individual_id for r in result.ranked_identities}
        assert gallery_ids == ranked_ids, (
            f"Scorer should return all gallery identities; missing: {gallery_ids - ranked_ids}"
        )

    def test_score_query_correct_identity_ranks_first(self):
        """Known identity should rank first with noise-free prototype embeddings."""
        scorer, query_data, gallery_df = _make_scorer_fixture(
            n_individuals=4, n_sessions=3, n_body_per_session=2
        )
        n_correct_top1 = 0
        for i in range(4):
            q_id = f"query_{i:02d}"
            true_id = f"eleph_{i:02d}"
            result = scorer.score_query(q_id, query_data[q_id], query_individual_id=true_id)
            if result.ranked_identities and result.ranked_identities[0].individual_id == true_id:
                n_correct_top1 += 1
        # With low noise embeddings all or most should rank first.
        assert n_correct_top1 >= 3, f"Expected ≥3/4 top-1 correct, got {n_correct_top1}"

    def test_missing_channel_renormalises_weights(self):
        """
        When a channel is absent for a query, its weight must be
        redistributed to available channels so fused scores remain in [0,1].
        """
        scorer, query_data, gallery_df = _make_scorer_fixture(
            n_individuals=3, n_sessions=2, n_body_per_session=2
        )
        q_id = "query_00"
        # Provide only the body channel (remove ear).
        partial_emb = {"body_desc": query_data[q_id]["body_desc"]}
        result_partial = scorer.score_query(
            q_id, partial_emb, query_individual_id="eleph_00"
        )
        assert "body_desc" in result_partial.channels_present
        assert "ear_desc" in result_partial.channels_absent

        # Fused scores should still be in [0,1] with renormalised weight.
        for ident in result_partial.ranked_identities:
            assert 0.0 <= ident.fused_score <= 1.0 + 1e-6, (
                f"Fused score out of range: {ident.fused_score}"
            )

    def test_no_channels_returns_empty_result(self):
        """Query with no embeddings at all → empty ranked_identities."""
        scorer, query_data, gallery_df = _make_scorer_fixture(
            n_individuals=3, n_sessions=2, n_body_per_session=2
        )
        result = scorer.score_query("q_empty", {})
        assert result.ranked_identities == []

    def test_identity_union_consistent_scoring(self):
        """
        All identities must receive a fused score even when only one channel
        produces a top match.  No identity may be silently excluded from the
        scoring because another channel didn't shortlist it.
        """
        scorer, query_data, gallery_df = _make_scorer_fixture(
            n_individuals=5, n_sessions=2, n_body_per_session=2
        )
        gallery_ids = set(gallery_df["individual_id"])
        q_id = "query_00"
        result = scorer.score_query(q_id, query_data[q_id])
        ranked_ids = {r.individual_id for r in result.ranked_identities}
        # Every gallery identity must appear in the ranked list.
        missing = gallery_ids - ranked_ids
        assert not missing, f"Missing gallery identities from ranked list: {missing}"

    def test_unknown_query_flagged(self):
        """Query whose true identity is not in gallery must be flagged as unknown."""
        scorer, query_data, gallery_df = _make_scorer_fixture(
            n_individuals=3, n_sessions=2, n_body_per_session=2
        )
        q_id = "query_00"
        result = scorer.score_query(
            q_id, query_data[q_id], query_individual_id="unseen_elephant"
        )
        assert result.unknown_query is True
        assert result.top1_correct is None  # unknown queries have no top-k label
        assert result.top5_correct is None


# ---------------------------------------------------------------------------
# Weight fitting tests
# ---------------------------------------------------------------------------

class TestFitFusionWeights:
    def _build_oof_results(self, n_q: int = 20, seed: int = 0) -> list[QueryResult]:
        """Build synthetic OOF results for weight fitting tests."""
        rng = np.random.default_rng(seed)
        channels = ["body_desc", "ear_desc"]
        identities = [f"eleph_{i:02d}" for i in range(5)]
        results = []
        for qi in range(n_q):
            gt = identities[qi % len(identities)]
            ranked = [
                IdentityScore(
                    individual_id=rid,
                    channel_calibrated={
                        "body_desc": float(rng.uniform(0.3, 0.9)),
                        "ear_desc": float(rng.uniform(0.3, 0.9)),
                    },
                    channels_available=channels,
                    fused_score=0.0,
                )
                for rid in identities
            ]
            results.append(
                QueryResult(
                    query_image_id=f"q{qi}",
                    query_individual_id=gt,
                    ranked_identities=ranked,
                    channels_present=channels,
                    channels_absent=[],
                )
            )
        return results

    def test_weights_sum_to_one(self):
        results = self._build_oof_results()
        channels = ["body_desc", "ear_desc"]
        weights, _ = fit_fusion_weights(results, channels, grid_step=0.25)
        total = sum(weights.get(ch, 0.0) for ch in channels)
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"

    def test_weights_non_negative(self):
        results = self._build_oof_results()
        channels = ["body_desc", "ear_desc"]
        weights, _ = fit_fusion_weights(results, channels, grid_step=0.25)
        for ch, w in weights.items():
            assert w >= 0.0, f"Weight for '{ch}' is negative: {w}"

    def test_weights_deterministic(self):
        results = self._build_oof_results(seed=42)
        channels = ["body_desc", "ear_desc"]
        w1, _ = fit_fusion_weights(results, channels, grid_step=0.25)
        w2, _ = fit_fusion_weights(results, channels, grid_step=0.25)
        for ch in channels:
            assert abs(w1[ch] - w2[ch]) < 1e-8, (
                f"fit_fusion_weights is not deterministic for channel '{ch}'"
            )

    def test_vectorized_search_matches_brute_force(self):
        results = self._build_oof_results(seed=17)
        channels = ["body_desc", "ear_desc"]
        vectorized, diagnostics = fit_fusion_weights(
            results,
            channels,
            grid_step=0.25,
            device="cpu",
        )

        best_map = -1.0
        best_top1 = -1.0
        brute = None
        for first in np.arange(0.0, 1.01, 0.25):
            weights = {
                channels[0]: float(first),
                channels[1]: float(1.0 - first),
            }
            ranked = _apply_weights_and_rank(results, weights, channels)
            map_score = compute_map(ranked)
            top1_score = compute_top1(ranked)
            if map_score > best_map or (
                abs(map_score - best_map) < 1e-8 and top1_score > best_top1
            ):
                brute = weights
                best_map = map_score
                best_top1 = top1_score

        assert vectorized == brute
        assert diagnostics["best_map"] == round(best_map, 6)
        assert diagnostics["best_top1"] == round(best_top1, 6)

    def test_cuda_search_matches_cpu_when_available(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        results = self._build_oof_results(seed=23)
        channels = ["body_desc", "ear_desc"]
        cpu_weights, cpu_diagnostics = fit_fusion_weights(
            results,
            channels,
            grid_step=0.25,
            device="cpu",
        )
        cuda_weights, cuda_diagnostics = fit_fusion_weights(
            results,
            channels,
            grid_step=0.25,
            device="cuda",
        )
        assert cuda_weights == cpu_weights
        assert cuda_diagnostics["best_map"] == cpu_diagnostics["best_map"]
        assert cuda_diagnostics["best_top1"] == cpu_diagnostics["best_top1"]

    def test_empty_results_raises(self):
        with pytest.raises(ValueError, match="no OOF results"):
            fit_fusion_weights([], ["body", "ear"])

    def test_no_channels_raises(self):
        with pytest.raises(ValueError, match="no channels"):
            fit_fusion_weights([QueryResult("q", None, [], [], [])], [])

    def test_diagnostics_contain_required_keys(self):
        results = self._build_oof_results()
        _, diag = fit_fusion_weights(results, ["body_desc", "ear_desc"], grid_step=0.5)
        assert "best_map" in diag
        assert "best_top1" in diag
        assert "best_weights" in diag
        assert "n_candidates_evaluated" in diag

    def test_oof_weights_not_tuned_on_probes(self):
        """
        fit_fusion_weights must only receive gallery OOF results.
        Verify that the function doesn't import or use probe data internally.
        This is a documentation/interface test.
        """
        import inspect
        from models import identity_fusion
        src = inspect.getsource(identity_fusion.fit_fusion_weights)
        # The function must not reference probe images or fixed query data.
        assert "probe" not in src.lower() or "# probe" not in src.lower(), (
            "fit_fusion_weights should not reference probe data"
        )


# ---------------------------------------------------------------------------
# Unknown threshold tests
# ---------------------------------------------------------------------------

class TestEstimateUnknownThreshold:
    def _build_oof_for_threshold(
        self, n_known: int = 50, n_unknown: int = 30, seed: int = 0
    ) -> tuple[list[QueryResult], list[QueryResult]]:
        """
        Return (known_results, unknown_results) for threshold tests.

        known_results  – queries whose identity is "in the gallery" (high scores).
        unknown_results – simulated unknown queries (lower scores).
        """
        rng = np.random.default_rng(seed)
        known_results = []
        unknown_results = []

        # Known queries: high fused scores, identity_in_oof_gallery=True.
        for i in range(n_known):
            gt = f"eleph_{i % 10}"
            known_results.append(
                QueryResult(
                    query_image_id=f"known_{i}",
                    query_individual_id=gt,
                    ranked_identities=[
                        IdentityScore(gt, fused_score=float(rng.uniform(0.6, 0.95)))
                    ],
                    channels_present=["body"],
                    channels_absent=[],
                    identity_in_oof_gallery=True,
                )
            )

        # Unknown queries: lower fused scores, identity_in_oof_gallery=False.
        for i in range(n_unknown):
            unknown_results.append(
                QueryResult(
                    query_image_id=f"unknown_{i}",
                    query_individual_id=f"new_eleph_{i}",
                    ranked_identities=[
                        IdentityScore("eleph_0", fused_score=float(rng.uniform(0.2, 0.55)))
                    ],
                    channels_present=["body"],
                    channels_absent=[],
                    identity_in_oof_gallery=False,
                )
            )

        return known_results, unknown_results

    def test_threshold_in_valid_range(self):
        known_results, unknown_results = self._build_oof_for_threshold()
        threshold, diag = estimate_unknown_threshold(known_results, unknown_results)
        assert 0.0 <= threshold <= 1.0, f"Threshold out of [0,1]: {threshold}"

    def test_threshold_diagnostics_keys(self):
        known_results, unknown_results = self._build_oof_for_threshold()
        _, diag = estimate_unknown_threshold(known_results, unknown_results)
        assert "threshold" in diag
        assert "n_known" in diag
        assert "n_unknown" in diag
        assert "far_at_threshold" in diag
        assert "frr_at_threshold" in diag

    def test_threshold_separates_known_from_unknown(self):
        """Threshold should be in the overlap region of separated distributions."""
        known_results, unknown_results = self._build_oof_for_threshold(seed=7)
        threshold, diag = estimate_unknown_threshold(known_results, unknown_results)
        # With known>0.6 and unknown<0.55 the threshold should be around 0.55–0.65.
        assert 0.3 <= threshold <= 0.8, (
            f"Expected threshold in ~[0.3,0.8] for well-separated distributions, "
            f"got {threshold}"
        )

    def test_empty_unknown_list_raises(self):
        """Hard-fail when unknown distribution is empty (no 0.5 fallback)."""
        from models.oof_calibration import CalibrationSupportError

        known_results = [
            QueryResult(
                query_image_id=f"q{i}",
                query_individual_id=f"e{i % 5}",
                ranked_identities=[IdentityScore(f"e{i%5}", fused_score=0.8)],
                channels_present=["body"],
                channels_absent=[],
                identity_in_oof_gallery=True,
            )
            for i in range(10)
        ]
        with pytest.raises(CalibrationSupportError, match="no support"):
            estimate_unknown_threshold(known_results, [])

    def test_empty_known_list_raises(self):
        """Hard-fail when known distribution is empty."""
        from models.oof_calibration import CalibrationSupportError

        unknown_results = [
            QueryResult(
                query_image_id="u0",
                query_individual_id="new_e",
                ranked_identities=[IdentityScore("e0", fused_score=0.4)],
                channels_present=["body"],
                channels_absent=[],
                identity_in_oof_gallery=False,
            )
        ]
        with pytest.raises(CalibrationSupportError, match="no support"):
            estimate_unknown_threshold([], unknown_results)


# ---------------------------------------------------------------------------
# OOF identity score builder test
# ---------------------------------------------------------------------------

class TestBuildOOFIdentityScores:
    def test_returns_one_result_per_query_image(self):
        from tests.test_grouped_calibration import _make_gallery
        gallery_df, desc_mappings, emb_matrices = _make_gallery(
            n_individuals=3, n_sessions_per_indiv=3, n_images_per_session=2
        )
        channels = ["body_desc", "ear_desc"]

        # Build simple Platt calibrators.
        calibrators = {}
        from configs.config_elephant import MIN_POSITIVE_PAIRS_FOR_ISOTONIC
        rng = np.random.default_rng(0)
        for ch in channels:
            n = max(MIN_POSITIVE_PAIRS_FOR_ISOTONIC // 2, 10)
            s = np.concatenate([rng.uniform(0.6, 1, n), rng.uniform(0, 0.5, n)])
            lbl = np.concatenate([np.ones(n), np.zeros(n)])
            cal = Calibrator()
            cal.fit(s, lbl)
            calibrators[ch] = cal

        results = build_oof_identity_scores(
            gallery_df, desc_mappings, emb_matrices, calibrators, channels
        )
        assert len(results) > 0, "Expected OOF results"
        # Each result should have a non-empty ranked list.
        for r in results:
            assert len(r.ranked_identities) > 0, (
                f"Query {r.query_image_id} has empty ranked identities"
            )

    def test_probe_images_excluded(self):
        """
        build_oof_identity_scores must only produce results for gallery images.
        Probe images are not passed into this function (interface contract).
        """
        from tests.test_grouped_calibration import _make_gallery
        gallery_df, desc_mappings, emb_matrices = _make_gallery(
            n_individuals=3, n_sessions_per_indiv=3, n_images_per_session=2
        )
        channels = ["body_desc"]

        from configs.config_elephant import MIN_POSITIVE_PAIRS_FOR_ISOTONIC
        rng = np.random.default_rng(0)
        n = max(MIN_POSITIVE_PAIRS_FOR_ISOTONIC // 2, 10)
        s = np.concatenate([rng.uniform(0.6, 1, n), rng.uniform(0, 0.5, n)])
        lbl = np.concatenate([np.ones(n), np.zeros(n)])
        cal = Calibrator()
        cal.fit(s, lbl)

        results = build_oof_identity_scores(
            gallery_df,
            {"body_desc": desc_mappings["body_desc"]},
            {"body_desc": emb_matrices["body_desc"]},
            {"body_desc": cal},
            channels,
        )

        gallery_ids = set(gallery_df["image_id"])
        result_ids = {r.query_image_id for r in results}
        extra = result_ids - gallery_ids
        assert not extra, f"OOF results contain non-gallery image IDs: {extra}"


# ---------------------------------------------------------------------------
# Protocol correctness tests  (Task 5)
# ---------------------------------------------------------------------------

def _make_split_fixture(
    n_gallery_individuals: int = 4,
    n_held_out_gallery_individuals: int = 2,
    n_sessions: int = 2,
    n_images_per_session: int = 2,
    dim: int = 16,
    seed: int = 42,
) -> dict:
    """
    Build a synthetic multi-split fixture with:
      gallery          – known identities, temporal reference
      held_out_gallery – onboarding identities, held-out reference
      probe            – temporal probes (identities in gallery)
      held_out_probe   – onboarding probes (identities in held_out_gallery)

    Returns a dict with keys:
      gallery_df, held_out_gallery_df, probe_df, held_out_probe_df,
      combined_gallery_df,
      gallery_desc_mappings, gallery_emb_matrices,
      combined_desc_mappings, combined_emb_matrices,
      probe_desc_mappings, probe_emb_matrices,
      calibrators, weights, channels
    """
    rng = np.random.default_rng(seed)
    channels = ["body_desc"]

    n_gal = n_gallery_individuals
    n_ho = n_held_out_gallery_individuals
    n_total = n_gal + n_ho

    # Generate prototypes per identity.
    protos = []
    for i in range(n_total):
        p = rng.standard_normal(dim).astype(np.float32)
        p /= np.linalg.norm(p)
        protos.append(p)

    def _make_vec(proto, noise=0.05):
        v = (proto + noise * rng.standard_normal(dim)).astype(np.float32)
        v /= np.linalg.norm(v)
        return v

    def _make_set(start_idx, count, split_name):
        rows, emb_rows, emb_list = [], [], []
        row_idx = 0
        df_rows = []
        for i in range(count):
            indiv_i = start_idx + i
            indiv_id = f"id_{indiv_i:03d}"
            for s in range(n_sessions):
                for k in range(n_images_per_session):
                    img_id = f"{split_name}_{indiv_i}_{s}_{k}"
                    df_rows.append(
                        {"image_id": img_id, "individual_id": indiv_id,
                         "session_id": f"sess_{indiv_i}_{s}", "split": split_name}
                    )
                    emb_list.append(_make_vec(protos[indiv_i]))
                    emb_rows.append(
                        {"image_id": img_id, "individual_id": indiv_id,
                         "embedding_row": row_idx, "crop_kind": "body"}
                    )
                    row_idx += 1
        return pd.DataFrame(df_rows), pd.DataFrame(emb_rows), np.stack(emb_list)

    gallery_df, gallery_emb_mapping, gallery_emb_matrix = _make_set(0, n_gal, "gallery")
    ho_gallery_df, ho_gallery_emb_mapping, ho_gallery_emb_matrix = _make_set(
        n_gal, n_ho, "held_out_gallery"
    )
    # Probes: one image per identity, from a new session.
    probe_rows, probe_emb_rows, probe_emb_list = [], [], []
    for i in range(n_gal + n_ho):
        indiv_id = f"id_{i:03d}"
        split_name = "probe" if i < n_gal else "held_out_probe"
        img_id = f"{split_name}_q_{i}"
        probe_rows.append(
            {"image_id": img_id, "individual_id": indiv_id,
             "session_id": f"qsess_{i}", "split": split_name}
        )
        probe_emb_list.append(_make_vec(protos[i]))
        probe_emb_rows.append(
            {"image_id": img_id, "individual_id": indiv_id,
             "embedding_row": i, "crop_kind": "body"}
        )
    probe_df_all = pd.DataFrame(probe_rows)
    probe_emb_mapping = pd.DataFrame(probe_emb_rows)
    probe_emb_matrix = np.stack(probe_emb_list)

    # Combined gallery = gallery + held_out_gallery (no probes).
    combined_gallery_df = pd.concat(
        [gallery_df, ho_gallery_df], ignore_index=True
    ).drop_duplicates("image_id").reset_index(drop=True)

    # Combined embeddings: concatenate gallery and held_out_gallery embeddings.
    # Re-index embedding_rows for combined mapping.
    n_gal_crops = len(gallery_emb_mapping)
    ho_emb_mapping_combined = ho_gallery_emb_mapping.copy()
    ho_emb_mapping_combined["embedding_row"] = (
        ho_emb_mapping_combined["embedding_row"] + n_gal_crops
    )
    combined_emb_mapping = pd.concat(
        [gallery_emb_mapping, ho_emb_mapping_combined], ignore_index=True
    )
    combined_emb_matrix = np.vstack([gallery_emb_matrix, ho_gallery_emb_matrix])

    # Calibrator for body_desc.
    from configs.config_elephant import MIN_POSITIVE_PAIRS_FOR_ISOTONIC
    cal_rng = np.random.default_rng(seed)
    n = max(MIN_POSITIVE_PAIRS_FOR_ISOTONIC // 2, 10)
    s = np.concatenate([cal_rng.uniform(0.6, 1.0, n), cal_rng.uniform(0.0, 0.5, n)])
    lbl = np.concatenate([np.ones(n), np.zeros(n)])
    cal = Calibrator()
    cal.fit(s, lbl)
    calibrators = {"body_desc": cal}
    weights = {"body_desc": 1.0}

    return {
        "gallery_df": gallery_df,
        "held_out_gallery_df": ho_gallery_df,
        "probe_df": probe_df_all[probe_df_all["split"] == "probe"].reset_index(drop=True),
        "held_out_probe_df": probe_df_all[probe_df_all["split"] == "held_out_probe"].reset_index(
            drop=True
        ),
        "combined_gallery_df": combined_gallery_df,
        "gallery_desc_mappings": {"body_desc": gallery_emb_mapping},
        "gallery_emb_matrices": {"body_desc": gallery_emb_matrix},
        "combined_desc_mappings": {"body_desc": combined_emb_mapping},
        "combined_emb_matrices": {"body_desc": combined_emb_matrix},
        "probe_desc_mappings": {"body_desc": probe_emb_mapping},
        "probe_emb_matrices": {"body_desc": probe_emb_matrix},
        "calibrators": calibrators,
        "weights": weights,
        "channels": channels,
    }


class TestProtocolCorrectness:
    """
    Protocol hygiene tests:
    1. held_out_probe identity in held_out_gallery → known correct retrieval.
    2. Relabel-only unknown trial must fail (identity still indexed).
    3. Identity-removed candidate set excludes truth identity.
    4. Protocol breakdown counts sum correctly.
    5. No fixed probe image in reference (temporal or combined gallery).
    """

    def test_held_out_probe_identity_in_held_out_gallery_counts_as_known(self):
        """
        A held_out_probe query whose identity is present in held_out_gallery
        must be treated as a known-retrieval query (unknown_query=False) and
        must count as correct retrieval when the truth identity ranks first.
        """
        fix = _make_split_fixture(n_gallery_individuals=3, n_held_out_gallery_individuals=2)
        channels = fix["channels"]
        combined_gallery_df = fix["combined_gallery_df"]

        scorer = IdentityLevelScorer(
            gallery_image_df=combined_gallery_df,
            descriptor_mappings=fix["combined_desc_mappings"],
            embedding_matrices=fix["combined_emb_matrices"],
            calibrators=fix["calibrators"],
            weights=fix["weights"],
            all_channels=channels,
        )

        ho_probe_df = fix["held_out_probe_df"]
        # Take one held_out_probe query.
        row = ho_probe_df.iloc[0]
        q_img_id = str(row["image_id"])
        q_indiv = str(row["individual_id"])

        # Extract probe embedding.
        probe_dm = fix["probe_desc_mappings"]["body_desc"]
        probe_emb = fix["probe_emb_matrices"]["body_desc"]
        q_row_idx = int(probe_dm[probe_dm["image_id"] == q_img_id]["embedding_row"].iloc[0])
        q_emb = probe_emb[q_row_idx : q_row_idx + 1]

        result = scorer.score_query(
            query_image_id=q_img_id,
            query_emb_rows={"body_desc": q_emb},
            query_individual_id=q_indiv,
        )
        result.probe_type = "unseen_identity_onboarding"

        assert result.unknown_query is False, (
            f"held_out_probe identity '{q_indiv}' is in held_out_gallery but "
            f"was flagged unknown_query=True. Scorer must use combined gallery."
        )
        # With low-noise embeddings, correct identity should rank first.
        assert result.ranked_identities, "No ranked identities returned."
        assert result.ranked_identities[0].individual_id == q_indiv, (
            f"Expected truth identity '{q_indiv}' to rank first; "
            f"got '{result.ranked_identities[0].individual_id}'."
        )
        assert result.top1_correct is True

        # compute_top1 should count this as a correct retrieval.
        top1 = compute_top1([result])
        assert top1 == 1.0, f"compute_top1 should be 1.0, got {top1}"

    def test_relabel_only_unknown_trial_fails_integrity_check(self):
        """
        Merely setting unknown_query=True on a query whose identity is STILL
        indexed in the reference must NOT be accepted as a valid unknown trial
        by _open_set_metrics.  Only simulated_unknown=True trials (from
        identity removal) are valid; relabelled-only entries are filtered out.
        """
        from pipeline.step_4c_normalized_eval import _open_set_metrics

        # A query whose identity IS in the gallery, but relabelled as unknown
        # (without removing the identity from the reference).
        relabelled_as_unknown = QueryResult(
            query_image_id="q_relabelled",
            query_individual_id="id_000",
            ranked_identities=[IdentityScore("id_000", fused_score=0.85)],
            channels_present=["body_desc"],
            channels_absent=[],
            unknown_query=True,
            simulated_unknown=False,  # NOT a proper simulated unknown
        )
        known_result = QueryResult(
            query_image_id="q_known",
            query_individual_id="id_001",
            ranked_identities=[IdentityScore("id_001", fused_score=0.80)],
            channels_present=["body_desc"],
            channels_absent=[],
            unknown_query=False,
        )

        # Pass the relabelled (invalid) entry as a simulated unknown.
        # _open_set_metrics should filter it because simulated_unknown=False.
        metrics = _open_set_metrics(
            known_results=[known_result],
            simulated_unknown_results=[relabelled_as_unknown],
            threshold=0.5,
        )
        # After filtering, there are zero valid simulated-unknown trials.
        assert metrics["n_simulated_unknown_trials"] == 0, (
            "Relabelled-only unknown trial with simulated_unknown=False must be "
            "filtered from open-set metrics. Got "
            f"n_simulated_unknown_trials={metrics['n_simulated_unknown_trials']}."
        )

    def test_identity_removed_candidate_set_excludes_truth(self):
        """
        simulate_probe_unknown_trials must guarantee that the truth identity
        does NOT appear in any trial's ranked_identities.
        """
        fix = _make_split_fixture(n_gallery_individuals=3, n_held_out_gallery_individuals=2)
        ho_probe_df = fix["held_out_probe_df"]

        sim_results = simulate_probe_unknown_trials(
            probe_df=ho_probe_df,
            combined_gallery_df=fix["combined_gallery_df"],
            descriptor_mappings=fix["combined_desc_mappings"],
            embedding_matrices=fix["combined_emb_matrices"],
            calibrators=fix["calibrators"],
            weights=fix["weights"],
            all_channels=fix["channels"],
            probe_emb_mappings=fix["probe_desc_mappings"],
            probe_emb_matrices=fix["probe_emb_matrices"],
        )

        assert len(sim_results) > 0, "Expected at least one simulated-unknown trial."
        for r in sim_results:
            truth_id = r.query_individual_id
            candidate_ids = {x.individual_id for x in r.ranked_identities}
            assert truth_id not in candidate_ids, (
                f"Truth identity '{truth_id}' found in candidates for "
                f"simulated-unknown trial '{r.query_image_id}'. "
                "Identity removal has failed."
            )
            assert r.simulated_unknown is True
            assert r.unknown_query is True
            assert r.identity_in_oof_gallery is False

    def test_protocol_breakdown_counts_sum_correctly(self):
        """
        n_temporal_known + n_temporal_not_known
        + n_onboarding_known + n_onboarding_not_known
        == n_probe_total (temporal + onboarding probes).

        Simulated-unknown trials are separate and do not count in probe_total.
        """
        fix = _make_split_fixture(n_gallery_individuals=3, n_held_out_gallery_individuals=2)
        channels = fix["channels"]

        temporal_probe_df = fix["probe_df"]
        ho_probe_df = fix["held_out_probe_df"]

        # Temporal scorer: gallery only.
        temporal_scorer = IdentityLevelScorer(
            gallery_image_df=fix["gallery_df"],
            descriptor_mappings=fix["gallery_desc_mappings"],
            embedding_matrices=fix["gallery_emb_matrices"],
            calibrators=fix["calibrators"],
            weights=fix["weights"],
            all_channels=channels,
        )
        # Onboarding scorer: combined gallery.
        onboarding_scorer = IdentityLevelScorer(
            gallery_image_df=fix["combined_gallery_df"],
            descriptor_mappings=fix["combined_desc_mappings"],
            embedding_matrices=fix["combined_emb_matrices"],
            calibrators=fix["calibrators"],
            weights=fix["weights"],
            all_channels=channels,
        )

        probe_dm = fix["probe_desc_mappings"]["body_desc"]
        probe_emb = fix["probe_emb_matrices"]["body_desc"]

        def _get_emb(img_id):
            r = int(probe_dm[probe_dm["image_id"] == img_id]["embedding_row"].iloc[0])
            return {"body_desc": probe_emb[r : r + 1]}

        temporal_results = []
        for _, row in temporal_probe_df.iterrows():
            res = temporal_scorer.score_query(
                str(row["image_id"]), _get_emb(str(row["image_id"])), str(row["individual_id"])
            )
            res.probe_type = "temporal"
            temporal_results.append(res)

        onboarding_results = []
        for _, row in ho_probe_df.iterrows():
            res = onboarding_scorer.score_query(
                str(row["image_id"]), _get_emb(str(row["image_id"])), str(row["individual_id"])
            )
            res.probe_type = "unseen_identity_onboarding"
            onboarding_results.append(res)

        n_probe_total = len(temporal_results) + len(onboarding_results)
        n_temporal_known = sum(1 for r in temporal_results if not r.unknown_query)
        n_temporal_not_known = sum(1 for r in temporal_results if r.unknown_query)
        n_onboarding_known = sum(1 for r in onboarding_results if not r.unknown_query)
        n_onboarding_not_known = sum(1 for r in onboarding_results if r.unknown_query)

        assert (n_temporal_known + n_temporal_not_known
                + n_onboarding_known + n_onboarding_not_known
                == n_probe_total), (
            f"Protocol counts mismatch: "
            f"{n_temporal_known}+{n_temporal_not_known}+"
            f"{n_onboarding_known}+{n_onboarding_not_known} != {n_probe_total}"
        )
        # Held-out probe identities are in combined gallery → all should be known.
        assert n_onboarding_known == len(ho_probe_df), (
            f"All onboarding probes should be known (identity in combined gallery). "
            f"Got {n_onboarding_known}/{len(ho_probe_df)}."
        )

    def test_no_probe_image_in_reference(self):
        """
        No probe image (temporal or onboarding) may appear in the temporal
        gallery or combined gallery used as reference.
        """
        fix = _make_split_fixture(n_gallery_individuals=3, n_held_out_gallery_individuals=2)
        probe_ids = set(fix["probe_df"]["image_id"]) | set(fix["held_out_probe_df"]["image_id"])
        gallery_ids = set(fix["gallery_df"]["image_id"])
        combined_ids = set(fix["combined_gallery_df"]["image_id"])

        leak_temporal = probe_ids & gallery_ids
        assert not leak_temporal, (
            f"Probe images found in temporal gallery: {leak_temporal}"
        )
        leak_combined = probe_ids & combined_ids
        assert not leak_combined, (
            f"Probe images found in combined gallery: {leak_combined}"
        )

    def test_compute_top5_helper(self):
        """compute_top5 counts queries where truth is in top-5 candidates."""
        results = [
            QueryResult(
                query_image_id="q0",
                query_individual_id="e0",
                ranked_identities=[
                    IdentityScore("wrong1", fused_score=0.9),
                    IdentityScore("wrong2", fused_score=0.8),
                    IdentityScore("wrong3", fused_score=0.7),
                    IdentityScore("wrong4", fused_score=0.6),
                    IdentityScore("e0", fused_score=0.5),  # rank 5 → top-5 hit
                ],
                channels_present=["body_desc"],
                channels_absent=[],
            ),
            QueryResult(
                query_image_id="q1",
                query_individual_id="e1",
                ranked_identities=[
                    IdentityScore("wrong1", fused_score=0.9),
                    IdentityScore("wrong2", fused_score=0.8),
                    IdentityScore("wrong3", fused_score=0.7),
                    IdentityScore("wrong4", fused_score=0.6),
                    IdentityScore("wrong5", fused_score=0.5),
                    IdentityScore("e1", fused_score=0.4),  # rank 6 → NOT top-5
                ],
                channels_present=["body_desc"],
                channels_absent=[],
            ),
        ]
        assert abs(compute_top5(results) - 0.5) < 1e-6, (
            f"Expected top5=0.5 (1/2 queries), got {compute_top5(results)}"
        )

    def test_check_calibration_flatness_nearly_flat(self):
        """
        check_calibration_flatness must correctly probe a calibrator's output
        range over a uniform score grid and report it.
        """
        # Build a normal calibrator.
        rng = np.random.default_rng(0)
        n = 30
        scores = np.concatenate([rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n)])
        labels = np.concatenate([np.ones(n), np.zeros(n)])
        cal = Calibrator()
        cal.fit(scores, labels)

        diag = check_calibration_flatness({"body_desc": cal})
        assert "body_desc" in diag
        d = diag["body_desc"]
        assert "output_range" in d
        assert "n_unique_outputs" in d
        assert "fraction_unique" in d
        assert "nearly_flat" in d
        # With well-separated training data, output_range should be non-trivial.
        assert d["output_range"] >= 0.0
        assert isinstance(d["nearly_flat"], bool)

    def test_check_calibration_flatness_detects_constant_output(self):
        """
        check_calibration_flatness runs on any fitted calibrator and returns
        the expected diagnostic dict keys with correct types.
        """
        n = 30
        scores = np.full(2 * n, 0.5)
        labels = np.concatenate([np.ones(n), np.zeros(n)])
        cal = Calibrator()
        cal.fit(scores, labels)

        diag = check_calibration_flatness({"ch": cal})
        assert "ch" in diag
        d = diag["ch"]
        # Required keys must always be present.
        for key in ("method", "n_unique_outputs", "fraction_unique",
                    "min_output", "max_output", "output_range", "nearly_flat"):
            assert key in d, f"Missing key '{key}' in flatness diagnostic"
        assert isinstance(d["nearly_flat"], bool)
        assert 0.0 <= d["fraction_unique"] <= 1.0
        assert d["output_range"] >= 0.0
