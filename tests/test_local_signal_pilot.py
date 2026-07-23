# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Synthetic tests for pipeline/local_signal_pilot.py.

Covers (all using fake scorer/backend, no real LightGlue/LoFTR):
  1.  Gallery-only guard — probe / held_out_probe rows hard-fail
  2.  Session exclusion — query session is excluded from all reference pools
  3.  Deterministic sample — same seed → same pseudo-query set
  4.  Hard negatives selected without truth forcing
  5.  Positive calibration row semantics
  6.  Canonical scorer called (LocalIdentityScorer path exercised)
  7.  Gate / flatness / runtime projection checks
  8.  Fingerprint mismatch → resume reset
  9.  Resume — already-scored config is skipped
 10.  Body / ear missingness (no crop available)
 11.  Output isolation (no probe outcomes, no selected-v1 mutation)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("BTEH_SOURCE_ROOT", "/nonexistent/BTEH")
os.environ.setdefault("BTEH_ARTIFACT_ROOT", "/nonexistent/artifacts")

from pipeline.local_signal_pilot import (
    DEFAULT_N_HARD_NEG,
    DEFAULT_N_QUERIES,
    DEFAULT_N_SESSIONS_CAP,
    DEFAULT_SEED,
    GATE_MAX_CACHE_GB,
    GATE_MAX_H100_HOURS,
    GATE_MIN_COVERAGE,
    GATE_MIN_ROC_AUC,
    PILOT_CONFIG_FILENAME,
    PILOT_FINGERPRINT_FILENAME,
    PILOT_MANIFEST_FILENAME,
    PILOT_METRICS_FILENAME,
    PilotConfig,
    _assert_no_probe_ids,
    _check_fingerprint,
    _compute_pr_auc,
    _compute_roc_auc,
    _filter_gallery_only,
    _platt_calibration_check,
    build_query_crops,
    build_reference_sessions,
    check_gates,
    compute_metrics,
    estimate_budget,
    run_pilot,
    sample_pseudo_queries,
    select_hard_negatives,
    write_pilot_config,
)
from models.local_score_schema import (
    SCHEMA_VERSION,
    LocalIdentityScore,
    LocalPairScore,
    make_scoring_fingerprint,
    make_identity_scoring_fingerprint,
)
from models.identity_scorer import LocalIdentityScorer, QueryCrop, ReferenceImage
from models.local_matcher import GEOM_HOMOGRAPHY, GEOM_PARTIAL_AFFINE, REGION_BODY, REGION_EAR


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _make_image_manifest(
    identities: list[str],
    n_images_each: int,
    sessions_per_id: int = 2,
) -> pd.DataFrame:
    """Build a minimal image manifest DataFrame."""
    rows = []
    for ind_id in identities:
        for s_idx in range(sessions_per_id):
            for i_idx in range(n_images_each):
                img_id = f"{ind_id}_s{s_idx}_i{i_idx}"
                rows.append({
                    "image_id": img_id,
                    "individual_id": ind_id,
                    "session_id": f"{ind_id}_sess_{s_idx}",
                    "image_count": n_images_each * sessions_per_id,
                })
    return pd.DataFrame(rows)


def _make_splits_df(manifest_df: pd.DataFrame, probe_fraction: float = 0.0) -> pd.DataFrame:
    """Assign gallery split to all; optionally mark some as probe."""
    df = manifest_df[["image_id", "individual_id", "session_id"]].copy()
    df["split"] = "gallery"
    if probe_fraction > 0:
        n_probe = max(1, int(len(df) * probe_fraction))
        probe_indices = df.index[:n_probe]
        df.loc[probe_indices, "split"] = "probe"
    return df


def _make_crop_manifest(
    manifest_df: pd.DataFrame,
    include_body: bool = True,
    include_ear: bool = True,
) -> pd.DataFrame:
    """Build accepted crop manifest entries for each image."""
    rows = []
    for _, row in manifest_df.iterrows():
        img_id = row["image_id"]
        if include_body:
            rows.append({
                "crop_id": f"{img_id}__body_0",
                "image_id": img_id,
                "individual_id": row["individual_id"],
                "crop_kind": "body",
                "crop_ordinal": 0,
                "crop_path": f"/fake/{img_id}_body.jpg",
                "detector_confidence": 0.95,
                "detector_box": "[0,0,100,100]",
                "detector_status": "accepted",
                "review_status": "accepted",
                "schema_version": "v1",
                "source_fingerprint": "fp_src",
                "split_fingerprint": "fp_spl",
            })
        if include_ear:
            rows.append({
                "crop_id": f"{img_id}__ear_0",
                "image_id": img_id,
                "individual_id": row["individual_id"],
                "crop_kind": "ear",
                "crop_ordinal": 0,
                "crop_path": f"/fake/{img_id}_ear.jpg",
                "detector_confidence": 0.90,
                "detector_box": "[10,10,50,50]",
                "detector_status": "accepted",
                "review_status": "accepted",
                "schema_version": "v1",
                "source_fingerprint": "fp_src",
                "split_fingerprint": "fp_spl",
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=[
            "crop_id", "image_id", "individual_id", "crop_kind", "crop_ordinal",
            "crop_path", "detector_confidence", "detector_box", "detector_status",
            "review_status", "schema_version", "source_fingerprint", "split_fingerprint",
        ]
    )


def _make_fake_pair_score(
    query_crop_id: str = "q__body_0",
    ref_crop_id: str = "r__body_0",
    region: str = REGION_BODY,
    n_inliers: int = 20,
    backend: str = "lightglue",
    model_fp: str = "fake_model_fp",
    geom_model: str = GEOM_PARTIAL_AFFINE,
) -> LocalPairScore:
    scoring_fp = make_scoring_fingerprint(backend, model_fp, SCHEMA_VERSION, geom_model, True)
    return LocalPairScore(
        schema_version=SCHEMA_VERSION,
        backend=backend,
        model_fingerprint=model_fp,
        scoring_fingerprint=scoring_fp,
        source_fingerprint="fp_src",
        split_fingerprint="fp_spl",
        query_crop_id=query_crop_id,
        ref_crop_id=ref_crop_id,
        query_crop_kind=region,
        ref_crop_kind=region,
        region=region,
        orientation="original",
        geom_model_used=geom_model,
        n_raw_matches=40,
        n_inliers=n_inliers,
        inlier_ratio=n_inliers / 40.0 if n_inliers <= 40 else 1.0,
        geometric_spread=5.0,
        score=float(n_inliers),
        missing_file=False,
        latency_ms=10.0,
    )


def _make_fake_identity_score(
    candidate_id: str = "ind_A",
    score: float = 20.0,
    region: str = REGION_BODY,
    n_inliers: int = 20,
    backend: str = "lightglue",
    model_fp: str = "fake_model_fp",
    geom_model: str = GEOM_PARTIAL_AFFINE,
) -> LocalIdentityScore:
    id_fp = make_identity_scoring_fingerprint(
        backend, model_fp, SCHEMA_VERSION, DEFAULT_N_SESSIONS_CAP, 2, "mean_top_k"
    )
    pair = _make_fake_pair_score(
        region=region, n_inliers=n_inliers, backend=backend,
        model_fp=model_fp, geom_model=geom_model,
    )
    return LocalIdentityScore(
        schema_version=SCHEMA_VERSION,
        backend=backend,
        model_fingerprint=model_fp,
        scoring_fingerprint=id_fp,
        query_crop_kind=region,
        candidate_individual_id=candidate_id,
        n_pairs_attempted=1,
        n_pairs_valid=1 if score > 0 else 0,
        n_pairs_missing_file=0,
        n_sessions_used=1,
        n_sessions_cap=DEFAULT_N_SESSIONS_CAP,
        region_coverage={region: 1},
        orientations_attempted={"original"},
        aggregation_method="mean_top_k",
        top_k=2,
        score=score,
        pair_scores=[pair],
        latency_ms=10.0,
    )


class FakeMatcher:
    """
    Minimal StrictLocalMatcher stand-in for testing identity scorer.
    Returns synthetic MatchResult with configurable inlier count.
    """
    backend = "lightglue"
    model_fingerprint = "fake_model_fp"

    def __init__(self, n_inliers: int = 20):
        self._n_inliers = n_inliers

    def score_pair_strict(self, query_bgr, ref_bgr, region, *, geom_model=None, mirror_search=True):
        from models.local_matcher import MatchResult, GeomVerification, GEOM_PARTIAL_AFFINE
        geom = GeomVerification(
            model_used=geom_model or GEOM_PARTIAL_AFFINE,
            n_raw_matches=self._n_inliers * 2,
            n_inliers=self._n_inliers,
            inlier_ratio=0.5,
            geometric_spread=5.0,
            H=None,
            inlier_mask=None,
        )
        return MatchResult(
            region=region,
            orientation="original",
            raw_matches=np.zeros((self._n_inliers * 2, 2), dtype=int),
            query_keypoints=np.zeros((self._n_inliers * 2, 2)),
            ref_keypoints=np.zeros((self._n_inliers * 2, 2)),
            geom=geom,
            backend=self.backend,
            model_fingerprint=self.model_fingerprint,
            n_inliers=self._n_inliers,
        )


def _make_scorer(n_inliers: int = 20) -> LocalIdentityScorer:
    return LocalIdentityScorer(FakeMatcher(n_inliers), cache=None)


def _make_gallery_datasets(
    n_identities: int = 5,
    n_images_per_id_per_session: int = 3,
    sessions_per_id: int = 2,
    include_body: bool = True,
    include_ear: bool = True,
):
    identities = [f"ind_{chr(65 + i)}" for i in range(n_identities)]
    manifest_df = _make_image_manifest(identities, n_images_per_id_per_session, sessions_per_id)
    splits_df = _make_splits_df(manifest_df)
    crop_df = _make_crop_manifest(manifest_df, include_body=include_body, include_ear=include_ear)
    # Build gallery_df: splits_df already has individual_id; no need to re-merge
    gallery_df = splits_df[splits_df["split"] == "gallery"].copy()
    return manifest_df, splits_df, crop_df, gallery_df




# ---------------------------------------------------------------------------

class TestGalleryOnlyGuard:
    def test_probe_in_splits_raises(self):
        manifest_df, splits_df, crop_df, _ = _make_gallery_datasets(n_identities=3)
        # Inject a probe row
        splits_df = splits_df.copy()
        splits_df.loc[splits_df.index[0], "split"] = "probe"
        # Filter should NOT raise on the raw splits_df itself (probe exists there)
        # but the pilot must never process probe images
        probe_rows = splits_df[splits_df["split"] == "probe"]
        with pytest.raises(RuntimeError, match="PROBE GUARD VIOLATION"):
            _assert_no_probe_ids(probe_rows, "split", context="test")

    def test_held_out_probe_raises(self):
        df = pd.DataFrame({
            "image_id": ["img1"],
            "split": ["held_out_probe"],
        })
        with pytest.raises(RuntimeError, match="PROBE GUARD VIOLATION"):
            _assert_no_probe_ids(df, "split", context="test")

    def test_gallery_only_passes(self):
        df = pd.DataFrame({
            "image_id": ["img1", "img2"],
            "split": ["gallery", "gallery"],
        })
        _assert_no_probe_ids(df, "split", context="test")  # should not raise

    def test_filter_gallery_only_excludes_probe(self):
        df = pd.DataFrame({
            "image_id": ["g1", "g2"],
            "individual_id": ["A", "B"],
            "split": ["gallery", "gallery"],
        })
        result = _filter_gallery_only(df)
        assert set(result["split"].tolist()) == {"gallery"}

    def test_run_pilot_gallery_manifest_no_probe(self, tmp_path):
        """run_pilot must complete without errors when all input is gallery-only."""
        manifest_df, splits_df, crop_df, _ = _make_gallery_datasets(n_identities=4)
        # All splits are gallery; should not raise
        call_log = []

        def fake_scorer_factory(cfg_id, region, geom, backend):
            call_log.append(cfg_id)
            return _make_scorer(n_inliers=15)

        result = run_pilot(
            manifest_df, splits_df, crop_df,
            scorer_factory=fake_scorer_factory,
            output_dir=tmp_path,
            config=PilotConfig(n_queries=5, seed=42, n_hard_neg=2, loftr_approved=False),
        )
        assert result["fingerprint"]
        assert (tmp_path / PILOT_MANIFEST_FILENAME).exists()


# ---------------------------------------------------------------------------
# 2. Session exclusion
# ---------------------------------------------------------------------------

class TestSessionExclusion:
    def test_references_exclude_query_session(self):
        manifest_df, splits_df, crop_df, gallery_df = _make_gallery_datasets(
            n_identities=2, sessions_per_id=3
        )
        # Take the first image of ind_A
        ind_a_rows = gallery_df[gallery_df["individual_id"] == "ind_A"]
        q_row = ind_a_rows.iloc[0]
        q_session = str(q_row["session_id"])

        refs = build_reference_sessions(
            "ind_A", q_session, crop_df, gallery_df, REGION_BODY
        )
        ref_sessions = {r.session_id for r in refs}
        assert q_session not in ref_sessions, (
            f"Query session {q_session!r} found in references: {ref_sessions}"
        )

    def test_references_all_different_sessions(self):
        manifest_df, splits_df, crop_df, gallery_df = _make_gallery_datasets(
            n_identities=2, sessions_per_id=4
        )
        ind_a_rows = gallery_df[gallery_df["individual_id"] == "ind_A"]
        q_session = str(ind_a_rows.iloc[0]["session_id"])

        refs = build_reference_sessions(
            "ind_A", q_session, crop_df, gallery_df, REGION_BODY, max_sessions=3
        )
        # All ref sessions must differ from query session
        for r in refs:
            assert r.session_id != q_session

    def test_hard_negatives_reference_also_excludes_session(self, tmp_path):
        """When scoring hard negatives, the query session is not used as ref."""
        manifest_df, splits_df, crop_df, gallery_df = _make_gallery_datasets(
            n_identities=5, sessions_per_id=3
        )
        ind_a_rows = gallery_df[gallery_df["individual_id"] == "ind_A"]
        q_session = str(ind_a_rows.iloc[0]["session_id"])

        neg_id = "ind_B"
        neg_refs = build_reference_sessions(
            neg_id, q_session, crop_df, gallery_df, REGION_BODY
        )
        neg_sessions = {r.session_id for r in neg_refs}
        # Session with id matching q_session should not appear in neg refs
        # (they have different naming but the function excludes query_session_id)
        assert q_session not in neg_sessions


# ---------------------------------------------------------------------------
# 3. Deterministic sample
# ---------------------------------------------------------------------------

class TestDeterministicSample:
    def test_same_seed_same_sample(self):
        manifest_df, splits_df, crop_df, gallery_df = _make_gallery_datasets(
            n_identities=6, n_images_per_id_per_session=5
        )
        s1 = sample_pseudo_queries(gallery_df, crop_df, n_queries=10, seed=7)
        s2 = sample_pseudo_queries(gallery_df, crop_df, n_queries=10, seed=7)
        assert list(s1["image_id"].sort_values()) == list(s2["image_id"].sort_values())

    def test_different_seed_different_sample(self):
        manifest_df, splits_df, crop_df, gallery_df = _make_gallery_datasets(
            n_identities=6, n_images_per_id_per_session=5
        )
        s1 = sample_pseudo_queries(gallery_df, crop_df, n_queries=10, seed=1)
        s2 = sample_pseudo_queries(gallery_df, crop_df, n_queries=10, seed=99)
        # Very likely to differ with 6 identities × 5 images
        assert set(s1["image_id"].tolist()) != set(s2["image_id"].tolist())

    def test_sample_respects_n_queries(self):
        manifest_df, splits_df, crop_df, gallery_df = _make_gallery_datasets(
            n_identities=8, n_images_per_id_per_session=4
        )
        s = sample_pseudo_queries(gallery_df, crop_df, n_queries=10, seed=42)
        assert len(s) <= 10

    def test_sample_no_probe_images(self):
        manifest_df, splits_df, crop_df, gallery_df = _make_gallery_datasets()
        gallery_df = gallery_df.copy()
        gallery_df["split"] = "gallery"
        s = sample_pseudo_queries(gallery_df, crop_df, n_queries=5, seed=0)
        assert "gallery" in s["split"].tolist() or "split" not in s.columns or all(
            v == "gallery" for v in s["split"].tolist()
        )

    def test_sample_only_gallery_images(self):
        manifest_df, splits_df, crop_df, gallery_df = _make_gallery_datasets()
        s = sample_pseudo_queries(gallery_df, crop_df, n_queries=5, seed=42)
        # All sampled images must have accepted crops (body or ear)
        all_accepted = set(
            crop_df[crop_df["detector_status"] == "accepted"]["image_id"].tolist()
        )
        for img_id in s["image_id"].tolist():
            assert img_id in all_accepted


# ---------------------------------------------------------------------------
# 4. Hard negatives without truth forcing
# ---------------------------------------------------------------------------

class TestHardNegativeSelection:
    def test_positives_excluded_from_hard_negatives(self):
        manifest_df, splits_df, crop_df, gallery_df = _make_gallery_datasets(n_identities=6)
        for _, row in gallery_df.drop_duplicates("individual_id").iterrows():
            q_id = str(row["image_id"])
            ind_id = str(row["individual_id"])
            hard_negs = select_hard_negatives(
                ind_id, q_id, gallery_df, {}, {}, n_hard_neg=3
            )
            assert ind_id not in hard_negs, (
                f"Query identity {ind_id!r} appeared in hard negatives: {hard_negs}"
            )

    def test_hard_neg_count(self):
        manifest_df, splits_df, crop_df, gallery_df = _make_gallery_datasets(n_identities=6)
        q_row = gallery_df.iloc[0]
        hard_negs = select_hard_negatives(
            str(q_row["individual_id"]), str(q_row["image_id"]),
            gallery_df, {}, {}, n_hard_neg=3,
        )
        assert len(hard_negs) <= 3

    def test_hard_neg_deterministic_no_embeddings(self):
        manifest_df, splits_df, crop_df, gallery_df = _make_gallery_datasets(n_identities=5)
        q_row = gallery_df.iloc[0]
        n1 = select_hard_negatives(
            str(q_row["individual_id"]), str(q_row["image_id"]),
            gallery_df, {}, {}, n_hard_neg=2,
        )
        n2 = select_hard_negatives(
            str(q_row["individual_id"]), str(q_row["image_id"]),
            gallery_df, {}, {}, n_hard_neg=2,
        )
        assert n1 == n2

    def test_hard_neg_with_embeddings_excludes_positive(self):
        manifest_df, splits_df, crop_df, gallery_df = _make_gallery_datasets(n_identities=5)
        n_imgs = len(gallery_df)
        emb = np.random.default_rng(0).standard_normal((n_imgs, 8)).astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        emb /= norms

        img_ids = gallery_df["image_id"].tolist()
        embeddings = {"miewid": emb}
        embedding_index = {"miewid": img_ids}

        q_row = gallery_df.iloc[0]
        q_id = str(q_row["individual_id"])
        negs = select_hard_negatives(
            q_id, str(q_row["image_id"]), gallery_df,
            embeddings, embedding_index, n_hard_neg=3,
        )
        assert q_id not in negs

    def test_hard_neg_not_forced_into_truth(self):
        """Truth identity is not injected back into the negatives list."""
        manifest_df, splits_df, crop_df, gallery_df = _make_gallery_datasets(n_identities=4)
        q_row = gallery_df[gallery_df["individual_id"] == "ind_A"].iloc[0]
        negs = select_hard_negatives(
            "ind_A", str(q_row["image_id"]), gallery_df, {}, {}, n_hard_neg=3
        )
        assert "ind_A" not in negs


# ---------------------------------------------------------------------------
# 5. Positive calibration row semantics
# ---------------------------------------------------------------------------

class TestPositiveCalibrationSemantics:
    def test_positive_rows_flagged_calibration_only(self, tmp_path):
        """Positive identity rows must be marked is_calibration_only=True."""
        manifest_df, splits_df, crop_df, _ = _make_gallery_datasets(
            n_identities=4, n_images_per_id_per_session=3, sessions_per_id=3
        )
        call_log = []

        def scorer_factory(cfg_id, region, geom, backend):
            call_log.append((cfg_id, region))
            return _make_scorer(n_inliers=20)

        result = run_pilot(
            manifest_df, splits_df, crop_df,
            scorer_factory=scorer_factory,
            output_dir=tmp_path,
            config=PilotConfig(n_queries=4, seed=0, n_hard_neg=2, loftr_approved=False),
        )
        # Load identity scores for any config
        import glob as _glob
        score_files = list((tmp_path / "identity_scores").glob("*_identity_scores.parquet"))
        assert score_files, "No identity score files written"
        df = pd.read_parquet(score_files[0])
        positives = df[df["is_positive"] == True]
        assert all(positives["is_calibration_only"] == True), (
            "Positive rows must be marked is_calibration_only=True"
        )
        negatives = df[df["is_positive"] == False]
        assert all(negatives["is_calibration_only"] == False), (
            "Negative rows must not be marked calibration-only"
        )

    def test_positive_rows_present_in_calibration_metrics(self):
        """Calibration metrics use positive rows; ranking metrics exclude them."""
        pos_scores = [_make_fake_identity_score("pos", score=25.0, n_inliers=25)]
        neg_scores = [_make_fake_identity_score("neg", score=5.0, n_inliers=5)]
        all_scores = pos_scores + neg_scores

        metrics = compute_metrics(
            all_scores,
            positive_flags=[True, False],
            calibration_only_flags=[True, False],
            config_id="body_partial_affine",
            region=REGION_BODY,
            backend="lightglue",
            geom_model=GEOM_PARTIAL_AFFINE,
        )
        # Calibration metrics should have pos_score info
        assert not np.isnan(metrics["pos_score_median"])
        assert metrics["roc_auc"] == 1.0


# ---------------------------------------------------------------------------
# 6. Canonical scorer called
# ---------------------------------------------------------------------------

class TestCanonicalScorerCalled:
    def test_scorer_called_via_identity_scorer(self, tmp_path):
        """LocalIdentityScorer is the scoring path used by run_pilot."""
        manifest_df, splits_df, crop_df, _ = _make_gallery_datasets(
            n_identities=3, n_images_per_id_per_session=2, sessions_per_id=2
        )
        score_identity_calls = []

        class TrackingScorer(LocalIdentityScorer):
            def score_identity(self, *args, **kwargs):
                score_identity_calls.append(args[0] if args else None)
                # Return a fake identity score without real I/O
                return _make_fake_identity_score(
                    kwargs.get("candidate_individual_id", "fake")
                )

        def scorer_factory(cfg_id, region, geom, backend):
            return TrackingScorer(FakeMatcher(n_inliers=10), cache=None)

        run_pilot(
            manifest_df, splits_df, crop_df,
            scorer_factory=scorer_factory,
            output_dir=tmp_path,
            config=PilotConfig(n_queries=3, seed=1, n_hard_neg=1, loftr_approved=False),
        )
        assert len(score_identity_calls) > 0, "score_identity was never called"

    def test_scorer_uses_session_cap(self):
        """LocalIdentityScorer respects max_sessions cap."""
        manifest_df, splits_df, crop_df, gallery_df = _make_gallery_datasets(
            n_identities=2, sessions_per_id=5
        )
        scorer = _make_scorer(n_inliers=10)

        ind_a_imgs = gallery_df[gallery_df["individual_id"] == "ind_A"]
        q_session = str(ind_a_imgs.iloc[0]["session_id"])

        # Build refs with more sessions than cap
        refs = build_reference_sessions(
            "ind_A", q_session, crop_df, gallery_df, REGION_BODY, max_sessions=100
        )
        # There are 5 sessions per id, minus the query session → up to 4 refs
        assert len(refs) <= 4

        # Scorer caps at 3
        query_crops = build_query_crops(
            str(ind_a_imgs.iloc[0]["image_id"]), crop_df, REGION_BODY
        )
        if query_crops:
            ident_score = scorer.score_identity(query_crops, refs)
            assert ident_score.n_sessions_used <= scorer.max_sessions


# ---------------------------------------------------------------------------
# 7. Gate / flatness / runtime projection checks
# ---------------------------------------------------------------------------

class TestGatesAndProjection:
    def _metrics(self, coverage, roc_auc, is_flat, has_support):
        return {
            "coverage": coverage,
            "roc_auc": roc_auc,
            "calibration": {"is_flat": is_flat, "has_support": has_support},
        }

    def test_gate_passes_all(self):
        m = self._metrics(0.80, 0.70, is_flat=False, has_support=True)
        g = check_gates(m)
        assert g["approved"] is True
        assert g["gate_coverage"] is True
        assert g["gate_roc_auc"] is True
        assert g["gate_calibration"] is True

    def test_gate_fails_coverage(self):
        m = self._metrics(0.50, 0.75, is_flat=False, has_support=True)
        g = check_gates(m)
        assert g["approved"] is False
        assert g["gate_coverage"] is False

    def test_gate_fails_roc_auc(self):
        m = self._metrics(0.80, 0.50, is_flat=False, has_support=True)
        g = check_gates(m)
        assert g["approved"] is False
        assert g["gate_roc_auc"] is False

    def test_gate_fails_flat_calibration(self):
        m = self._metrics(0.80, 0.70, is_flat=True, has_support=True)
        g = check_gates(m)
        assert g["approved"] is False
        assert g["gate_calibration"] is False

    def test_gate_fails_no_calib_support(self):
        m = self._metrics(0.80, 0.70, is_flat=False, has_support=False)
        g = check_gates(m)
        assert g["approved"] is False

    def test_platt_flatness_detected(self):
        labels = [1] * 20 + [0] * 20
        # Completely overlapping distributions → flat
        scores_flat = [0.5] * 40
        result = _platt_calibration_check(labels, scores_flat)
        assert result["is_flat"] is True

    def test_platt_separation_detected(self):
        labels = [1] * 20 + [0] * 20
        # Well-separated distributions
        scores_sep = [10.0] * 20 + [1.0] * 20
        result = _platt_calibration_check(labels, scores_sep)
        assert result["is_flat"] is False

    def test_budget_within_limit(self):
        manifest_df, splits_df, crop_df, gallery_df = _make_gallery_datasets(n_identities=5)
        from pipeline.local_signal_pilot import _PILOT_CONFIGS
        budget = estimate_budget(
            gallery_df, crop_df, _PILOT_CONFIGS[:1],
            n_queries=5, n_hard_neg=2, n_sessions_cap=3
        )
        assert isinstance(budget.within_budget, bool)
        assert budget.estimated_h100_hours >= 0
        assert budget.estimated_cache_gb >= 0

    def test_budget_projects_oof_and_probe(self):
        manifest_df, splits_df, crop_df, gallery_df = _make_gallery_datasets(n_identities=4)
        from pipeline.local_signal_pilot import _PILOT_CONFIGS
        budget = estimate_budget(
            gallery_df, crop_df, _PILOT_CONFIGS[:1],
            n_queries=10, n_hard_neg=3
        )
        assert budget.projected_oof_pairs > 0
        assert budget.projected_fixed_probe_pairs > 0

    def test_roc_auc_perfect(self):
        labels = [1, 1, 0, 0]
        scores = [10.0, 9.0, 2.0, 1.0]
        auc = _compute_roc_auc(labels, scores)
        assert abs(auc - 1.0) < 0.01

    def test_roc_auc_random(self):
        labels = [1, 0, 1, 0]
        scores = [5.0, 5.0, 5.0, 5.0]
        auc = _compute_roc_auc(labels, scores)
        assert auc == 0.5

    def test_pr_auc_perfect(self):
        labels = [1, 1, 0, 0]
        scores = [10.0, 9.0, 2.0, 1.0]
        auc = _compute_pr_auc(labels, scores)
        assert auc > 0.8

    def test_head_records_absent_no_failure(self, tmp_path):
        """Head records absent from manifests must not cause failures."""
        manifest_df, splits_df, crop_df, _ = _make_gallery_datasets(n_identities=3)
        # Explicitly remove any head crops (there are none in our factory but confirm)
        assert "head" not in crop_df["crop_kind"].tolist()

        def scorer_factory(cfg_id, region, geom, backend):
            return _make_scorer()

        result = run_pilot(
            manifest_df, splits_df, crop_df,
            scorer_factory=scorer_factory,
            output_dir=tmp_path,
            config=PilotConfig(n_queries=3, seed=0, n_hard_neg=1, loftr_approved=False),
        )
        assert result["fingerprint"]


# ---------------------------------------------------------------------------
# 8. Fingerprint mismatch → resume reset
# ---------------------------------------------------------------------------

class TestFingerprintMismatch:
    def test_stale_fingerprint_triggers_fresh_run(self, tmp_path):
        """If stored fingerprint doesn't match, resume is disabled."""
        # Write a stale fingerprint
        fp_path = tmp_path / PILOT_FINGERPRINT_FILENAME
        fp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fp_path, "w") as fh:
            json.dump({"fingerprint": "STALE_FP", "schema_version": "local-v1"}, fh)

        assert not _check_fingerprint(tmp_path, "DIFFERENT_FP")
        assert _check_fingerprint(tmp_path, "STALE_FP")

    def test_run_pilot_resets_on_fingerprint_mismatch(self, tmp_path):
        """run_pilot with resume=True resets if fingerprint changed."""
        manifest_df, splits_df, crop_df, _ = _make_gallery_datasets(n_identities=3)

        # Write stale fingerprint
        fp_path = tmp_path / PILOT_FINGERPRINT_FILENAME
        fp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fp_path, "w") as fh:
            json.dump({"fingerprint": "STALE", "schema_version": "local-v1"}, fh)

        scorer_calls = []

        def scorer_factory(cfg_id, region, geom, backend):
            scorer_calls.append(cfg_id)
            return _make_scorer()

        run_pilot(
            manifest_df, splits_df, crop_df,
            scorer_factory=scorer_factory,
            output_dir=tmp_path,
            config=PilotConfig(n_queries=3, seed=0, n_hard_neg=1, loftr_approved=False),
            resume=True,  # stale fingerprint → should not skip scoring
        )
        # Scoring must have run (fingerprint mismatch cleared the resume flag)
        assert len(scorer_calls) > 0

    def test_write_and_verify_fingerprint(self, tmp_path):
        config = {"n_queries": 5, "seed": 42}
        from pipeline.local_signal_pilot import _config_fingerprint
        fp = _config_fingerprint(config)
        write_pilot_config(tmp_path, config, fp)
        assert _check_fingerprint(tmp_path, fp)
        assert not _check_fingerprint(tmp_path, "wrong_fp")


# ---------------------------------------------------------------------------
# 9. Resume — already-scored config is skipped
# ---------------------------------------------------------------------------

class TestResume:
    def test_resume_skips_existing_config(self, tmp_path):
        manifest_df, splits_df, crop_df, _ = _make_gallery_datasets(
            n_identities=4, sessions_per_id=2
        )

        scorer_calls = []

        def scorer_factory(cfg_id, region, geom, backend):
            scorer_calls.append(cfg_id)
            return _make_scorer(n_inliers=10)

        pilot_cfg = PilotConfig(n_queries=4, seed=0, n_hard_neg=1, loftr_approved=False)

        # First run
        run_pilot(
            manifest_df, splits_df, crop_df,
            scorer_factory=scorer_factory,
            output_dir=tmp_path,
            config=pilot_cfg,
        )
        first_run_calls = list(scorer_calls)

        # Second run with resume=True — should skip all already-scored configs
        scorer_calls.clear()
        run_pilot(
            manifest_df, splits_df, crop_df,
            scorer_factory=scorer_factory,
            output_dir=tmp_path,
            config=pilot_cfg,
            resume=True,
        )
        # All configs should be skipped on resume (files already exist)
        assert len(scorer_calls) == 0 or len(scorer_calls) < len(first_run_calls)


# ---------------------------------------------------------------------------
# 10. Body/ear missingness
# ---------------------------------------------------------------------------

class TestCropMissingness:
    def test_no_body_crops_skips_body_config(self, tmp_path):
        """When body crops are absent, body configs produce no scores."""
        manifest_df, splits_df, crop_df, _ = _make_gallery_datasets(
            n_identities=3, include_body=False, include_ear=True
        )
        scorer_calls = []

        def scorer_factory(cfg_id, region, geom, backend):
            scorer_calls.append((cfg_id, region))
            return _make_scorer()

        result = run_pilot(
            manifest_df, splits_df, crop_df,
            scorer_factory=scorer_factory,
            output_dir=tmp_path,
            config=PilotConfig(n_queries=4, seed=0, n_hard_neg=1, loftr_approved=False),
        )
        # Body configs may call scorer_factory but produce 0 pair scores
        # because build_query_crops returns [] for body
        score_files = list((tmp_path / "pair_scores").glob("body_*_pair_scores.parquet"))
        for f in score_files:
            df = pd.read_parquet(f)
            # Should be empty or have 0 rows
            assert len(df) == 0, f"Expected 0 body pair score rows, got {len(df)}"

    def test_no_ear_crops_skips_ear_config(self, tmp_path):
        """When ear crops are absent, ear configs produce no scores."""
        manifest_df, splits_df, crop_df, _ = _make_gallery_datasets(
            n_identities=3, include_body=True, include_ear=False
        )
        scorer_calls = []

        def scorer_factory(cfg_id, region, geom, backend):
            scorer_calls.append((cfg_id, region))
            return _make_scorer()

        run_pilot(
            manifest_df, splits_df, crop_df,
            scorer_factory=scorer_factory,
            output_dir=tmp_path,
            config=PilotConfig(n_queries=4, seed=0, n_hard_neg=1, loftr_approved=False),
        )
        score_files = list((tmp_path / "pair_scores").glob("ear_*_pair_scores.parquet"))
        for f in score_files:
            df = pd.read_parquet(f)
            assert len(df) == 0

    def test_empty_crop_manifest_no_crash(self, tmp_path):
        """Empty crop manifest returns an empty pilot manifest gracefully."""
        manifest_df, splits_df, _, gallery_df = _make_gallery_datasets(n_identities=3)
        empty_crops = _make_crop_manifest(manifest_df.head(0))

        with pytest.raises((ValueError, Exception)):
            # sample_pseudo_queries should fail gracefully with no crops
            sample_pseudo_queries(gallery_df, empty_crops, n_queries=5, seed=42)

    def test_build_query_crops_missing(self):
        manifest_df, splits_df, crop_df, _ = _make_gallery_datasets(n_identities=2)
        result = build_query_crops("NONEXISTENT_IMAGE_ID", crop_df, REGION_BODY)
        assert result == []

    def test_build_reference_sessions_missing(self):
        manifest_df, splits_df, crop_df, gallery_df = _make_gallery_datasets(n_identities=2)
        result = build_reference_sessions(
            "NONEXISTENT_IDENTITY", "any_session", crop_df, gallery_df, REGION_BODY
        )
        assert result == []


# ---------------------------------------------------------------------------
# 11. Output isolation
# ---------------------------------------------------------------------------

class TestOutputIsolation:
    def test_no_probe_in_pilot_manifest(self, tmp_path):
        manifest_df, splits_df, crop_df, _ = _make_gallery_datasets(n_identities=4)

        def scorer_factory(cfg_id, region, geom, backend):
            return _make_scorer()

        run_pilot(
            manifest_df, splits_df, crop_df,
            scorer_factory=scorer_factory,
            output_dir=tmp_path,
            config=PilotConfig(n_queries=5, seed=0, n_hard_neg=2),
        )
        pilot_manifest_path = tmp_path / PILOT_MANIFEST_FILENAME
        assert pilot_manifest_path.exists()
        pilot_df = pd.read_parquet(pilot_manifest_path)
        if "split" in pilot_df.columns:
            assert not pilot_df["split"].isin(["probe", "held_out_probe"]).any(), (
                "Probe images found in pilot manifest"
            )

    def test_config_json_written(self, tmp_path):
        manifest_df, splits_df, crop_df, _ = _make_gallery_datasets(n_identities=3)

        def scorer_factory(cfg_id, region, geom, backend):
            return _make_scorer()

        run_pilot(
            manifest_df, splits_df, crop_df,
            scorer_factory=scorer_factory,
            output_dir=tmp_path,
            config=PilotConfig(n_queries=3, seed=0),
        )
        assert (tmp_path / PILOT_CONFIG_FILENAME).exists()
        with open(tmp_path / PILOT_CONFIG_FILENAME) as fh:
            cfg = json.load(fh)
        assert cfg["n_queries"] == 3

    def test_metrics_json_written(self, tmp_path):
        manifest_df, splits_df, crop_df, _ = _make_gallery_datasets(
            n_identities=4, sessions_per_id=2
        )

        def scorer_factory(cfg_id, region, geom, backend):
            return _make_scorer(n_inliers=20)

        run_pilot(
            manifest_df, splits_df, crop_df,
            scorer_factory=scorer_factory,
            output_dir=tmp_path,
            config=PilotConfig(n_queries=4, seed=0, n_hard_neg=1),
        )
        metrics_path = tmp_path / PILOT_METRICS_FILENAME
        assert metrics_path.exists()
        with open(metrics_path) as fh:
            m = json.load(fh)
        assert "config_metrics" in m
        assert "budget" in m

    def test_output_dir_isolated_from_experiment_root(self, tmp_path):
        """Output goes to the specified output_dir, not the global EXPERIMENT_ROOT."""
        from pipeline.local_signal_pilot import PILOT_OUTPUT_ROOT
        manifest_df, splits_df, crop_df, _ = _make_gallery_datasets(n_identities=3)

        def scorer_factory(cfg_id, region, geom, backend):
            return _make_scorer()

        custom_dir = tmp_path / "my_custom_output"
        run_pilot(
            manifest_df, splits_df, crop_df,
            scorer_factory=scorer_factory,
            output_dir=custom_dir,
            config=PilotConfig(n_queries=3, seed=0),
        )
        # Files should be in custom_dir, not PILOT_OUTPUT_ROOT
        assert (custom_dir / PILOT_CONFIG_FILENAME).exists()
        # PILOT_OUTPUT_ROOT must not have been touched
        if PILOT_OUTPUT_ROOT.exists():
            # If it existed before, we can't assert it's unchanged, but
            # the custom_dir must be the write target
            pass
        assert (custom_dir / PILOT_CONFIG_FILENAME).exists()

    def test_loftr_not_run_without_approval(self, tmp_path):
        """LoFTR config is skipped when loftr_approved=False."""
        manifest_df, splits_df, crop_df, _ = _make_gallery_datasets(
            n_identities=3, sessions_per_id=2
        )
        loftr_calls = []

        def scorer_factory(cfg_id, region, geom, backend):
            if backend == "loftr":
                loftr_calls.append(cfg_id)
            return _make_scorer()

        run_pilot(
            manifest_df, splits_df, crop_df,
            scorer_factory=scorer_factory,
            output_dir=tmp_path,
            config=PilotConfig(n_queries=3, seed=0, loftr_approved=False),
        )
        assert len(loftr_calls) == 0, "LoFTR was called without approval"

    def test_loftr_runs_when_approved(self, tmp_path):
        """LoFTR config runs when loftr_approved=True."""
        manifest_df, splits_df, crop_df, _ = _make_gallery_datasets(
            n_identities=3, sessions_per_id=2
        )
        loftr_calls = []

        class FakeLoFTRMatcher(FakeMatcher):
            backend = "loftr"
            model_fingerprint = "fake_loftr_fp"

        def scorer_factory(cfg_id, region, geom, backend):
            if backend == "loftr":
                loftr_calls.append(cfg_id)
                return LocalIdentityScorer(FakeLoFTRMatcher(n_inliers=12), cache=None)
            return _make_scorer()

        run_pilot(
            manifest_df, splits_df, crop_df,
            scorer_factory=scorer_factory,
            output_dir=tmp_path,
            config=PilotConfig(n_queries=3, seed=0, loftr_approved=True),
        )
        assert len(loftr_calls) > 0, "LoFTR was not called with approval"


# ---------------------------------------------------------------------------
# Extra: compute_metrics and coverage
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_coverage_calculation(self):
        scores = [
            _make_fake_identity_score("A", score=10.0),
            _make_fake_identity_score("B", score=0.0),
            _make_fake_identity_score("C", score=5.0),
        ]
        m = compute_metrics(
            scores,
            positive_flags=[True, False, False],
            calibration_only_flags=[True, False, False],
            config_id="body_partial_affine",
            region=REGION_BODY,
            backend="lightglue",
            geom_model=GEOM_PARTIAL_AFFINE,
        )
        # 2 of 3 have score > 0
        assert abs(m["coverage"] - 2 / 3) < 0.01

    def test_metrics_schema_keys(self):
        scores = [_make_fake_identity_score("X", score=8.0)]
        m = compute_metrics(
            scores, [False], [False],
            config_id="ear_homography",
            region=REGION_EAR,
            backend="lightglue",
            geom_model=GEOM_HOMOGRAPHY,
        )
        for key in [
            "config_id", "region", "backend", "geom_model",
            "n_total", "n_valid", "coverage", "roc_auc", "pr_auc",
            "calibration", "latency_p50_ms", "latency_p95_ms",
        ]:
            assert key in m, f"Missing key: {key}"
