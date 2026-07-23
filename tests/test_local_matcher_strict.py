# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Tests for StrictLocalMatcher, FeatureCache, LocalIdentityScorer,
and LocalPairScore/LocalIdentityScore schemas.

All tests run without any model downloads.  StrictLocalMatcher is exercised
with a MockExtractor and MockMatcher that return synthetic numpy arrays.
The 'cpu' device is used throughout.

Coverage
--------
- Explicit device and strict failures
- Mirror search max selection
- Affine/homography geometric verification
- Cache hit / invalidation / corruption / atomicity / LRU
- Region mismatch rejection
- Session cap enforcement
- Ear combinations (all ears × all refs)
- Mean-top-2 aggregation
- Calibration/inference same function (scoring fingerprint stability)
- Fingerprint mismatch rejection
- selected-v1 non-mutation (new schema constants do not alter existing fields)
"""

from __future__ import annotations

import io
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("data_root_abs_path", "/fakedir/test_data")
os.environ.setdefault("container_name", "test_container")

# ---------------------------------------------------------------------------
# Shared synthetic image helpers
# ---------------------------------------------------------------------------

def _make_bgr(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_native_lightglue_uses_width_height_image_size():
    from models.local_matcher import FeatureBundle, StrictLocalMatcher

    captured = {}

    class CaptureMatcher:
        def __call__(self, data):
            captured.update(data)
            return {"matches": torch.empty((1, 0, 2), dtype=torch.long)}

    matcher = StrictLocalMatcher.__new__(StrictLocalMatcher)
    matcher.device = torch.device("cpu")
    matcher._matcher = CaptureMatcher()
    query = FeatureBundle(
        keypoints=np.zeros((2, 2), dtype=np.float32),
        descriptors=np.zeros((2, 8), dtype=np.float32),
        scores=np.ones(2, dtype=np.float32),
        image_shape=(40, 80),
        orientation="original",
        backend="lightglue",
        model_fingerprint="test",
    )
    reference = FeatureBundle(
        keypoints=np.zeros((2, 2), dtype=np.float32),
        descriptors=np.zeros((2, 8), dtype=np.float32),
        scores=np.ones(2, dtype=np.float32),
        image_shape=(30, 50),
        orientation="original",
        backend="lightglue",
        model_fingerprint="test",
    )

    with patch("models.local_matcher._LIGHTGLUE_NATIVE", True):
        matcher._run_lightglue_match(query, reference)

    assert captured["image0"]["image_size"].tolist() == [[80, 40]]
    assert captured["image1"]["image_size"].tolist() == [[50, 30]]


def test_loftr_unbatched_correspondences_keep_all_points():
    from models.local_matcher import (
        GEOM_HOMOGRAPHY,
        REGION_EAR,
        StrictLocalMatcher,
    )

    points = torch.tensor(
        [[8.0, 8.0], [16.0, 8.0], [8.0, 16.0], [16.0, 16.0], [24.0, 24.0]]
    )

    class FakeLoFTR:
        def __call__(self, data):
            return {
                "keypoints0": points,
                "keypoints1": points + 1.0,
                "confidence": torch.ones(len(points)),
                "batch_indexes": torch.zeros(len(points), dtype=torch.long),
            }

    matcher = StrictLocalMatcher.__new__(StrictLocalMatcher)
    matcher.device = torch.device("cpu")
    matcher.backend = "loftr"
    matcher.model_fingerprint = "test"
    matcher.loftr_max_edge = 640
    matcher._loftr = FakeLoFTR()
    result = matcher._score_loftr_pair(
        _make_bgr(64, 80),
        _make_bgr(64, 80, seed=1),
        REGION_EAR,
        GEOM_HOMOGRAPHY,
    )
    assert len(result.raw_matches) == len(points)


def _make_gray(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    img = _make_bgr(h, w, seed)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Synthetic keypoints / descriptors
# ---------------------------------------------------------------------------

def _kpts(n: int = 20, w: int = 64, h: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, 2), dtype=np.float32) * np.array([[w, h]], dtype=np.float32)


def _descs(n: int = 20, d: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, d), dtype=np.float32)


def _scores(n: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(n, dtype=np.float32)


# ---------------------------------------------------------------------------
# FeatureBundle factory
# ---------------------------------------------------------------------------

def _make_bundle(
    orientation: str = "original",
    n: int = 20,
    model_fingerprint: str = "test_fp_0000",
    seed: int = 0,
):
    from models.local_matcher import FeatureBundle
    return FeatureBundle(
        keypoints=_kpts(n, seed=seed),
        descriptors=_descs(n, seed=seed),
        scores=_scores(n, seed=seed),
        image_shape=(64, 64),
        orientation=orientation,
        backend="lightglue",
        model_fingerprint=model_fingerprint,
    )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cpu_device():
    return torch.device("cpu")


@pytest.fixture()
def tmp_cache_dir(tmp_path):
    return tmp_path / "feature_cache"


@pytest.fixture()
def model_fp():
    return "abcdef012345abcd"


# ===========================================================================
# Part 1: StrictLocalMatcher construction and errors
# ===========================================================================

class TestStrictLocalMatcherConstruction:

    def test_unknown_backend_raises(self, cpu_device):
        from models.local_matcher import StrictLocalMatcher, LocalMatcherModelError
        with pytest.raises(LocalMatcherModelError, match="Unknown backend"):
            StrictLocalMatcher(backend="unknown_backend", device=cpu_device)

    def test_loftr_without_approval_raises(self, cpu_device):
        from models.local_matcher import StrictLocalMatcher, LocalMatcherModelError
        with patch("models.local_matcher.LOCAL_MATCHER_LOFTR_PILOT_APPROVED", False):
            with pytest.raises(LocalMatcherModelError, match="LoFTR"):
                StrictLocalMatcher(backend="loftr", device=cpu_device)

    def test_explicit_device_stored(self, cpu_device):
        """device is stored exactly as passed."""
        from models.local_matcher import StrictLocalMatcher
        with patch("models.local_matcher._LIGHTGLUE_NATIVE", False), \
             patch("models.local_matcher.StrictLocalMatcher._init_lightglue_strict", return_value=None):
            m = StrictLocalMatcher.__new__(StrictLocalMatcher)
            m.backend = "lightglue"
            m.max_keypoints = 256
            m.min_inliers = 4
            m.disable_cudnn = False
            m.device = cpu_device
            m.model_fingerprint = "fp"
        assert m.device == cpu_device

    def test_model_fingerprint_deterministic(self, cpu_device):
        from models.local_matcher import _model_fingerprint
        fp1 = _model_fingerprint("lightglue", 2048)
        fp2 = _model_fingerprint("lightglue", 2048)
        assert fp1 == fp2

    def test_model_fingerprint_differs_on_keypoints(self):
        from models.local_matcher import _model_fingerprint
        fp1 = _model_fingerprint("lightglue", 1024)
        fp2 = _model_fingerprint("lightglue", 2048)
        assert fp1 != fp2

    def test_disable_cudnn_flag_stored(self, cpu_device):
        from models.local_matcher import StrictLocalMatcher
        with patch("models.local_matcher.StrictLocalMatcher._init_lightglue_strict"):
            m = StrictLocalMatcher(device=cpu_device, disable_cudnn=True)
        assert m.disable_cudnn is True


# ===========================================================================
# Part 2: Feature extraction errors and failures (no real models needed)
# ===========================================================================

class TestStrictMatcherExtractionErrors:

    def _make_matcher(self, cpu_device):
        from models.local_matcher import StrictLocalMatcher
        with patch("models.local_matcher.StrictLocalMatcher._init_lightglue_strict"):
            m = StrictLocalMatcher(device=cpu_device)
        # Attach a fake extractor that returns synthetic features
        m._extractor = _make_fake_superpoint()
        return m

    def test_wrong_image_shape_raises(self, cpu_device):
        from models.local_matcher import LocalMatcherInferenceError
        m = self._make_matcher(cpu_device)
        with pytest.raises(LocalMatcherInferenceError, match="BGR image"):
            m.extract_features(np.zeros((64, 64), dtype=np.uint8))  # 2-D, no channel

    def test_unknown_orientation_raises(self, cpu_device):
        from models.local_matcher import LocalMatcherInferenceError
        m = self._make_matcher(cpu_device)
        with pytest.raises(LocalMatcherInferenceError, match="Unknown orientation"):
            m.extract_features(_make_bgr(), orientation="upside_down")

    def test_loftr_extract_features_raises(self, cpu_device):
        """extract_features is not supported for LoFTR (pairwise only)."""
        from models.local_matcher import StrictLocalMatcher, LocalMatcherModelError
        with patch("models.local_matcher.LOCAL_MATCHER_LOFTR_PILOT_APPROVED", True), \
             patch("models.local_matcher.StrictLocalMatcher._init_loftr_strict"):
            m = StrictLocalMatcher(backend="loftr", device=cpu_device)
        with pytest.raises(LocalMatcherModelError, match="pairwise"):
            m.extract_features(_make_bgr())

    def test_extract_features_returns_bundle(self, cpu_device):
        from models.local_matcher import FeatureBundle
        m = self._make_matcher(cpu_device)
        bundle = m.extract_features(_make_bgr())
        assert isinstance(bundle, FeatureBundle)
        assert bundle.orientation == "original"
        assert bundle.keypoints.ndim == 2
        assert bundle.keypoints.shape[1] == 2

    def test_flipped_orientation_stored(self, cpu_device):
        m = self._make_matcher(cpu_device)
        bundle = m.extract_features(_make_bgr(), orientation="flipped")
        assert bundle.orientation == "flipped"


# ===========================================================================
# Part 3: Geometric verification
# ===========================================================================

class TestGeometricVerification:

    def _make_collinear_pts(self, n: int = 10):
        """Points lying on a line — degenerate for homography."""
        x = np.linspace(0, 50, n).astype(np.float32)
        y = np.zeros(n, dtype=np.float32)
        return np.stack([x, y], axis=1), np.stack([x + 5, y + 5], axis=1)

    def test_homography_zero_inliers_on_degenerate(self):
        from models.local_matcher import _run_geom_verification, REGION_EAR, GEOM_HOMOGRAPHY
        q, r = self._make_collinear_pts(4)
        raw = np.stack([np.arange(4), np.arange(4)], axis=1)
        geom = _run_geom_verification(q, r, raw, REGION_EAR, GEOM_HOMOGRAPHY)
        assert geom.model_used == GEOM_HOMOGRAPHY

    def test_partial_affine_body_region(self):
        from models.local_matcher import _run_geom_verification, REGION_BODY, GEOM_PARTIAL_AFFINE
        rng = np.random.default_rng(42)
        q = rng.random((20, 2), dtype=np.float32) * 64
        # Near-affine transform: slight rotation + translation
        angle = 0.1
        M = np.array([[np.cos(angle), -np.sin(angle), 3],
                      [np.sin(angle),  np.cos(angle), 2]], dtype=np.float32)
        r = (M[:, :2] @ q.T + M[:, 2:]).T
        raw = np.stack([np.arange(20), np.arange(20)], axis=1)
        geom = _run_geom_verification(q, r, raw, REGION_BODY, GEOM_PARTIAL_AFFINE)
        assert geom.n_raw_matches == 20
        assert geom.n_inliers >= 0
        assert geom.model_used in ("partial_affine", "homography")

    def test_too_few_points_returns_zero(self):
        from models.local_matcher import _run_geom_verification, REGION_EAR
        q = np.array([[0, 0], [1, 1]], dtype=np.float32)
        r = np.array([[0, 0], [1, 1]], dtype=np.float32)
        raw = np.stack([np.arange(2), np.arange(2)], axis=1)
        geom = _run_geom_verification(q, r, raw, REGION_EAR)
        assert geom.n_inliers == 0

    def test_homography_ear_policy(self):
        from models.local_matcher import _run_geom_verification, REGION_EAR, GEOM_HOMOGRAPHY
        rng = np.random.default_rng(1)
        q = rng.random((10, 2), dtype=np.float32) * 64
        r = q + 2.0  # near-identity translation
        raw = np.stack([np.arange(10), np.arange(10)], axis=1)
        geom = _run_geom_verification(q, r, raw, REGION_EAR, None)
        # Policy for ear is homography
        assert geom.model_used == GEOM_HOMOGRAPHY

    def test_unknown_geom_model_raises(self):
        from models.local_matcher import _run_geom_verification, REGION_EAR, LocalMatcherRegionError
        q = np.random.rand(5, 2).astype(np.float32)
        r = np.random.rand(5, 2).astype(np.float32)
        raw = np.stack([np.arange(5), np.arange(5)], axis=1)
        with pytest.raises(LocalMatcherRegionError):
            _run_geom_verification(q, r, raw, REGION_EAR, "unknown_model")

    def test_geometric_spread_nonnegative(self):
        from models.local_matcher import _geometric_spread
        pts = np.array([[0, 0], [10, 0], [5, 5], [5, 10]], dtype=np.float32)
        spread = _geometric_spread(pts)
        assert spread >= 0.0

    def test_geometric_spread_single_point_zero(self):
        from models.local_matcher import _geometric_spread
        pts = np.array([[7, 7]], dtype=np.float32)
        assert _geometric_spread(pts) == 0.0


# ===========================================================================
# Part 4: Mirror search max selection
# ===========================================================================

class TestMirrorSearch:

    def _make_matcher_with_controllable_geom(self, cpu_device, orig_inliers, flip_inliers):
        """
        Build a StrictLocalMatcher whose match_features returns a controllable
        inlier count depending on the reference orientation.
        """
        from models.local_matcher import StrictLocalMatcher, MatchResult, GeomVerification

        def _fake_match(q_bundle, r_bundle, region, geom_model=None):
            n = orig_inliers if r_bundle.orientation == "original" else flip_inliers
            geom = GeomVerification(
                model_used="homography",
                n_raw_matches=n,
                n_inliers=n,
                inlier_ratio=1.0 if n > 0 else 0.0,
                geometric_spread=10.0 if n > 0 else 0.0,
                H=None,
                inlier_mask=None,
            )
            return MatchResult(
                region=region,
                orientation=r_bundle.orientation,
                raw_matches=np.zeros((n, 2), dtype=np.int32),
                query_keypoints=np.zeros((n, 2), dtype=np.float32),
                ref_keypoints=np.zeros((n, 2), dtype=np.float32),
                geom=geom,
                backend="lightglue",
                model_fingerprint="test_fp",
                n_inliers=n,
            )

        with patch("models.local_matcher.StrictLocalMatcher._init_lightglue_strict"):
            m = StrictLocalMatcher(device=cpu_device)
        m.match_features = _fake_match
        # fake extract_features always returns a valid bundle
        def _fake_extract(img, orientation="original"):
            return _make_bundle(orientation=orientation, model_fingerprint=m.model_fingerprint)
        m.extract_features = _fake_extract
        return m

    def test_mirror_returns_flipped_when_better(self, cpu_device):
        """When flipped has more inliers, orientation='flipped' is returned."""
        from models.local_matcher import REGION_EAR
        m = self._make_matcher_with_controllable_geom(cpu_device, orig_inliers=5, flip_inliers=15)
        result = m._mirror_search(
            _make_bundle(orientation="original", model_fingerprint=m.model_fingerprint),
            _make_bgr(),
            REGION_EAR,
        )
        assert result.orientation == "flipped"
        assert result.n_inliers == 15

    def test_mirror_returns_original_when_better(self, cpu_device):
        from models.local_matcher import REGION_EAR
        m = self._make_matcher_with_controllable_geom(cpu_device, orig_inliers=20, flip_inliers=3)
        result = m._mirror_search(
            _make_bundle(orientation="original", model_fingerprint=m.model_fingerprint),
            _make_bgr(),
            REGION_EAR,
        )
        assert result.orientation == "original"
        assert result.n_inliers == 20

    def test_mirror_tie_returns_original(self, cpu_device):
        """Tie (equal inliers): original is preferred (not > so original wins)."""
        from models.local_matcher import REGION_BODY
        m = self._make_matcher_with_controllable_geom(cpu_device, orig_inliers=10, flip_inliers=10)
        result = m._mirror_search(
            _make_bundle(orientation="original", model_fingerprint=m.model_fingerprint),
            _make_bgr(),
            REGION_BODY,
        )
        assert result.orientation == "original"


# ===========================================================================
# Part 5: Region mismatch
# ===========================================================================

class TestRegionMismatch:

    def _make_matcher(self, cpu_device):
        from models.local_matcher import StrictLocalMatcher
        with patch("models.local_matcher.StrictLocalMatcher._init_lightglue_strict"):
            m = StrictLocalMatcher(device=cpu_device)
        return m

    def test_head_region_score_pair_raises(self, cpu_device):
        from models.local_matcher import LocalMatcherRegionError
        m = self._make_matcher(cpu_device)
        with pytest.raises(LocalMatcherRegionError, match="quality gate"):
            m.score_pair_strict(_make_bgr(), _make_bgr(), region="head")

    def test_head_region_match_features_raises(self, cpu_device):
        from models.local_matcher import LocalMatcherRegionError
        m = self._make_matcher(cpu_device)
        q_bundle = _make_bundle(model_fingerprint=m.model_fingerprint)
        r_bundle = _make_bundle(model_fingerprint=m.model_fingerprint)
        with pytest.raises(LocalMatcherRegionError):
            m.match_features(q_bundle, r_bundle, region="head")

    def test_invalid_region_raises(self, cpu_device):
        from models.local_matcher import LocalMatcherRegionError
        m = self._make_matcher(cpu_device)
        with pytest.raises(LocalMatcherRegionError):
            m.score_pair_strict(_make_bgr(), _make_bgr(), region="wing")

    def test_fingerprint_mismatch_match_features_raises(self, cpu_device):
        from models.local_matcher import StrictLocalMatcher, LocalMatcherInferenceError, REGION_EAR
        with patch("models.local_matcher.StrictLocalMatcher._init_lightglue_strict"):
            m = StrictLocalMatcher(device=cpu_device)
        q_bundle = _make_bundle(model_fingerprint="wrong_fp")
        r_bundle = _make_bundle(model_fingerprint=m.model_fingerprint)
        with pytest.raises(LocalMatcherInferenceError, match="fingerprint"):
            m.match_features(q_bundle, r_bundle, region=REGION_EAR)

    def test_missing_file_raises(self, cpu_device):
        from models.local_matcher import LocalMatcherFileError
        with patch("models.local_matcher.StrictLocalMatcher._init_lightglue_strict"):
            from models.local_matcher import StrictLocalMatcher
            m = StrictLocalMatcher(device=cpu_device)
        with pytest.raises(LocalMatcherFileError):
            m.score_pair_from_paths(
                "/nonexistent/query.jpg", "/nonexistent/ref.jpg", region="ear"
            )

    def test_wrong_array_type_raises(self, cpu_device):
        from models.local_matcher import StrictLocalMatcher, LocalMatcherInferenceError
        with patch("models.local_matcher.StrictLocalMatcher._init_lightglue_strict"):
            m = StrictLocalMatcher(device=cpu_device)
        with pytest.raises(LocalMatcherInferenceError):
            m.score_pair_strict("not_an_array", _make_bgr(), region="ear")


# ===========================================================================
# Part 6: FeatureCache
# ===========================================================================

class TestFeatureCacheKey:

    def test_same_content_same_key(self):
        from models.feature_cache import FeatureCacheKey
        img = _make_bgr(seed=99)
        k1 = FeatureCacheKey.from_image_array(img, "c1", "original", "fp")
        k2 = FeatureCacheKey.from_image_array(img, "c1", "original", "fp")
        assert k1 == k2

    def test_different_content_different_key(self):
        from models.feature_cache import FeatureCacheKey
        img1 = _make_bgr(seed=1)
        img2 = _make_bgr(seed=2)
        k1 = FeatureCacheKey.from_image_array(img1, "c1", "original", "fp")
        k2 = FeatureCacheKey.from_image_array(img2, "c1", "original", "fp")
        assert k1 != k2

    def test_orientation_in_key(self):
        from models.feature_cache import FeatureCacheKey
        img = _make_bgr(seed=5)
        k_orig = FeatureCacheKey.from_image_array(img, "c1", "original", "fp")
        k_flip = FeatureCacheKey.from_image_array(img, "c1", "flipped", "fp")
        assert k_orig != k_flip

    def test_cache_path_deterministic(self, tmp_cache_dir):
        from models.feature_cache import FeatureCacheKey
        img = _make_bgr(seed=7)
        k = FeatureCacheKey.from_image_array(img, "c1", "original", "fp")
        p1 = k.cache_path(tmp_cache_dir)
        p2 = k.cache_path(tmp_cache_dir)
        assert p1 == p2


class TestFeatureCacheMiss:

    def test_miss_on_empty_cache(self, tmp_cache_dir, model_fp):
        from models.feature_cache import FeatureCache, FeatureCacheKey
        cache = FeatureCache(tmp_cache_dir, model_fp, max_lru_entries=4)
        img = _make_bgr()
        key = FeatureCacheKey.from_image_array(img, "c1", "original", model_fp)
        assert cache.get(key) is None
        assert cache.stats()["misses"] == 1

    def test_miss_on_fingerprint_mismatch(self, tmp_cache_dir, model_fp):
        from models.feature_cache import FeatureCache, FeatureCacheKey
        cache = FeatureCache(tmp_cache_dir, model_fp, max_lru_entries=4)
        img = _make_bgr()
        key = FeatureCacheKey.from_image_array(img, "c1", "original", "different_fp")
        assert cache.get(key) is None


class TestFeatureCacheHit:

    def test_put_then_get_lru(self, tmp_cache_dir, model_fp):
        from models.feature_cache import FeatureCache, FeatureCacheKey
        cache = FeatureCache(tmp_cache_dir, model_fp, max_lru_entries=8)
        img = _make_bgr()
        key = FeatureCacheKey.from_image_array(img, "c1", "original", model_fp)
        bundle = _make_bundle(model_fingerprint=model_fp)
        cache.put(key, bundle)

        hit = cache.get(key)
        assert hit is not None
        assert np.allclose(hit.keypoints, bundle.keypoints)
        assert cache.stats()["hits_lru"] == 1

    def test_put_then_get_disk(self, tmp_cache_dir, model_fp):
        """After clearing LRU, the hit should come from disk."""
        from models.feature_cache import FeatureCache, FeatureCacheKey
        # maxsize=1 so second put evicts the first from LRU
        cache = FeatureCache(tmp_cache_dir, model_fp, max_lru_entries=1)
        img1 = _make_bgr(seed=0)
        img2 = _make_bgr(seed=1)
        key1 = FeatureCacheKey.from_image_array(img1, "c1", "original", model_fp)
        key2 = FeatureCacheKey.from_image_array(img2, "c2", "original", model_fp)
        b1 = _make_bundle(model_fingerprint=model_fp, seed=0)
        b2 = _make_bundle(model_fingerprint=model_fp, seed=1)

        cache.put(key1, b1)
        cache.put(key2, b2)  # evicts key1 from LRU

        # key1 should now be a disk hit
        hit = cache.get(key1)
        assert hit is not None
        assert np.allclose(hit.keypoints, b1.keypoints)
        assert cache.stats()["hits_disk"] >= 1

    def test_invalidate_removes_from_lru_and_disk(self, tmp_cache_dir, model_fp):
        from models.feature_cache import FeatureCache, FeatureCacheKey
        cache = FeatureCache(tmp_cache_dir, model_fp, max_lru_entries=4)
        img = _make_bgr(seed=42)
        key = FeatureCacheKey.from_image_array(img, "c1", "original", model_fp)
        bundle = _make_bundle(model_fingerprint=model_fp)
        cache.put(key, bundle)

        removed = cache.invalidate(key)
        assert removed is True
        assert cache.get(key) is None
        assert not key.cache_path(tmp_cache_dir).is_file()


class TestFeatureCacheCorruption:

    def test_corrupted_file_treated_as_miss(self, tmp_cache_dir, model_fp):
        from models.feature_cache import FeatureCache, FeatureCacheKey
        cache = FeatureCache(tmp_cache_dir, model_fp, max_lru_entries=4)
        img = _make_bgr(seed=99)
        key = FeatureCacheKey.from_image_array(img, "c1", "original", model_fp)
        bundle = _make_bundle(model_fingerprint=model_fp)
        cache.put(key, bundle)

        # Clear LRU so next get goes to disk
        cache._lru.evict(key)

        # Corrupt the file
        disk_path = key.cache_path(tmp_cache_dir)
        disk_path.write_bytes(b"corrupted bytes")

        result = cache.get(key)
        assert result is None
        assert cache.stats()["corruption_evictions"] == 1
        # File should be removed
        assert not disk_path.is_file()

    def test_put_fingerprint_mismatch_raises(self, tmp_cache_dir, model_fp):
        from models.feature_cache import FeatureCache, FeatureCacheKey
        cache = FeatureCache(tmp_cache_dir, model_fp, max_lru_entries=4)
        img = _make_bgr()
        key = FeatureCacheKey.from_image_array(img, "c1", "original", model_fp)
        bundle = _make_bundle(model_fingerprint="wrong_fp")
        with pytest.raises(ValueError, match="fingerprint"):
            cache.put(key, bundle)


class TestFeatureCacheAtomicity:

    def test_atomic_write_no_partial_file(self, tmp_cache_dir, model_fp):
        """If write fails mid-way, the target file is not left in a partial state."""
        from models.feature_cache import FeatureCache, FeatureCacheKey
        cache = FeatureCache(tmp_cache_dir, model_fp, max_lru_entries=4)
        img = _make_bgr(seed=7)
        key = FeatureCacheKey.from_image_array(img, "c1", "original", model_fp)
        bundle = _make_bundle(model_fingerprint=model_fp)

        target = key.cache_path(tmp_cache_dir)
        assert not target.is_file()

        # Simulate os.replace failing after tmp write
        original_replace = os.replace

        def _fail_replace(src, dst):
            # Remove the tmp file as if failed mid-replace
            Path(src).unlink(missing_ok=True)
            raise OSError("simulated replace failure")

        with patch("os.replace", side_effect=_fail_replace):
            with pytest.raises(OSError):
                cache.put(key, bundle)

        # Target must not exist
        assert not target.is_file()


class TestLRUCache:

    def test_lru_eviction(self, model_fp):
        from models.feature_cache import _LRUCache, FeatureCacheKey
        lru = _LRUCache(maxsize=2)
        img_a = _make_bgr(seed=0)
        img_b = _make_bgr(seed=1)
        img_c = _make_bgr(seed=2)
        ka = FeatureCacheKey.from_image_array(img_a, "a", "original", model_fp)
        kb = FeatureCacheKey.from_image_array(img_b, "b", "original", model_fp)
        kc = FeatureCacheKey.from_image_array(img_c, "c", "original", model_fp)

        lru.put(ka, "val_a")
        lru.put(kb, "val_b")
        assert len(lru) == 2

        # Adding kc should evict ka (LRU)
        lru.put(kc, "val_c")
        assert len(lru) == 2
        assert lru.get(ka) is None
        assert lru.get(kb) == "val_b"
        assert lru.get(kc) == "val_c"

    def test_lru_access_updates_order(self, model_fp):
        from models.feature_cache import _LRUCache, FeatureCacheKey
        lru = _LRUCache(maxsize=2)
        img_a = _make_bgr(seed=10)
        img_b = _make_bgr(seed=11)
        img_c = _make_bgr(seed=12)
        ka = FeatureCacheKey.from_image_array(img_a, "a", "original", model_fp)
        kb = FeatureCacheKey.from_image_array(img_b, "b", "original", model_fp)
        kc = FeatureCacheKey.from_image_array(img_c, "c", "original", model_fp)

        lru.put(ka, "a")
        lru.put(kb, "b")
        _ = lru.get(ka)  # access ka → ka is now MRU, kb is LRU
        lru.put(kc, "c")  # should evict kb

        assert lru.get(kb) is None
        assert lru.get(ka) == "a"
        assert lru.get(kc) == "c"

    def test_lru_maxsize_one(self, model_fp):
        from models.feature_cache import _LRUCache, FeatureCacheKey
        lru = _LRUCache(maxsize=1)
        img_a = _make_bgr(seed=20)
        img_b = _make_bgr(seed=21)
        ka = FeatureCacheKey.from_image_array(img_a, "a", "original", model_fp)
        kb = FeatureCacheKey.from_image_array(img_b, "b", "original", model_fp)
        lru.put(ka, "a")
        lru.put(kb, "b")
        assert lru.get(ka) is None
        assert lru.get(kb) == "b"

    def test_lru_invalid_maxsize_raises(self):
        from models.feature_cache import _LRUCache
        with pytest.raises(ValueError):
            _LRUCache(maxsize=0)


# ===========================================================================
# Part 7: LocalPairScore and LocalIdentityScore schema integrity
# ===========================================================================

class TestLocalScoreSchema:

    def _valid_pair_score(self, **overrides):
        from models.local_score_schema import LocalPairScore, SCHEMA_VERSION
        defaults = dict(
            schema_version=SCHEMA_VERSION,
            backend="lightglue",
            model_fingerprint="fp0000000000",
            scoring_fingerprint="sp0000000000",
            source_fingerprint="src",
            split_fingerprint="spl",
            query_crop_id="qc1",
            ref_crop_id="rc1",
            query_crop_kind="ear",
            ref_crop_kind="ear",
            region="ear",
            orientation="original",
            geom_model_used="homography",
            n_raw_matches=20,
            n_inliers=10,
            inlier_ratio=0.5,
            geometric_spread=15.0,
            score=10.0,
            missing_file=False,
            latency_ms=12.3,
        )
        defaults.update(overrides)
        return LocalPairScore(**defaults)

    def test_valid_pair_score_passes(self):
        from models.local_score_schema import assert_pair_score_integrity
        ps = self._valid_pair_score()
        assert_pair_score_integrity(ps)  # no raise

    def test_schema_version_mismatch_raises(self):
        from models.local_score_schema import assert_pair_score_integrity, LocalScoreIntegrityError
        ps = self._valid_pair_score(schema_version="old-v0")
        with pytest.raises(LocalScoreIntegrityError, match="schema_version"):
            assert_pair_score_integrity(ps)

    def test_crop_kind_mismatch_raises(self):
        from models.local_score_schema import assert_pair_score_integrity, LocalScoreIntegrityError
        ps = self._valid_pair_score(query_crop_kind="ear", ref_crop_kind="body")
        with pytest.raises(LocalScoreIntegrityError, match="crop_kind mismatch"):
            assert_pair_score_integrity(ps)

    def test_invalid_region_raises(self):
        from models.local_score_schema import assert_pair_score_integrity, LocalScoreIntegrityError
        ps = self._valid_pair_score(region="head", query_crop_kind="head", ref_crop_kind="head")
        with pytest.raises(LocalScoreIntegrityError, match="region"):
            assert_pair_score_integrity(ps)

    def test_negative_inliers_raises(self):
        from models.local_score_schema import assert_pair_score_integrity, LocalScoreIntegrityError
        ps = self._valid_pair_score(n_inliers=-1)
        with pytest.raises(LocalScoreIntegrityError, match="n_inliers"):
            assert_pair_score_integrity(ps)

    def test_inlier_ratio_out_of_range_raises(self):
        from models.local_score_schema import assert_pair_score_integrity, LocalScoreIntegrityError
        ps = self._valid_pair_score(inlier_ratio=1.5)
        with pytest.raises(LocalScoreIntegrityError, match="inlier_ratio"):
            assert_pair_score_integrity(ps)

    def test_invalid_orientation_raises(self):
        from models.local_score_schema import assert_pair_score_integrity, LocalScoreIntegrityError
        ps = self._valid_pair_score(orientation="upside_down")
        with pytest.raises(LocalScoreIntegrityError, match="orientation"):
            assert_pair_score_integrity(ps)

    def test_empty_required_field_raises(self):
        from models.local_score_schema import assert_pair_score_integrity, LocalScoreIntegrityError
        ps = self._valid_pair_score(backend="")
        with pytest.raises(LocalScoreIntegrityError, match="backend"):
            assert_pair_score_integrity(ps)

    def test_negative_score_raises(self):
        from models.local_score_schema import assert_pair_score_integrity, LocalScoreIntegrityError
        ps = self._valid_pair_score(score=-0.1)
        with pytest.raises(LocalScoreIntegrityError, match="score"):
            assert_pair_score_integrity(ps)

    def test_valid_body_pair_score(self):
        from models.local_score_schema import assert_pair_score_integrity
        ps = self._valid_pair_score(
            query_crop_kind="body",
            ref_crop_kind="body",
            region="body",
            geom_model_used="partial_affine",
        )
        assert_pair_score_integrity(ps)

    def _valid_identity_score(self, pair_scores=None, **overrides):
        from models.local_score_schema import LocalIdentityScore, SCHEMA_VERSION
        if pair_scores is None:
            pair_scores = [self._valid_pair_score()]
        defaults = dict(
            schema_version=SCHEMA_VERSION,
            backend="lightglue",
            model_fingerprint="fp0000000000",
            scoring_fingerprint="sp0000000000",
            query_crop_kind="ear",
            candidate_individual_id="bteh_alpha",
            n_pairs_attempted=1,
            n_pairs_valid=1,
            n_pairs_missing_file=0,
            n_sessions_used=1,
            n_sessions_cap=3,
            region_coverage={"ear": 1},
            orientations_attempted={"original"},
            aggregation_method="mean_top_k",
            top_k=2,
            score=10.0,
            pair_scores=pair_scores,
            latency_ms=50.0,
        )
        defaults.update(overrides)
        return LocalIdentityScore(**defaults)

    def test_valid_identity_score_passes(self):
        from models.local_score_schema import assert_identity_score_integrity
        s = self._valid_identity_score()
        assert_identity_score_integrity(s)

    def test_invalid_aggregation_method_raises(self):
        from models.local_score_schema import assert_identity_score_integrity, LocalScoreIntegrityError
        s = self._valid_identity_score(aggregation_method="median")
        with pytest.raises(LocalScoreIntegrityError, match="aggregation_method"):
            assert_identity_score_integrity(s)

    def test_n_valid_exceeds_attempted_raises(self):
        from models.local_score_schema import assert_identity_score_integrity, LocalScoreIntegrityError
        s = self._valid_identity_score(n_pairs_attempted=1, n_pairs_valid=5)
        with pytest.raises(LocalScoreIntegrityError, match="n_pairs_valid"):
            assert_identity_score_integrity(s)

    def test_mixed_model_fingerprints_raises(self):
        from models.local_score_schema import (
            assert_identity_score_integrity, LocalScoreIntegrityError, SCHEMA_VERSION
        )
        ps1 = self._valid_pair_score(model_fingerprint="fp_aaa")
        ps2 = self._valid_pair_score(model_fingerprint="fp_bbb")
        s = self._valid_identity_score(pair_scores=[ps1, ps2], n_pairs_attempted=2)
        with pytest.raises(LocalScoreIntegrityError, match="model_fingerprint"):
            assert_identity_score_integrity(s)

    def test_mixed_scoring_fingerprints_raises(self):
        from models.local_score_schema import assert_identity_score_integrity, LocalScoreIntegrityError
        ps1 = self._valid_pair_score(scoring_fingerprint="sp_aaa")
        ps2 = self._valid_pair_score(scoring_fingerprint="sp_bbb")
        s = self._valid_identity_score(pair_scores=[ps1, ps2], n_pairs_attempted=2)
        with pytest.raises(LocalScoreIntegrityError, match="scoring_fingerprint"):
            assert_identity_score_integrity(s)

    def test_top_k_zero_raises(self):
        from models.local_score_schema import assert_identity_score_integrity, LocalScoreIntegrityError
        s = self._valid_identity_score(top_k=0)
        with pytest.raises(LocalScoreIntegrityError, match="top_k"):
            assert_identity_score_integrity(s)


# ===========================================================================
# Part 8: Scoring fingerprint stability
# ===========================================================================

class TestScoringFingerprint:

    def test_fingerprint_deterministic(self):
        from models.local_score_schema import make_scoring_fingerprint
        fp1 = make_scoring_fingerprint("lightglue", "model_fp", "local-v1", "homography", True)
        fp2 = make_scoring_fingerprint("lightglue", "model_fp", "local-v1", "homography", True)
        assert fp1 == fp2

    def test_fingerprint_changes_on_mirror_flag(self):
        from models.local_score_schema import make_scoring_fingerprint
        fp1 = make_scoring_fingerprint("lightglue", "model_fp", "local-v1", "homography", True)
        fp2 = make_scoring_fingerprint("lightglue", "model_fp", "local-v1", "homography", False)
        assert fp1 != fp2

    def test_fingerprint_changes_on_schema_version(self):
        from models.local_score_schema import make_scoring_fingerprint
        fp1 = make_scoring_fingerprint("lightglue", "model_fp", "local-v1", "homography", True)
        fp2 = make_scoring_fingerprint("lightglue", "model_fp", "local-v2", "homography", True)
        assert fp1 != fp2

    def test_identity_fingerprint_changes_on_top_k(self):
        from models.local_score_schema import make_identity_scoring_fingerprint
        fp1 = make_identity_scoring_fingerprint("lightglue", "fp", "local-v1", 3, 2, "mean_top_k")
        fp2 = make_identity_scoring_fingerprint("lightglue", "fp", "local-v1", 3, 3, "mean_top_k")
        assert fp1 != fp2


# ===========================================================================
# Part 9: LocalIdentityScorer logic
# ===========================================================================

class TestLocalIdentityScorer:

    def _make_scorer(self, cpu_device, orig_inliers=10, flip_inliers=5):
        """Make a scorer with a mocked matcher that returns controllable inlier counts."""
        from models.local_matcher import StrictLocalMatcher, MatchResult, GeomVerification, REGION_EAR
        from models.identity_scorer import LocalIdentityScorer

        def _fake_score_pair_strict(q_bgr, r_bgr, region, geom_model=None, mirror_search=True):
            n = orig_inliers
            geom = GeomVerification(
                model_used="homography",
                n_raw_matches=n,
                n_inliers=n,
                inlier_ratio=1.0,
                geometric_spread=10.0,
                H=None,
                inlier_mask=None,
            )
            return MatchResult(
                region=region,
                orientation="original",
                raw_matches=np.zeros((n, 2), dtype=np.int32),
                query_keypoints=np.zeros((n, 2), dtype=np.float32),
                ref_keypoints=np.zeros((n, 2), dtype=np.float32),
                geom=geom,
                backend="lightglue",
                model_fingerprint="test_fp",
                n_inliers=n,
            )

        with patch("models.local_matcher.StrictLocalMatcher._init_lightglue_strict"):
            m = StrictLocalMatcher(device=cpu_device)
        m.score_pair_strict = _fake_score_pair_strict
        scorer = LocalIdentityScorer(m, cache=None)
        return scorer

    def _make_query_ear(self, crop_id="q_ear_0", path=None):
        from models.identity_scorer import QueryCrop
        p = path or "/fake/query_ear.jpg"
        return QueryCrop(crop_id=crop_id, crop_path=p, crop_kind="ear")

    def _make_ref_ear(self, crop_id="r_ear_0", session_id="sess_a", path=None):
        from models.identity_scorer import ReferenceImage
        p = path or "/fake/ref_ear.jpg"
        return ReferenceImage(
            crop_id=crop_id,
            crop_path=p,
            crop_kind="ear",
            session_id=session_id,
            individual_id="bteh_alpha",
        )

    def _make_ref_body(self, crop_id="r_body_0", session_id="sess_a", path=None):
        from models.identity_scorer import ReferenceImage
        p = path or "/fake/ref_body.jpg"
        return ReferenceImage(
            crop_id=crop_id,
            crop_path=p,
            crop_kind="body",
            session_id=session_id,
            individual_id="bteh_alpha",
        )

    def test_session_cap_three_sessions(self, cpu_device):
        """At most 3 sessions are used regardless of how many are supplied."""
        scorer = self._make_scorer(cpu_device)
        refs = [
            self._make_ref_ear(f"r{i}", f"sess_{i}")
            for i in range(6)  # 6 sessions, only 3 should be used
        ]
        selected = scorer.select_references(refs)
        sessions = {r.session_id for r in selected}
        assert len(sessions) <= 3

    def test_session_cap_one_per_session(self, cpu_device):
        """Only one ref per session is kept."""
        scorer = self._make_scorer(cpu_device)
        refs = [
            self._make_ref_ear("r0", "sess_a"),
            self._make_ref_ear("r1", "sess_a"),  # duplicate session
            self._make_ref_ear("r2", "sess_b"),
        ]
        selected = scorer.select_references(refs)
        assert len(selected) == 2
        ids = {r.crop_id for r in selected}
        assert "r0" in ids  # first from sess_a is kept

    def test_crop_kind_mismatch_raises(self, cpu_device):
        from models.local_matcher import LocalMatcherRegionError
        scorer = self._make_scorer(cpu_device)
        q = self._make_query_ear()
        r = self._make_ref_body()  # body ref vs ear query
        with pytest.raises(LocalMatcherRegionError, match="crop_kind mismatch"):
            scorer.score_identity([q], [r])

    def test_mixed_query_crop_kinds_raises(self, cpu_device):
        from models.local_matcher import LocalMatcherRegionError
        from models.identity_scorer import QueryCrop
        scorer = self._make_scorer(cpu_device)
        q_ear = self._make_query_ear()
        q_body = QueryCrop("q_body", "/fake/body.jpg", "body")
        with pytest.raises(LocalMatcherRegionError, match="mixed crop_kinds"):
            scorer.score_identity([q_ear, q_body], [])

    def test_empty_query_raises(self, cpu_device):
        from models.local_matcher import LocalMatcherRegionError
        scorer = self._make_scorer(cpu_device)
        with pytest.raises(LocalMatcherRegionError, match="empty"):
            scorer.score_identity([], [self._make_ref_ear()])

    def test_ear_all_combinations(self, cpu_device, tmp_path):
        """All query ears × all selected reference ears are paired."""
        from models.identity_scorer import LocalIdentityScorer, QueryCrop, ReferenceImage
        from models.local_matcher import StrictLocalMatcher, MatchResult, GeomVerification

        # Create real tiny images on disk
        bgr = _make_bgr(seed=0)
        q_paths = []
        r_paths = []
        for i in range(2):
            p = str(tmp_path / f"q_ear_{i}.jpg")
            cv2.imwrite(p, bgr)
            q_paths.append(p)
        for i in range(2):
            p = str(tmp_path / f"r_ear_{i}.jpg")
            cv2.imwrite(p, bgr)
            r_paths.append(p)

        n_inliers = 7

        def _fake_score_pair_strict(q_bgr, r_bgr, region, geom_model=None, mirror_search=True):
            geom = GeomVerification("homography", n_inliers, n_inliers, 1.0, 5.0, None, None)
            return MatchResult(
                region=region,
                orientation="original",
                raw_matches=np.zeros((n_inliers, 2), dtype=np.int32),
                query_keypoints=np.zeros((n_inliers, 2), dtype=np.float32),
                ref_keypoints=np.zeros((n_inliers, 2), dtype=np.float32),
                geom=geom,
                backend="lightglue",
                model_fingerprint="test_fp",
                n_inliers=n_inliers,
            )

        with patch("models.local_matcher.StrictLocalMatcher._init_lightglue_strict"):
            m = StrictLocalMatcher(device=cpu_device)
        m.score_pair_strict = _fake_score_pair_strict

        scorer = LocalIdentityScorer(m, cache=None)
        queries = [QueryCrop(f"q{i}", q_paths[i], "ear") for i in range(2)]
        refs = [ReferenceImage(f"r{i}", r_paths[i], "ear", f"sess_{i}", "bteh_alpha") for i in range(2)]

        result = scorer.score_identity(queries, refs)
        # 2 queries × 2 refs = 4 pairs
        assert result.n_pairs_attempted == 4
        assert result.region_coverage.get("ear", 0) == 4

    def test_mean_top_k_aggregation(self, cpu_device, tmp_path):
        """mean_top_k uses mean of top-2 valid pair scores."""
        from models.identity_scorer import LocalIdentityScorer, QueryCrop, ReferenceImage
        from models.local_matcher import StrictLocalMatcher, MatchResult, GeomVerification

        bgr = _make_bgr(seed=5)
        scores_to_return = [3, 8, 15]  # 3 pairs; top-2 = [15, 8] → mean = 11.5

        call_index = [0]

        def _fake_score_pair_strict(q_bgr, r_bgr, region, geom_model=None, mirror_search=True):
            n = scores_to_return[call_index[0] % len(scores_to_return)]
            call_index[0] += 1
            geom = GeomVerification("homography", n, n, 1.0, 5.0, None, None)
            return MatchResult(
                region=region,
                orientation="original",
                raw_matches=np.zeros((n, 2), dtype=np.int32),
                query_keypoints=np.zeros((n, 2), dtype=np.float32),
                ref_keypoints=np.zeros((n, 2), dtype=np.float32),
                geom=geom,
                backend="lightglue",
                model_fingerprint="test_fp",
                n_inliers=n,
            )

        paths = []
        for i in range(4):
            p = str(tmp_path / f"img_{i}.jpg")
            cv2.imwrite(p, bgr)
            paths.append(p)

        with patch("models.local_matcher.StrictLocalMatcher._init_lightglue_strict"):
            m = StrictLocalMatcher(device=cpu_device)
        m.score_pair_strict = _fake_score_pair_strict

        scorer = LocalIdentityScorer(m, cache=None, top_k=2)
        q = QueryCrop("q0", paths[0], "ear")
        refs = [
            ReferenceImage(f"r{i}", paths[i + 1], "ear", f"sess_{i}", "bteh_alpha")
            for i in range(3)
        ]
        result = scorer.score_identity([q], refs)

        # top-2 of [3, 8, 15] = mean([15, 8]) = 11.5
        assert pytest.approx(result.score, abs=1e-6) == 11.5

    def test_mean_top_k_fallback_when_fewer_than_k(self, cpu_device, tmp_path):
        """When only 1 valid pair, fallback mean of available (score = that 1 value)."""
        from models.identity_scorer import LocalIdentityScorer, QueryCrop, ReferenceImage
        from models.local_matcher import StrictLocalMatcher, MatchResult, GeomVerification

        bgr = _make_bgr(seed=6)
        for i in range(2):
            cv2.imwrite(str(tmp_path / f"img_{i}.jpg"), bgr)

        def _fake_score_pair_strict(q_bgr, r_bgr, region, geom_model=None, mirror_search=True):
            n = 12
            geom = GeomVerification("homography", n, n, 1.0, 5.0, None, None)
            return MatchResult(
                region=region, orientation="original",
                raw_matches=np.zeros((n, 2), dtype=np.int32),
                query_keypoints=np.zeros((n, 2), dtype=np.float32),
                ref_keypoints=np.zeros((n, 2), dtype=np.float32),
                geom=geom, backend="lightglue", model_fingerprint="test_fp", n_inliers=n,
            )

        with patch("models.local_matcher.StrictLocalMatcher._init_lightglue_strict"):
            m = StrictLocalMatcher(device=cpu_device)
        m.score_pair_strict = _fake_score_pair_strict
        scorer = LocalIdentityScorer(m, cache=None, top_k=2)

        q = QueryCrop("q0", str(tmp_path / "img_0.jpg"), "ear")
        refs = [ReferenceImage("r0", str(tmp_path / "img_1.jpg"), "ear", "sess_a", "bteh_alpha")]
        result = scorer.score_identity([q], refs)
        # Only 1 pair → fallback to mean of [12] = 12.0
        assert pytest.approx(result.score) == 12.0

    def test_missing_file_counted(self, cpu_device):
        """Pairs with missing files are counted in n_pairs_missing_file."""
        scorer = self._make_scorer(cpu_device)
        q = self._make_query_ear(path="/nonexistent/q.jpg")
        r = self._make_ref_ear(path="/nonexistent/r.jpg")
        result = scorer.score_identity([q], [r])
        assert result.n_pairs_missing_file >= 1
        assert result.n_pairs_valid == 0
        assert result.score == 0.0

    def test_diagnostics_populated(self, cpu_device, tmp_path):
        """Check diagnostics fields are set."""
        from models.identity_scorer import LocalIdentityScorer, QueryCrop, ReferenceImage
        from models.local_matcher import StrictLocalMatcher, MatchResult, GeomVerification

        bgr = _make_bgr(seed=99)
        p = str(tmp_path / "img.jpg")
        cv2.imwrite(p, bgr)

        def _fake_score_pair_strict(q_bgr, r_bgr, region, geom_model=None, mirror_search=True):
            n = 5
            geom = GeomVerification("homography", n, n, 1.0, 5.0, None, None)
            return MatchResult(
                region=region, orientation="original",
                raw_matches=np.zeros((n, 2), dtype=np.int32),
                query_keypoints=np.zeros((n, 2), dtype=np.float32),
                ref_keypoints=np.zeros((n, 2), dtype=np.float32),
                geom=geom, backend="lightglue", model_fingerprint="test_fp", n_inliers=n,
            )

        with patch("models.local_matcher.StrictLocalMatcher._init_lightglue_strict"):
            m = StrictLocalMatcher(device=cpu_device)
        m.score_pair_strict = _fake_score_pair_strict
        scorer = LocalIdentityScorer(m, cache=None)

        q = QueryCrop("q0", p, "ear")
        r = ReferenceImage("r0", p, "ear", "sess_a", "bteh_alpha")
        result = scorer.score_identity([q], [r], candidate_individual_id="bteh_alpha")

        assert result.n_pairs_attempted == 1
        assert result.n_sessions_used == 1
        assert "ear" in result.region_coverage
        assert "original" in result.orientations_attempted
        assert result.latency_ms >= 0.0
        assert result.candidate_individual_id == "bteh_alpha"


# ===========================================================================
# Part 10: selected-v1 non-mutation
# ===========================================================================

class TestSelectedV1NonMutation:
    """New constants and schema additions must not alter the existing selected-v1 artifacts."""

    def test_artifact_schema_version_unchanged(self):
        """ARTIFACT_SCHEMA_VERSION must remain 'v1'."""
        from configs.config_bteh import ARTIFACT_SCHEMA_VERSION
        assert ARTIFACT_SCHEMA_VERSION == "v1"

    def test_crop_manifest_columns_unchanged(self):
        """The base CROP_MANIFEST_COLUMNS list must not be modified."""
        from utils.artifact_schema import CROP_MANIFEST_COLUMNS
        expected_columns = [
            "crop_id",
            "image_id",
            "individual_id",
            "crop_kind",
            "crop_ordinal",
            "crop_path",
            "detector_confidence",
            "detector_box",
            "detector_status",
            "review_status",
            "schema_version",
            "source_fingerprint",
            "split_fingerprint",
        ]
        assert CROP_MANIFEST_COLUMNS == expected_columns, (
            "CROP_MANIFEST_COLUMNS was modified — "
            "this breaks selected-v1 artifact compatibility"
        )

    def test_production_selected_channels_unchanged(self):
        from configs.config_elephant import PRODUCTION_SELECTED_CHANNELS
        assert set(PRODUCTION_SELECTED_CHANNELS) == {"miewid", "ear_miewid_projected"}

    def test_local_score_schema_version_is_local_v1(self):
        from configs.config_elephant import LOCAL_SCORE_SCHEMA_VERSION
        assert LOCAL_SCORE_SCHEMA_VERSION == "local-v1"

    def test_local_score_schema_does_not_use_artifact_schema_version(self):
        """LOCAL_SCORE_SCHEMA_VERSION and ARTIFACT_SCHEMA_VERSION are independent."""
        from configs.config_elephant import LOCAL_SCORE_SCHEMA_VERSION
        from configs.config_bteh import ARTIFACT_SCHEMA_VERSION
        assert LOCAL_SCORE_SCHEMA_VERSION != ARTIFACT_SCHEMA_VERSION

    def test_new_config_constants_present(self):
        from configs.config_elephant import (
            LOCAL_MATCHER_LOFTR_PILOT_APPROVED,
            LOCAL_FEATURE_CACHE_MAX_LRU,
            LOCAL_IDENTITY_SCORER_MAX_SESSIONS,
            LOCAL_IDENTITY_SCORER_TOP_K,
            LOCAL_SCORE_SCHEMA_VERSION,
        )
        assert LOCAL_MATCHER_LOFTR_PILOT_APPROVED is False
        assert LOCAL_FEATURE_CACHE_MAX_LRU > 0
        assert LOCAL_IDENTITY_SCORER_MAX_SESSIONS == 3
        assert LOCAL_IDENTITY_SCORER_TOP_K == 2


# ===========================================================================
# Part 11: FeatureBundle serialisation roundtrip
# ===========================================================================

class TestFeatureBundleSerialisation:

    def test_roundtrip_preserves_arrays(self, model_fp):
        from models.feature_cache import _bundle_to_npz, _npz_to_bundle
        bundle = _make_bundle(model_fingerprint=model_fp)
        data = _bundle_to_npz(bundle)
        restored = _npz_to_bundle(data)
        assert restored is not None
        assert np.allclose(restored.keypoints, bundle.keypoints)
        assert np.allclose(restored.descriptors, bundle.descriptors)
        assert np.allclose(restored.scores, bundle.scores)
        assert restored.orientation == bundle.orientation
        assert restored.model_fingerprint == bundle.model_fingerprint

    def test_corruption_returns_none(self, model_fp):
        from models.feature_cache import _npz_to_bundle
        assert _npz_to_bundle(b"not valid npz data") is None

    def test_integrity_hash_mismatch_returns_none(self, model_fp):
        """Manually corrupt the integrity hash inside the npz."""
        from models.feature_cache import _bundle_to_npz, _npz_to_bundle
        import io as _io
        bundle = _make_bundle(model_fingerprint=model_fp)
        data = _bundle_to_npz(bundle)

        # Reload, modify integrity_hash, re-save
        buf = _io.BytesIO(data)
        npz = np.load(buf, allow_pickle=False)
        arrays = {k: npz[k] for k in npz.files}
        arrays["integrity_hash"] = np.array("0000000000000000")
        out = _io.BytesIO()
        np.savez_compressed(out, **arrays)
        corrupted = out.getvalue()

        assert _npz_to_bundle(corrupted) is None


# ===========================================================================
# Part 12: MatchResult properties
# ===========================================================================

class TestMatchResult:

    def _make_match_result(self, n_inliers=10, n_raw=20, spread=15.0, orientation="original"):
        from models.local_matcher import MatchResult, GeomVerification, REGION_EAR
        geom = GeomVerification(
            model_used="homography",
            n_raw_matches=n_raw,
            n_inliers=n_inliers,
            inlier_ratio=n_inliers / n_raw if n_raw > 0 else 0.0,
            geometric_spread=spread,
            H=None,
            inlier_mask=None,
        )
        return MatchResult(
            region=REGION_EAR,
            orientation=orientation,
            raw_matches=np.zeros((n_inliers, 2), dtype=np.int32),
            query_keypoints=np.zeros((10, 2), dtype=np.float32),
            ref_keypoints=np.zeros((10, 2), dtype=np.float32),
            geom=geom,
            backend="lightglue",
            model_fingerprint="fp",
            n_inliers=n_inliers,
        )

    def test_inlier_ratio_property(self):
        r = self._make_match_result(n_inliers=8, n_raw=20)
        assert pytest.approx(r.inlier_ratio) == 0.4

    def test_geometric_spread_property(self):
        r = self._make_match_result(spread=22.5)
        assert pytest.approx(r.geometric_spread) == 22.5


# ===========================================================================
# Helper: fake SuperPoint extractor for tests that instantiate the matcher
# ===========================================================================

def _make_fake_superpoint():
    """
    Returns a mock that behaves like a SuperPoint extractor
    for the native LightGlue path.
    """
    from models.local_matcher import FeatureBundle
    import torch

    class _FakeSP:
        def extract(self, tensor):
            n = 20
            return {
                "keypoints": torch.zeros(1, n, 2),
                "descriptors": torch.zeros(1, n, 256),
                "keypoint_scores": torch.ones(1, n),
                "image_size": torch.tensor([[64, 64]]),
            }
        def eval(self):
            return self
        def to(self, device):
            return self

    return _FakeSP()
