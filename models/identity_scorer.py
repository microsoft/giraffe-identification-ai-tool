# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Canonical local identity scorer.

Used unchanged by OOF evaluation and fixed inference.  Any code that computes
a local identity score MUST go through this module — never duplicate scoring
logic outside it.

Design contracts
----------------
* Both query and reference crops must have the same crop_kind; raises on mismatch.
* Reference selection: up to LOCAL_IDENTITY_SCORER_MAX_SESSIONS distinct
  sessions, one globally strongest reference image per session (supplied
  deterministically by the caller — this module never selects references).
* Query all available ear crops against each selected reference ear crop;
  body is one-to-one per selected image.
* Orientation search (mirror) is identical to StrictLocalMatcher.score_pair_strict
  with mirror_search=True — done the same way here as everywhere else.
* Primary aggregation: mean of top-k valid pair scores (default k=2).
  Fallback to mean of all valid pairs when fewer than k are available.
  max_available is exposed only as an explicit exploratory option and is
  never used by default.
* No truth forcing and no hidden candidate expansion.
* All diagnostics (comparison count, valid count, missing files, region
  coverage, attempted orientations, latency) are returned on LocalIdentityScore.

Not implemented here
--------------------
* Real local signal pilot, OOF pair selection, calibration, fusion,
  probe evaluation.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import numpy as np
import cv2

from configs.config_elephant import (
    LOCAL_IDENTITY_SCORER_MAX_SESSIONS,
    LOCAL_IDENTITY_SCORER_TOP_K,
)
from models.local_matcher import (
    StrictLocalMatcher,
    LocalMatcherFileError,
    LocalMatcherRegionError,
    REGION_EAR,
    REGION_BODY,
    STRICT_SUPPORTED_REGIONS,
)
from models.local_score_schema import (
    LocalPairScore,
    LocalIdentityScore,
    SCHEMA_VERSION,
    assert_pair_score_integrity,
    assert_identity_score_integrity,
    make_scoring_fingerprint,
    make_identity_scoring_fingerprint,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reference session record
# ---------------------------------------------------------------------------

class ReferenceImage:
    """
    Single reference image entry.

    Parameters
    ----------
    crop_id:
        Unique crop identifier (from crop manifest).
    crop_path:
        Absolute or resolvable path to the crop image file.
    crop_kind:
        'body' | 'ear'.
    session_id:
        Session identifier (used for the 3-session cap).
    individual_id:
        Ground-truth individual identity label.
    """

    __slots__ = ("crop_id", "crop_path", "crop_kind", "session_id", "individual_id")

    def __init__(
        self,
        crop_id: str,
        crop_path: str,
        crop_kind: str,
        session_id: str,
        individual_id: str,
    ):
        if crop_kind not in STRICT_SUPPORTED_REGIONS:
            raise LocalMatcherRegionError(
                f"ReferenceImage: crop_kind {crop_kind!r} not in "
                f"{sorted(STRICT_SUPPORTED_REGIONS)}"
            )
        self.crop_id = crop_id
        self.crop_path = crop_path
        self.crop_kind = crop_kind
        self.session_id = session_id
        self.individual_id = individual_id


class QueryCrop:
    """
    Single query crop entry.

    Parameters
    ----------
    crop_id:
        Unique crop identifier.
    crop_path:
        Absolute or resolvable path to the crop image file.
    crop_kind:
        'body' | 'ear'.
    """

    __slots__ = ("crop_id", "crop_path", "crop_kind")

    def __init__(self, crop_id: str, crop_path: str, crop_kind: str):
        if crop_kind not in STRICT_SUPPORTED_REGIONS:
            raise LocalMatcherRegionError(
                f"QueryCrop: crop_kind {crop_kind!r} not in "
                f"{sorted(STRICT_SUPPORTED_REGIONS)}"
            )
        self.crop_id = crop_id
        self.crop_path = crop_path
        self.crop_kind = crop_kind


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate(scores: list[float], method: str, top_k: int) -> float:
    """
    Aggregate a list of valid pair scores.

    method: 'mean_top_k' | 'mean_all' | 'max_available'
    Returns 0.0 if *scores* is empty.
    """
    if not scores:
        return 0.0
    if method == "mean_top_k":
        top = sorted(scores, reverse=True)[:top_k]
        return float(np.mean(top))
    if method == "mean_all":
        return float(np.mean(scores))
    if method == "max_available":
        return float(max(scores))
    raise ValueError(f"Unknown aggregation method: {method!r}")


# ---------------------------------------------------------------------------
# Canonical identity scorer
# ---------------------------------------------------------------------------

class LocalIdentityScorer:
    """
    Canonical local identity scorer.

    Parameters
    ----------
    matcher:
        A StrictLocalMatcher instance.
    cache:
        Optional FeatureCache.  If None, features are not cached to disk.
    max_sessions:
        Maximum number of distinct reference sessions to use (default 3).
    top_k:
        Number of top valid pair scores to average for 'mean_top_k' aggregation.
    geom_model:
        Override for RANSAC model; if None, region policy is applied.
    """

    def __init__(
        self,
        matcher: StrictLocalMatcher,
        cache=None,
        max_sessions: int = LOCAL_IDENTITY_SCORER_MAX_SESSIONS,
        top_k: int = LOCAL_IDENTITY_SCORER_TOP_K,
        geom_model: Optional[str] = None,
    ):
        self.matcher = matcher
        self.cache = cache
        self.max_sessions = max_sessions
        self.top_k = top_k
        self.geom_model = geom_model
        self._identity_result_cache: dict[tuple, LocalIdentityScore] = {}

        self._scoring_fingerprint = make_scoring_fingerprint(
            backend=matcher.backend,
            model_fingerprint=matcher.model_fingerprint,
            schema_version=SCHEMA_VERSION,
            geom_model=geom_model or "region_default",
            mirror_search=True,
        )
        self._identity_scoring_fingerprint = make_identity_scoring_fingerprint(
            backend=matcher.backend,
            model_fingerprint=matcher.model_fingerprint,
            schema_version=SCHEMA_VERSION,
            max_sessions=max_sessions,
            top_k=top_k,
            aggregation_method="mean_top_k",
        )

    # ------------------------------------------------------------------
    # Reference selection
    # ------------------------------------------------------------------

    def select_references(
        self,
        candidates: list[ReferenceImage],
    ) -> list[ReferenceImage]:
        """
        Apply the session cap: keep at most one image per session, up to
        max_sessions distinct sessions.

        The caller is responsible for supplying the globally strongest
        reference image per session in *candidates* (ordered by descending
        strength within each session).  This method only enforces the cap.
        """
        seen_sessions: dict[str, ReferenceImage] = {}
        for img in candidates:
            if img.session_id not in seen_sessions:
                seen_sessions[img.session_id] = img
            if len(seen_sessions) >= self.max_sessions:
                break
        return list(seen_sessions.values())

    # ------------------------------------------------------------------
    # Single-pair scoring
    # ------------------------------------------------------------------

    def _score_one_pair(
        self,
        query: QueryCrop,
        ref: ReferenceImage,
        source_fingerprint: str = "",
        split_fingerprint: str = "",
    ) -> LocalPairScore:
        """Score a single query↔reference pair."""
        t0 = time.perf_counter()

        ps = LocalPairScore(
            schema_version=SCHEMA_VERSION,
            backend=self.matcher.backend,
            model_fingerprint=self.matcher.model_fingerprint,
            scoring_fingerprint=self._scoring_fingerprint,
            source_fingerprint=source_fingerprint,
            split_fingerprint=split_fingerprint,
            query_crop_id=query.crop_id,
            ref_crop_id=ref.crop_id,
            query_crop_kind=query.crop_kind,
            ref_crop_kind=ref.crop_kind,
            region=query.crop_kind,  # region matches crop_kind for local matching
            orientation="original",
            geom_model_used=self.geom_model or (
                "homography" if query.crop_kind == REGION_EAR else "partial_affine"
            ),
        )

        # Load images
        try:
            query_bgr = self._load_image(query.crop_path)
        except LocalMatcherFileError as exc:
            logger.warning("Missing query image: %s", exc)
            ps.missing_file = True
            ps.latency_ms = (time.perf_counter() - t0) * 1000
            return ps

        try:
            ref_bgr = self._load_image(ref.crop_path)
        except LocalMatcherFileError as exc:
            logger.warning("Missing ref image: %s", exc)
            ps.missing_file = True
            ps.latency_ms = (time.perf_counter() - t0) * 1000
            return ps

        # Feature extraction with optional caching
        if self.cache is not None and self.matcher.backend != "loftr":
            query_bundle, _ = self.cache.get_or_extract(
                self.matcher, query_bgr, query.crop_id, orientation="original"
            )
            # Mirror search: get both ref orientations
            ref_bundle_orig, _ = self.cache.get_or_extract(
                self.matcher, ref_bgr, ref.crop_id, orientation="original"
            )
            ref_bundle_flip, _ = self.cache.get_or_extract(
                self.matcher, ref_bgr, ref.crop_id, orientation="flipped"
            )
            result_orig = self.matcher.match_features(
                query_bundle, ref_bundle_orig, query.crop_kind, self.geom_model
            )
            result_flip = self.matcher.match_features(
                query_bundle, ref_bundle_flip, query.crop_kind, self.geom_model
            )
            result = result_flip if result_flip.n_inliers > result_orig.n_inliers else result_orig
        else:
            result = self.matcher.score_pair_strict(
                query_bgr, ref_bgr, query.crop_kind,
                geom_model=self.geom_model,
                mirror_search=True,
            )

        ps.orientation = result.orientation
        ps.geom_model_used = result.geom.model_used
        ps.n_raw_matches = result.geom.n_raw_matches
        ps.n_inliers = result.n_inliers
        ps.inlier_ratio = result.geom.inlier_ratio
        ps.geometric_spread = result.geom.geometric_spread
        ps.score = float(result.n_inliers)
        ps.latency_ms = (time.perf_counter() - t0) * 1000
        return ps

    # ------------------------------------------------------------------
    # Identity scoring
    # ------------------------------------------------------------------

    def score_identity(
        self,
        query_crops: list[QueryCrop],
        reference_sessions: list[ReferenceImage],
        candidate_individual_id: str = "",
        *,
        source_fingerprint: str = "",
        split_fingerprint: str = "",
        aggregation_method: str = "mean_top_k",
    ) -> LocalIdentityScore:
        """
        Compute an aggregated local identity score for a candidate individual.

        Parameters
        ----------
        query_crops:
            One or more query crops.  All must have the same crop_kind.
        reference_sessions:
            Reference images (up to max_sessions after select_references).
            Caller supplies one globally strongest image per session;
            this method enforces the cap.
        candidate_individual_id:
            Identity label for the candidate (informational).
        source_fingerprint, split_fingerprint:
            Pass-through to LocalPairScore for traceability.
        aggregation_method:
            'mean_top_k' (default) | 'mean_all' | 'max_available' (exploratory).

        Raises
        ------
        LocalMatcherRegionError  if query crops or reference images have mixed crop_kinds.
        """
        t0 = time.perf_counter()

        # Validate crop_kind uniformity
        query_kinds = {q.crop_kind for q in query_crops}
        if len(query_kinds) > 1:
            raise LocalMatcherRegionError(
                f"Query crops have mixed crop_kinds: {query_kinds}. "
                "All query crops must have the same crop_kind."
            )
        if not query_crops:
            raise LocalMatcherRegionError("query_crops must not be empty.")
        query_crop_kind = next(iter(query_kinds))

        ref_kinds = {r.crop_kind for r in reference_sessions}
        if len(ref_kinds) > 1:
            raise LocalMatcherRegionError(
                f"Reference images have mixed crop_kinds: {ref_kinds}."
            )

        if reference_sessions:
            ref_crop_kind = next(iter(ref_kinds))
            if ref_crop_kind != query_crop_kind:
                raise LocalMatcherRegionError(
                    f"crop_kind mismatch: query={query_crop_kind!r}, "
                    f"ref={ref_crop_kind!r}. Cannot mix crop kinds in identity scoring."
                )

        # Apply session cap
        selected_refs = self.select_references(reference_sessions)
        n_sessions_used = len({r.session_id for r in selected_refs})
        cache_key = (
            self._identity_scoring_fingerprint,
            aggregation_method,
            tuple(sorted(q.crop_id for q in query_crops)),
            tuple(r.crop_id for r in selected_refs),
            candidate_individual_id,
            source_fingerprint,
            split_fingerprint,
        )
        cached = self._identity_result_cache.get(cache_key)
        if cached is not None:
            return cached

        # Build pair list
        #   - ear: all query ears × all selected reference ears
        #   - body: one-to-one per selected image (first query body vs each ref body)
        pair_scores: list[LocalPairScore] = []
        region_coverage: dict[str, int] = {}
        orientations_attempted: set[str] = set()
        n_missing = 0

        if query_crop_kind == REGION_EAR:
            # All query ears × all selected reference ears
            for q in query_crops:
                for r in selected_refs:
                    ps = self._score_one_pair(
                        q, r, source_fingerprint, split_fingerprint
                    )
                    pair_scores.append(ps)
                    region_coverage[REGION_EAR] = region_coverage.get(REGION_EAR, 0) + 1
                    orientations_attempted.add(ps.orientation)
                    if ps.missing_file:
                        n_missing += 1
        else:
            # Body: one-to-one — one query body vs each selected reference
            q = query_crops[0]
            for r in selected_refs:
                ps = self._score_one_pair(q, r, source_fingerprint, split_fingerprint)
                pair_scores.append(ps)
                region_coverage[REGION_BODY] = region_coverage.get(REGION_BODY, 0) + 1
                orientations_attempted.add(ps.orientation)
                if ps.missing_file:
                    n_missing += 1

        n_valid = sum(1 for ps in pair_scores if ps.score > 0 and not ps.missing_file)
        valid_scores = [ps.score for ps in pair_scores if ps.score > 0 and not ps.missing_file]
        agg_score = _aggregate(valid_scores, aggregation_method, self.top_k)

        identity_score = LocalIdentityScore(
            schema_version=SCHEMA_VERSION,
            backend=self.matcher.backend,
            model_fingerprint=self.matcher.model_fingerprint,
            scoring_fingerprint=self._identity_scoring_fingerprint,
            query_crop_kind=query_crop_kind,
            candidate_individual_id=candidate_individual_id,
            n_pairs_attempted=len(pair_scores),
            n_pairs_valid=n_valid,
            n_pairs_missing_file=n_missing,
            n_sessions_used=n_sessions_used,
            n_sessions_cap=self.max_sessions,
            region_coverage=region_coverage,
            orientations_attempted=orientations_attempted,
            aggregation_method=aggregation_method,
            top_k=self.top_k,
            score=agg_score,
            pair_scores=pair_scores,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        assert_identity_score_integrity(identity_score)
        self._identity_result_cache[cache_key] = identity_score
        return identity_score

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        if not os.path.isfile(path):
            raise LocalMatcherFileError(f"Image file not found: {path!r}")
        img = cv2.imread(path)
        if img is None:
            raise LocalMatcherFileError(f"cv2.imread returned None for {path!r}")
        return img
