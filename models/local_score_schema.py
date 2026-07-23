# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Versioned schemas for local pair and identity scores.

LocalPairScore  — result of matching one query crop against one reference crop.
LocalIdentityScore — aggregated result across all reference pairs for one identity.

Both schemas carry source/split/backend/scoring fingerprints and support
fail-loud integrity checks.  Schema version is LOCAL_SCORE_SCHEMA_VERSION
from config_elephant.py.  Bump the version when any required field is added
or removed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

from configs.config_elephant import LOCAL_SCORE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Schema version sentinel
# ---------------------------------------------------------------------------

SCHEMA_VERSION: str = LOCAL_SCORE_SCHEMA_VERSION

# Required fields that must be non-empty strings in every score record.
_PAIR_REQUIRED_STRING_FIELDS: tuple[str, ...] = (
    "schema_version",
    "backend",
    "model_fingerprint",
    "scoring_fingerprint",
    "query_crop_id",
    "ref_crop_id",
    "query_crop_kind",
    "ref_crop_kind",
    "region",
    "orientation",
    "geom_model_used",
)

_IDENTITY_REQUIRED_STRING_FIELDS: tuple[str, ...] = (
    "schema_version",
    "backend",
    "model_fingerprint",
    "scoring_fingerprint",
    "query_crop_kind",
    "aggregation_method",
)


# ---------------------------------------------------------------------------
# Scoring fingerprint helpers
# ---------------------------------------------------------------------------

def make_scoring_fingerprint(
    backend: str,
    model_fingerprint: str,
    schema_version: str,
    geom_model: str,
    mirror_search: bool,
) -> str:
    """Deterministic fingerprint over the scoring configuration."""
    raw = "|".join([
        backend,
        model_fingerprint,
        schema_version,
        geom_model,
        str(mirror_search),
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def make_identity_scoring_fingerprint(
    backend: str,
    model_fingerprint: str,
    schema_version: str,
    max_sessions: int,
    top_k: int,
    aggregation_method: str,
) -> str:
    raw = "|".join([
        backend,
        model_fingerprint,
        schema_version,
        str(max_sessions),
        str(top_k),
        aggregation_method,
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# LocalPairScore
# ---------------------------------------------------------------------------

@dataclass
class LocalPairScore:
    """
    Score from matching one query crop against one reference crop.

    Fields
    ------
    schema_version:
        Must equal LOCAL_SCORE_SCHEMA_VERSION; fail-loud check on creation.
    backend:
        'lightglue' | 'loftr'.
    model_fingerprint:
        Fingerprint from StrictLocalMatcher.model_fingerprint.
    scoring_fingerprint:
        From make_scoring_fingerprint(); encodes the full scoring config.
    source_fingerprint:
        Content fingerprint of the data source (may be empty string if not set).
    split_fingerprint:
        Fingerprint of the data split assignment (may be empty string).
    query_crop_id / ref_crop_id:
        Identifiers from the crop manifest.
    query_crop_kind / ref_crop_kind:
        Must be equal; enforced at scorer level (logged here for traceability).
    region:
        'ear' | 'body'.
    orientation:
        Reference orientation used: 'original' | 'flipped'.
    geom_model_used:
        RANSAC model: 'homography' | 'partial_affine'.
    n_raw_matches:
        Count of raw feature matches before geometric verification.
    n_inliers:
        Count after geometric verification.
    inlier_ratio:
        n_inliers / n_raw_matches (0.0 if n_raw_matches == 0).
    geometric_spread:
        Spatial spread of inlier points (stddev, pixels).
    score:
        Primary numeric score exposed to callers — equals n_inliers.
        Calibration maps this to a probability; that is separate.
    missing_file:
        True when a required image file was not found (score == 0).
    latency_ms:
        Wall-clock time for the pair scoring call (ms), optional.
    """

    # Schema
    schema_version: str = SCHEMA_VERSION
    backend: str = ""
    model_fingerprint: str = ""
    scoring_fingerprint: str = ""
    source_fingerprint: str = ""
    split_fingerprint: str = ""

    # Identifiers
    query_crop_id: str = ""
    ref_crop_id: str = ""
    query_crop_kind: str = ""
    ref_crop_kind: str = ""

    # Geometry
    region: str = ""
    orientation: str = "original"
    geom_model_used: str = ""
    n_raw_matches: int = 0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    geometric_spread: float = 0.0

    # Score
    score: float = 0.0

    # Diagnostics
    missing_file: bool = False
    latency_ms: float = 0.0


@dataclass
class LocalIdentityScore:
    """
    Aggregated local identity score for one candidate identity.

    Primary aggregation: mean of top-k valid pair scores (default k=2).
    Fallback: mean of available valid pairs when fewer than k exist.
    Max-available is exposed only as an explicit exploratory option.

    Fields
    ------
    schema_version, backend, model_fingerprint, scoring_fingerprint:
        As in LocalPairScore.
    query_crop_kind:
        Must be uniform across all pairs; checked at scorer level.
    candidate_individual_id:
        Identity being compared.
    n_pairs_attempted:
        Total pairs attempted (all reference crops × all query crops).
    n_pairs_valid:
        Pairs with score > 0 (not missing file, not zero inliers).
    n_pairs_missing_file:
        Pairs skipped due to missing image file.
    n_sessions_used:
        Number of distinct reference sessions represented.
    n_sessions_cap:
        Cap applied (LOCAL_IDENTITY_SCORER_MAX_SESSIONS).
    region_coverage:
        Dict mapping region name to pair count, e.g. {'ear': 4, 'body': 2}.
    orientations_attempted:
        Set of orientations tried, e.g. {'original', 'flipped'}.
    aggregation_method:
        'mean_top_k' | 'mean_all' | 'max_available' (last is exploratory only).
    top_k:
        k used in aggregation.
    score:
        Final aggregated score.
    pair_scores:
        All individual LocalPairScore instances (in scoring order).
    latency_ms:
        Total wall-clock time for the full identity score call.
    """

    # Schema
    schema_version: str = SCHEMA_VERSION
    backend: str = ""
    model_fingerprint: str = ""
    scoring_fingerprint: str = ""

    # Identity
    query_crop_kind: str = ""
    candidate_individual_id: str = ""

    # Diagnostics
    n_pairs_attempted: int = 0
    n_pairs_valid: int = 0
    n_pairs_missing_file: int = 0
    n_sessions_used: int = 0
    n_sessions_cap: int = 3
    region_coverage: dict = field(default_factory=dict)
    orientations_attempted: set = field(default_factory=set)

    # Aggregation
    aggregation_method: str = "mean_top_k"
    top_k: int = 2
    score: float = 0.0

    # Sub-scores
    pair_scores: list = field(default_factory=list)

    # Latency
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------

class LocalScoreIntegrityError(ValueError):
    """Raised when a score record fails an integrity check."""


def assert_pair_score_integrity(score: LocalPairScore) -> None:
    """
    Fail loudly if *score* violates schema contracts.

    Checks
    ------
    1. schema_version must equal SCHEMA_VERSION.
    2. All required string fields must be non-empty.
    3. query_crop_kind == ref_crop_kind (same region both sides).
    4. region must be 'ear' or 'body'.
    5. n_inliers, n_raw_matches >= 0.
    6. inlier_ratio in [0, 1].
    7. score >= 0.
    8. orientation must be 'original' or 'flipped'.
    """
    if score.schema_version != SCHEMA_VERSION:
        raise LocalScoreIntegrityError(
            f"schema_version mismatch: expected {SCHEMA_VERSION!r}, "
            f"got {score.schema_version!r}"
        )

    for fname in _PAIR_REQUIRED_STRING_FIELDS:
        val = getattr(score, fname)
        if not isinstance(val, str) or not val:
            raise LocalScoreIntegrityError(
                f"LocalPairScore.{fname} is empty or not a string: {val!r}"
            )

    if score.query_crop_kind != score.ref_crop_kind:
        raise LocalScoreIntegrityError(
            f"crop_kind mismatch: query={score.query_crop_kind!r}, "
            f"ref={score.ref_crop_kind!r}. Scores across crop kinds are invalid."
        )

    if score.region not in ("ear", "body"):
        raise LocalScoreIntegrityError(
            f"region must be 'ear' or 'body', got {score.region!r}"
        )

    if score.n_inliers < 0:
        raise LocalScoreIntegrityError(f"n_inliers must be >= 0, got {score.n_inliers}")
    if score.n_raw_matches < 0:
        raise LocalScoreIntegrityError(
            f"n_raw_matches must be >= 0, got {score.n_raw_matches}"
        )

    if not (0.0 <= score.inlier_ratio <= 1.0 + 1e-6):
        raise LocalScoreIntegrityError(
            f"inlier_ratio must be in [0, 1], got {score.inlier_ratio}"
        )

    if score.score < 0.0:
        raise LocalScoreIntegrityError(
            f"score must be >= 0, got {score.score}"
        )

    if score.orientation not in ("original", "flipped"):
        raise LocalScoreIntegrityError(
            f"orientation must be 'original' or 'flipped', got {score.orientation!r}"
        )


def assert_identity_score_integrity(score: LocalIdentityScore) -> None:
    """
    Fail loudly if *score* violates schema contracts.

    Checks
    ------
    1. schema_version must equal SCHEMA_VERSION.
    2. All required string fields must be non-empty.
    3. aggregation_method must be in the allowed set.
    4. n_pairs_valid <= n_pairs_attempted.
    5. score >= 0.
    6. top_k >= 1.
    7. Each pair_score passes assert_pair_score_integrity.
    8. query_crop_kind uniform across all pair scores.
    9. model_fingerprint and scoring_fingerprint are consistent
       across all pair_scores.
    """
    if score.schema_version != SCHEMA_VERSION:
        raise LocalScoreIntegrityError(
            f"schema_version mismatch: expected {SCHEMA_VERSION!r}, "
            f"got {score.schema_version!r}"
        )

    for fname in _IDENTITY_REQUIRED_STRING_FIELDS:
        val = getattr(score, fname)
        if not isinstance(val, str) or not val:
            raise LocalScoreIntegrityError(
                f"LocalIdentityScore.{fname} is empty or not a string: {val!r}"
            )

    allowed_methods = {"mean_top_k", "mean_all", "max_available"}
    if score.aggregation_method not in allowed_methods:
        raise LocalScoreIntegrityError(
            f"aggregation_method must be one of {sorted(allowed_methods)}, "
            f"got {score.aggregation_method!r}"
        )

    if score.n_pairs_valid > score.n_pairs_attempted:
        raise LocalScoreIntegrityError(
            f"n_pairs_valid ({score.n_pairs_valid}) > n_pairs_attempted "
            f"({score.n_pairs_attempted})"
        )

    if score.score < 0.0:
        raise LocalScoreIntegrityError(f"score must be >= 0, got {score.score}")

    if score.top_k < 1:
        raise LocalScoreIntegrityError(f"top_k must be >= 1, got {score.top_k}")

    # Validate all sub-scores
    for i, ps in enumerate(score.pair_scores):
        try:
            assert_pair_score_integrity(ps)
        except LocalScoreIntegrityError as exc:
            raise LocalScoreIntegrityError(
                f"pair_scores[{i}] failed integrity check: {exc}"
            ) from exc

    if score.pair_scores:
        kinds = {ps.query_crop_kind for ps in score.pair_scores}
        if len(kinds) > 1:
            raise LocalScoreIntegrityError(
                f"query_crop_kind is not uniform across pair_scores: {kinds}"
            )

        fps = {ps.model_fingerprint for ps in score.pair_scores}
        if len(fps) > 1:
            raise LocalScoreIntegrityError(
                f"model_fingerprint varies across pair_scores: {fps}"
            )

        sps = {ps.scoring_fingerprint for ps in score.pair_scores}
        if len(sps) > 1:
            raise LocalScoreIntegrityError(
                f"scoring_fingerprint varies across pair_scores: {sps}"
            )
