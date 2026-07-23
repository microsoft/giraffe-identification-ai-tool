# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Identity-level fusion for BTEH normalized evaluation.

Key design decisions
--------------------
- Rank **identities**, not individual crops.  For each query image and each
  candidate identity, we aggregate channel scores by taking the maximum
  calibrated similarity across the available crops.
- Channels can be absent (body/ear).  When absent they do not contribute
  a score or a weight; their portion of the weight budget is redistributed
  to available channels.  Availability is explicitly reported.
- Score a **consistent candidate identity union**: all identities that
  appear in any channel's shortlist are evaluated on *every* channel (the
  FAISS index is re-queried deeply enough, or the score is set to an
  explicit "missing" sentinel).  A candidate cannot rank from one raw high
  score simply because other channels missed their shortlist.
- Fusion weights must be non-negative and sum to 1 (over enabled channels
  after channel availability renormalisation).
- Weights are fitted on gallery OOF predictions only (not on probe images).

Weight fitting
--------------
Grid/coordinate search over the simplex {w >= 0, sum(w) == 1} optimising
OOF retrieval mAP (primary) and top-1 (secondary tie-break).
The grid step is 0.05 (21 points per axis for 4 channels → feasible).
"""

import logging
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.calibration import Calibrator
from models.oof_calibration import CalibrationSupportError, ChannelOOFResult, _cosine_identity_max

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class IdentityScore:
    """Per-channel and fused scores for one (query, candidate-identity) pair."""

    individual_id: str
    channel_raw: Dict[str, float] = field(default_factory=dict)
    channel_calibrated: Dict[str, float] = field(default_factory=dict)
    channels_available: List[str] = field(default_factory=list)
    fused_score: float = 0.0


@dataclass
class QueryResult:
    """Ranked identity list for one query image."""

    query_image_id: str
    query_individual_id: Optional[str]
    ranked_identities: List[IdentityScore] = field(default_factory=list)
    channels_present: List[str] = field(default_factory=list)
    channels_absent: List[str] = field(default_factory=list)
    accept_threshold: float = 0.0
    top1_correct: Optional[bool] = None
    top5_correct: Optional[bool] = None
    unknown_query: bool = False
    # True when the query's identity was present in the OOF rest-gallery for
    # this fold (i.e. the correct match existed).  False when the identity
    # was absent from the rest gallery (single-session identity).  None when
    # not computed (e.g. probe queries or simulated unknown trials are False).
    identity_in_oof_gallery: Optional[bool] = None
    # Probe sub-type: "temporal", "unseen_identity_onboarding", or None (OOF).
    probe_type: Optional[str] = None
    # True for trials produced by explicit identity-removal from reference.
    # These are genuine unknown simulations (truth identity absent from
    # candidates), NOT merely relabelled queries whose identity stays indexed.
    simulated_unknown: bool = False


# ---------------------------------------------------------------------------
# Retrieval metric helpers
# ---------------------------------------------------------------------------

def _average_precision(ranked_identities: List[IdentityScore], gt_id: str) -> float:
    """Compute average precision for a single ranked result list."""
    hits = 0
    sum_prec = 0.0
    for rank, item in enumerate(ranked_identities, start=1):
        if item.individual_id == gt_id:
            hits += 1
            sum_prec += hits / rank
    if hits == 0:
        return 0.0
    return sum_prec / hits


def compute_map(results: List["QueryResult"], known_ids: Optional[set] = None) -> float:
    aps = []
    for qr in results:
        if qr.query_individual_id is None:
            continue
        if known_ids is not None and qr.query_individual_id not in known_ids:
            continue
        ap = _average_precision(qr.ranked_identities, qr.query_individual_id)
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def compute_top1(results: List["QueryResult"], known_ids: Optional[set] = None) -> float:
    correct = 0
    total = 0
    for qr in results:
        if qr.query_individual_id is None:
            continue
        if known_ids is not None and qr.query_individual_id not in known_ids:
            continue
        if qr.ranked_identities:
            correct += int(qr.ranked_identities[0].individual_id == qr.query_individual_id)
        total += 1
    return correct / total if total else 0.0


def compute_top5(results: List["QueryResult"], known_ids: Optional[set] = None) -> float:
    """Fraction of known queries where truth identity appears in top-5 candidates."""
    correct = 0
    total = 0
    for qr in results:
        if qr.query_individual_id is None:
            continue
        if known_ids is not None and qr.query_individual_id not in known_ids:
            continue
        ranked_ids = [x.individual_id for x in qr.ranked_identities]
        correct += int(qr.query_individual_id in ranked_ids[:5])
        total += 1
    return correct / total if total else 0.0


# ---------------------------------------------------------------------------
# Core identity-level scorer
# ---------------------------------------------------------------------------

class IdentityLevelScorer:
    """
    Score all gallery identities for a set of query images, calibrate each
    channel, and fuse with explicit per-channel weights.

    Parameters
    ----------
    gallery_image_df : DataFrame with image_id, individual_id, session_id
        (gallery split only).
    descriptor_mappings : {channel: normalized descriptor mapping DataFrame}
    embedding_matrices : {channel: np.ndarray} L2-normalised.
    calibrators : {channel: Calibrator}
    weights : {channel: float} – must be non-negative and sum to 1.
        If a channel is absent for a query, its weight is redistributed.
    all_channels : ordered list of all enabled channel names.
    """

    def __init__(
        self,
        gallery_image_df: pd.DataFrame,
        descriptor_mappings: Dict[str, pd.DataFrame],
        embedding_matrices: Dict[str, np.ndarray],
        calibrators: Dict[str, Calibrator],
        weights: Dict[str, float],
        all_channels: List[str],
        accept_threshold: float = 0.5,
    ):
        self.gallery_image_df = gallery_image_df.copy()
        self.gallery_image_df["image_id"] = self.gallery_image_df["image_id"].astype(str)
        self.gallery_image_df["individual_id"] = self.gallery_image_df["individual_id"].astype(str)
        self.descriptor_mappings = descriptor_mappings
        self.embedding_matrices = embedding_matrices
        self.calibrators = calibrators
        self.weights = weights
        self.all_channels = all_channels
        self.accept_threshold = accept_threshold

        # Validate weights.
        self._validate_weights(weights, all_channels)

        # Pre-build gallery index per channel.
        gallery_ids_set = set(self.gallery_image_df["image_id"])
        self._gallery_emb_rows: Dict[str, Dict[str, np.ndarray]] = {}
        self._gallery_indiv_arr: Dict[str, np.ndarray] = {}
        self._gallery_unique_ids: Dict[str, List[str]] = {}

        for ch, dm in descriptor_mappings.items():
            if dm is None or dm.empty:
                continue
            dm = dm.copy()
            dm["image_id"] = dm["image_id"].astype(str)
            gm = dm[dm["image_id"].isin(gallery_ids_set)].reset_index(drop=True)
            if gm.empty:
                continue

            image_to_indiv = dict(
                zip(
                    self.gallery_image_df["image_id"],
                    self.gallery_image_df["individual_id"],
                )
            )
            rows = gm["embedding_row"].astype(int).to_numpy()
            self._gallery_emb_rows[ch] = {
                str(iid): [] for iid in gallery_ids_set
            }
            img_row_map: Dict[str, List[int]] = {}
            for r in gm.itertuples(index=False):
                iid = str(r.image_id)
                img_row_map.setdefault(iid, []).append(int(r.embedding_row))

            self._gallery_emb_rows[ch] = img_row_map

            # Flat arrays for identity-max computation.
            self._gallery_indiv_arr[ch] = np.array(
                [image_to_indiv.get(str(r.image_id), "") for r in gm.itertuples(index=False)]
            )
            self._gallery_unique_ids[ch] = sorted(
                set(self._gallery_indiv_arr[ch]) - {""}
            )

    @staticmethod
    def _validate_weights(weights: Dict[str, float], channels: List[str]) -> None:
        for ch in channels:
            w = weights.get(ch, 0.0)
            if w < 0:
                raise ValueError(f"Fusion weight for channel '{ch}' is negative: {w}")
        total = sum(weights.get(ch, 0.0) for ch in channels)
        if not np.isclose(total, 1.0, atol=1e-5):
            raise ValueError(
                f"Fusion weights do not sum to 1.0 over enabled channels: sum={total:.6f}"
            )

    def score_query(
        self,
        query_image_id: str,
        query_emb_rows: Dict[str, np.ndarray],
        query_individual_id: Optional[str] = None,
    ) -> QueryResult:
        """
        Score all gallery identities for one query image.

        Parameters
        ----------
        query_image_id : str
        query_emb_rows : {channel: np.ndarray shape (n_crops, D)} – may be
            empty/missing for absent channels.
        query_individual_id : ground-truth identity if known (for evaluation).

        Returns
        -------
        QueryResult with ranked_identities sorted by fused_score descending.
        """
        # Determine which channels are present for this query.
        channels_present = [
            ch for ch in self.all_channels
            if ch in query_emb_rows and query_emb_rows[ch] is not None and len(query_emb_rows[ch]) > 0
        ]
        channels_absent = [ch for ch in self.all_channels if ch not in channels_present]

        if not channels_present:
            logger.warning(
                "Query image '%s' has no embeddings in any channel; returning empty result.",
                query_image_id,
            )
            return QueryResult(
                query_image_id=query_image_id,
                query_individual_id=query_individual_id,
                channels_present=[],
                channels_absent=channels_absent,
                accept_threshold=self.accept_threshold,
            )

        # Renormalise weights over available channels.
        total_w = sum(self.weights.get(ch, 0.0) for ch in channels_present)
        if total_w <= 0:
            logger.warning(
                "Query '%s': all present channels have weight 0; using equal weights.",
                query_image_id,
            )
            renorm_weights = {ch: 1.0 / len(channels_present) for ch in channels_present}
        else:
            renorm_weights = {
                ch: self.weights.get(ch, 0.0) / total_w for ch in channels_present
            }

        # Build identity candidate union across all channels.
        candidate_ids: set = set()
        for ch in channels_present:
            candidate_ids.update(self._gallery_unique_ids.get(ch, []))

        if not candidate_ids:
            return QueryResult(
                query_image_id=query_image_id,
                query_individual_id=query_individual_id,
                channels_present=channels_present,
                channels_absent=channels_absent,
                accept_threshold=self.accept_threshold,
            )

        unique_candidate_ids = sorted(candidate_ids)

        # Compute per-channel identity-level raw and calibrated scores.
        channel_identity_raw: Dict[str, Dict[str, float]] = {}
        channel_identity_cal: Dict[str, Dict[str, float]] = {}

        for ch in channels_present:
            q_emb = query_emb_rows[ch]  # (n_q_crops, D)
            ref_matrix = self.embedding_matrices[ch]
            ref_indiv_arr = self._gallery_indiv_arr.get(ch)
            if ref_indiv_arr is None:
                continue

            # Score against full gallery for this channel.
            gallery_rows = []
            gallery_mapping = self.descriptor_mappings.get(ch)
            if gallery_mapping is None:
                continue
            gallery_mapping = gallery_mapping.copy()
            gallery_mapping["image_id"] = gallery_mapping["image_id"].astype(str)
            gal_ids_set = set(self.gallery_image_df["image_id"])
            gm = gallery_mapping[gallery_mapping["image_id"].isin(gal_ids_set)]
            if gm.empty:
                continue

            image_to_indiv = dict(
                zip(
                    self.gallery_image_df["image_id"],
                    self.gallery_image_df["individual_id"],
                )
            )
            gm_indiv_arr = np.array(
                [image_to_indiv.get(str(r.image_id), "") for r in gm.itertuples(index=False)]
            )
            gm_rows = gm["embedding_row"].astype(int).to_numpy()
            gm_matrix = ref_matrix[gm_rows]

            identity_scores = _cosine_identity_max(
                q_emb, gm_matrix, gm_indiv_arr, unique_candidate_ids
            )
            channel_identity_raw[ch] = identity_scores

            # Calibrate.
            cal = self.calibrators.get(ch)
            if cal is not None:
                all_raw_scores = np.array(
                    [identity_scores.get(rid, 0.0) for rid in unique_candidate_ids],
                    dtype=np.float64,
                )
                all_cal = cal.transform(all_raw_scores)
                channel_identity_cal[ch] = {
                    rid: float(all_cal[i])
                    for i, rid in enumerate(unique_candidate_ids)
                }
            else:
                channel_identity_cal[ch] = identity_scores

        # Fuse.
        identity_results: List[IdentityScore] = []
        for rid in unique_candidate_ids:
            ch_raw = {
                ch: channel_identity_raw[ch].get(rid, 0.0)
                for ch in channels_present
                if ch in channel_identity_raw
            }
            ch_cal = {
                ch: channel_identity_cal[ch].get(rid, 0.0)
                for ch in channels_present
                if ch in channel_identity_cal
            }
            available = [ch for ch in channels_present if ch in ch_cal]
            if not available:
                fused = 0.0
            else:
                fused = sum(renorm_weights.get(ch, 0.0) * ch_cal[ch] for ch in available)

            identity_results.append(
                IdentityScore(
                    individual_id=rid,
                    channel_raw=ch_raw,
                    channel_calibrated=ch_cal,
                    channels_available=available,
                    fused_score=float(fused),
                )
            )

        identity_results.sort(key=lambda x: x.fused_score, reverse=True)

        # Determine known identities for this query.
        gallery_known_ids = set(self.gallery_image_df["individual_id"])
        unknown_query = (
            query_individual_id is not None
            and query_individual_id not in gallery_known_ids
        )

        # Top-k correctness (only for known queries).
        top1_correct = None
        top5_correct = None
        if query_individual_id and not unknown_query:
            ranked_ids = [x.individual_id for x in identity_results]
            top1_correct = bool(ranked_ids[:1] and ranked_ids[0] == query_individual_id)
            top5_correct = query_individual_id in ranked_ids[:5]

        return QueryResult(
            query_image_id=query_image_id,
            query_individual_id=query_individual_id,
            ranked_identities=identity_results,
            channels_present=channels_present,
            channels_absent=channels_absent,
            accept_threshold=self.accept_threshold,
            top1_correct=top1_correct,
            top5_correct=top5_correct,
            unknown_query=unknown_query,
        )


# ---------------------------------------------------------------------------
# OOF identity-level score builder (for weight fitting)
# ---------------------------------------------------------------------------

def build_oof_identity_scores(
    gallery_image_df: pd.DataFrame,
    descriptor_mappings: Dict[str, pd.DataFrame],
    embedding_matrices: Dict[str, np.ndarray],
    calibrators: Dict[str, Calibrator],
    all_channels: List[str],
) -> List[QueryResult]:
    """
    Build identity-level OOF predictions from gallery (session-based folds),
    using fitted calibrators, for weight optimisation.

    Returns a list of QueryResult (one per gallery image that has at least
    one embedding), using session-based OOF fold exclusion.
    """
    gallery_image_df = gallery_image_df.copy()
    gallery_image_df["image_id"] = gallery_image_df["image_id"].astype(str)
    gallery_image_df["individual_id"] = gallery_image_df["individual_id"].astype(str)
    gallery_image_df["session_id"] = gallery_image_df["session_id"].astype(str)

    sessions = sorted(gallery_image_df["session_id"].unique())
    all_results: List[QueryResult] = []

    # Build gallery query-embedding lookup for each channel and session.
    gallery_ids_set = set(gallery_image_df["image_id"])
    channel_img_emb: Dict[str, Dict[str, np.ndarray]] = {}
    for ch, dm in descriptor_mappings.items():
        if dm is None or dm.empty:
            continue
        dm = dm.copy()
        dm["image_id"] = dm["image_id"].astype(str)
        gm = dm[dm["image_id"].isin(gallery_ids_set)]
        rows_per_img: Dict[str, List[int]] = {}
        for r in gm.itertuples(index=False):
            rows_per_img.setdefault(str(r.image_id), []).append(int(r.embedding_row))
        channel_img_emb[ch] = {
            iid: embedding_matrices[ch][np.array(rows, dtype=int)]
            for iid, rows in rows_per_img.items()
        }

    for session_id in sessions:
        pq_df = gallery_image_df[gallery_image_df["session_id"] == session_id]
        rest_df = gallery_image_df[gallery_image_df["session_id"] != session_id]
        if len(rest_df) < 2:
            continue

        rest_ids_set = set(rest_df["image_id"])
        image_to_indiv = dict(
            zip(gallery_image_df["image_id"], gallery_image_df["individual_id"])
        )

        # Build rest matrices per channel.
        rest_channel_matrix: Dict[str, np.ndarray] = {}
        rest_channel_indiv: Dict[str, np.ndarray] = {}
        rest_channel_ids: Dict[str, List[str]] = {}

        for ch, dm in descriptor_mappings.items():
            if dm is None or dm.empty:
                continue
            dm = dm.copy()
            dm["image_id"] = dm["image_id"].astype(str)
            gm_rest = dm[dm["image_id"].isin(rest_ids_set)]
            if gm_rest.empty:
                continue
            rows = gm_rest["embedding_row"].astype(int).to_numpy()
            rest_channel_matrix[ch] = embedding_matrices[ch][rows]
            rest_channel_indiv[ch] = np.array(
                [image_to_indiv.get(str(iid), "") for iid in gm_rest["image_id"].astype(str)]
            )
            rest_channel_ids[ch] = sorted(
                {x for x in rest_channel_indiv[ch] if x}
            )

        for _, q_row in pq_df.iterrows():
            q_img_id = str(q_row["image_id"])
            q_indiv = str(q_row["individual_id"])

            query_emb_rows: Dict[str, np.ndarray] = {
                ch: channel_img_emb[ch][q_img_id]
                for ch in all_channels
                if ch in channel_img_emb and q_img_id in channel_img_emb[ch]
            }
            if not query_emb_rows:
                continue

            channels_present = list(query_emb_rows.keys())
            candidate_ids: set = set()
            for ch in channels_present:
                candidate_ids.update(rest_channel_ids.get(ch, []))
            if not candidate_ids:
                continue

            identity_in_oof_gallery = q_indiv in candidate_ids
            unique_ids = sorted(candidate_ids)

            # Per-channel raw and calibrated scores.
            ch_raw: Dict[str, Dict[str, float]] = {}
            ch_cal: Dict[str, Dict[str, float]] = {}
            for ch in channels_present:
                if ch not in rest_channel_matrix:
                    continue
                scores_dict = _cosine_identity_max(
                    query_emb_rows[ch],
                    rest_channel_matrix[ch],
                    rest_channel_indiv[ch],
                    unique_ids,
                )
                ch_raw[ch] = scores_dict
                cal = calibrators.get(ch)
                if cal is not None:
                    raw_arr = np.array([scores_dict.get(rid, 0.0) for rid in unique_ids])
                    cal_arr = cal.transform(raw_arr)
                    ch_cal[ch] = {rid: float(cal_arr[i]) for i, rid in enumerate(unique_ids)}
                else:
                    ch_cal[ch] = scores_dict

            identity_scores_list = [
                IdentityScore(
                    individual_id=rid,
                    channel_raw={ch: ch_raw[ch].get(rid, 0.0) for ch in ch_raw},
                    channel_calibrated={ch: ch_cal[ch].get(rid, 0.0) for ch in ch_cal},
                    channels_available=list(ch_cal.keys()),
                    fused_score=0.0,
                )
                for rid in unique_ids
            ]

            all_results.append(
                QueryResult(
                    query_image_id=q_img_id,
                    query_individual_id=q_indiv,
                    ranked_identities=identity_scores_list,
                    channels_present=channels_present,
                    channels_absent=[ch for ch in all_channels if ch not in channels_present],
                    identity_in_oof_gallery=identity_in_oof_gallery,
                )
            )

    return all_results


# ---------------------------------------------------------------------------
# Weight fitting via grid search
# ---------------------------------------------------------------------------

def _apply_weights_and_rank(
    oof_results: List[QueryResult],
    weights: Dict[str, float],
    all_channels: List[str],
) -> List[QueryResult]:
    """Apply explicit weights and re-rank OOF results (returns new list)."""
    new_results = []
    for qr in oof_results:
        channels_present = qr.channels_present
        total_w = sum(weights.get(ch, 0.0) for ch in channels_present)
        if total_w <= 0:
            w_renorm = {ch: 1.0 / len(channels_present) for ch in channels_present}
        else:
            w_renorm = {ch: weights.get(ch, 0.0) / total_w for ch in channels_present}

        new_ranked = []
        for ident in qr.ranked_identities:
            fused = sum(
                w_renorm.get(ch, 0.0) * ident.channel_calibrated.get(ch, 0.0)
                for ch in channels_present
            )
            new_ranked.append(
                IdentityScore(
                    individual_id=ident.individual_id,
                    channel_raw=ident.channel_raw,
                    channel_calibrated=ident.channel_calibrated,
                    channels_available=ident.channels_available,
                    fused_score=float(fused),
                )
            )
        new_ranked.sort(key=lambda x: x.fused_score, reverse=True)

        new_results.append(
            QueryResult(
                query_image_id=qr.query_image_id,
                query_individual_id=qr.query_individual_id,
                ranked_identities=new_ranked,
                channels_present=channels_present,
                channels_absent=qr.channels_absent,
                identity_in_oof_gallery=qr.identity_in_oof_gallery,
            )
        )
    return new_results


def fit_fusion_weights(
    oof_results: List[QueryResult],
    all_channels: List[str],
    grid_step: float = 0.05,
) -> Tuple[Dict[str, float], Dict]:
    """
    Deterministic constrained grid search over the weight simplex
    {w >= 0, sum == 1} optimising OOF retrieval mAP (primary) and
    top-1 (secondary tie-break).

    Only gallery OOF predictions are used — probe images must not appear
    in oof_results.

    Parameters
    ----------
    oof_results : list of QueryResult with filled channel_calibrated scores.
    all_channels : ordered list of enabled channel names.
    grid_step : step size for weight grid (default 0.05).

    Returns
    -------
    best_weights : {channel: float}
    diagnostics : dict with search summary
    """
    n_ch = len(all_channels)
    if n_ch == 0:
        raise ValueError("fit_fusion_weights: no channels provided.")

    if not oof_results:
        raise ValueError("fit_fusion_weights: no OOF results provided.")

    n_steps = int(round(1.0 / grid_step))

    # Generate all weight combinations summing to 1 over the simplex.
    # For efficiency, parameterise with n_ch-1 free variables.
    best_map = -1.0
    best_top1 = -1.0
    best_w = {ch: 1.0 / n_ch for ch in all_channels}
    candidate_count = 0

    def _grid_weight_gen(n: int, remaining: float, step: float):
        """Generate all non-negative integer multiples of step summing <= remaining."""
        if n == 1:
            yield (remaining,)
            return
        k = 0
        while k * step <= remaining + 1e-9:
            w_k = round(k * step, 10)
            for rest in _grid_weight_gen(n - 1, remaining - w_k, step):
                yield (w_k,) + rest
            k += 1

    for w_tuple in _grid_weight_gen(n_ch, 1.0, grid_step):
        if abs(sum(w_tuple) - 1.0) > 1e-6:
            continue
        weights = {ch: float(w) for ch, w in zip(all_channels, w_tuple)}
        candidate_count += 1
        ranked = _apply_weights_and_rank(oof_results, weights, all_channels)
        map_score = compute_map(ranked)
        top1_score = compute_top1(ranked)

        if map_score > best_map or (
            abs(map_score - best_map) < 1e-8 and top1_score > best_top1
        ):
            best_map = map_score
            best_top1 = top1_score
            best_w = weights

    logger.info(
        "fit_fusion_weights: %d candidates evaluated; best mAP=%.4f top1=%.4f weights=%s",
        candidate_count,
        best_map,
        best_top1,
        {ch: round(v, 4) for ch, v in best_w.items()},
    )

    diagnostics = {
        "grid_step": grid_step,
        "n_candidates_evaluated": candidate_count,
        "best_map": round(best_map, 6),
        "best_top1": round(best_top1, 6),
        "best_weights": {ch: round(v, 6) for ch, v in best_w.items()},
    }
    return best_w, diagnostics


# ---------------------------------------------------------------------------
# Gallery-identity-removal unknown simulation
# ---------------------------------------------------------------------------

def simulate_gallery_unknown_scores(
    gallery_image_df: pd.DataFrame,
    descriptor_mappings: Dict[str, pd.DataFrame],
    embedding_matrices: Dict[str, np.ndarray],
    calibrators: Dict[str, Calibrator],
    weights: Dict[str, float],
    all_channels: List[str],
    n_queries_per_identity: int = 1,
) -> List[QueryResult]:
    """
    Simulate unknown-query scores via gallery identity removal.

    For each gallery identity I:
      1. Pick up to ``n_queries_per_identity`` representative images of I
         (deterministic: first images by sorted image_id).
      2. Construct a restricted gallery by excluding **all** crops/images of
         identity I.
      3. Score each query against the restricted gallery using the provided
         calibrators and fusion weights, applying identity-level max aggregation.
      4. Return a QueryResult per trial with fused_score populated and
         ``identity_in_oof_gallery=False``.

    The top-1 fused score for each trial is the best wrong-identity confidence
    and forms the "unknown" distribution for threshold fitting.

    Parameters
    ----------
    gallery_image_df : DataFrame with image_id, individual_id (gallery only).
    descriptor_mappings : {channel: mapping DataFrame}
    embedding_matrices  : {channel: np.ndarray} L2-normalised.
    calibrators         : {channel: Calibrator}
    weights             : {channel: float} non-negative, summing to 1.
    all_channels        : ordered list of enabled channel names.
    n_queries_per_identity : max images used as queries per identity.

    Returns
    -------
    List[QueryResult] – one entry per (identity, query-image) trial.

    Raises
    ------
    CalibrationSupportError
        If the gallery has fewer than 2 identities or if no unknown trials
        can be constructed (no identity has any embeddings).
    """
    gallery_image_df = gallery_image_df.copy()
    gallery_image_df["image_id"] = gallery_image_df["image_id"].astype(str)
    gallery_image_df["individual_id"] = gallery_image_df["individual_id"].astype(str)

    all_identities = sorted(gallery_image_df["individual_id"].unique())
    if len(all_identities) < 2:
        raise CalibrationSupportError(
            "simulate_gallery_unknown_scores: gallery must have at least 2 "
            "identities to construct unknown trials via identity removal."
        )

    # Pre-build per-channel {image_id → embedding matrix} for gallery images.
    gallery_ids_set = set(gallery_image_df["image_id"])
    img_to_indiv = dict(
        zip(gallery_image_df["image_id"], gallery_image_df["individual_id"])
    )
    channel_img_emb: Dict[str, Dict[str, np.ndarray]] = {}

    for ch in all_channels:
        dm = descriptor_mappings.get(ch)
        emb = embedding_matrices.get(ch)
        if dm is None or dm.empty or emb is None:
            continue
        dm = dm.copy()
        dm["image_id"] = dm["image_id"].astype(str)
        gm = dm[dm["image_id"].isin(gallery_ids_set)]
        rows_per_img: Dict[str, List[int]] = {}
        for r in gm.itertuples(index=False):
            rows_per_img.setdefault(str(r.image_id), []).append(int(r.embedding_row))
        channel_img_emb[ch] = {
            iid: emb[np.array(rows, dtype=int)]
            for iid, rows in rows_per_img.items()
        }

    results: List[QueryResult] = []

    for identity in all_identities:
        # Representative query images for this identity (deterministic).
        identity_images = sorted(
            gallery_image_df[
                gallery_image_df["individual_id"] == identity
            ]["image_id"].tolist()
        )
        query_images = identity_images[:n_queries_per_identity]

        # Restricted gallery: exclude all images of this identity.
        rest_ids_set = {
            iid for iid in gallery_ids_set
            if img_to_indiv.get(iid) != identity
        }
        if not rest_ids_set:
            continue  # shouldn't happen with len(all_identities) >= 2

        # Build per-channel rest reference matrices (including the identity
        # strings so we can call _cosine_identity_max).
        rest_channel_matrix: Dict[str, np.ndarray] = {}
        rest_channel_indiv: Dict[str, np.ndarray] = {}
        rest_channel_unique_ids: Dict[str, List[str]] = {}

        for ch in all_channels:
            dm = descriptor_mappings.get(ch)
            emb = embedding_matrices.get(ch)
            if dm is None or dm.empty or emb is None:
                continue
            dm = dm.copy()
            dm["image_id"] = dm["image_id"].astype(str)
            gm_rest = dm[dm["image_id"].isin(rest_ids_set)]
            if gm_rest.empty:
                continue
            rows = gm_rest["embedding_row"].astype(int).to_numpy()
            rest_channel_matrix[ch] = emb[rows]
            rest_channel_indiv[ch] = np.array(
                [img_to_indiv.get(str(iid), "") for iid in gm_rest["image_id"].astype(str)]
            )
            rest_channel_unique_ids[ch] = sorted(
                {x for x in rest_channel_indiv[ch] if x and x != identity}
            )

        for q_img_id in query_images:
            # Collect query embeddings across channels.
            query_emb_rows: Dict[str, np.ndarray] = {
                ch: channel_img_emb[ch][q_img_id]
                for ch in all_channels
                if ch in channel_img_emb and q_img_id in channel_img_emb[ch]
            }
            if not query_emb_rows:
                continue  # no embeddings for this image in any channel

            channels_present = list(query_emb_rows.keys())
            candidate_ids: set = set()
            for ch in channels_present:
                candidate_ids.update(rest_channel_unique_ids.get(ch, []))
            if not candidate_ids:
                continue

            unique_ids = sorted(candidate_ids)

            # Renormalise weights over present channels.
            total_w = sum(weights.get(ch, 0.0) for ch in channels_present)
            if total_w <= 0:
                w_renorm = {ch: 1.0 / len(channels_present) for ch in channels_present}
            else:
                w_renorm = {
                    ch: weights.get(ch, 0.0) / total_w for ch in channels_present
                }

            # Per-channel raw → calibrated scores.
            ch_raw: Dict[str, Dict[str, float]] = {}
            ch_cal: Dict[str, Dict[str, float]] = {}

            for ch in channels_present:
                if ch not in rest_channel_matrix:
                    continue
                scores_dict = _cosine_identity_max(
                    query_emb_rows[ch],
                    rest_channel_matrix[ch],
                    rest_channel_indiv[ch],
                    unique_ids,
                )
                ch_raw[ch] = scores_dict
                cal = calibrators.get(ch)
                if cal is not None:
                    raw_arr = np.array(
                        [scores_dict.get(rid, 0.0) for rid in unique_ids],
                        dtype=np.float64,
                    )
                    cal_arr = cal.transform(raw_arr)
                    ch_cal[ch] = {rid: float(cal_arr[i]) for i, rid in enumerate(unique_ids)}
                else:
                    ch_cal[ch] = scores_dict

            available_chs = [ch for ch in channels_present if ch in ch_cal]
            if not available_chs:
                continue

            # Build IdentityScore list with fused scores applied.
            identity_scores_list: List[IdentityScore] = []
            for rid in unique_ids:
                fused = sum(
                    w_renorm.get(ch, 0.0) * ch_cal[ch].get(rid, 0.0)
                    for ch in available_chs
                )
                identity_scores_list.append(
                    IdentityScore(
                        individual_id=rid,
                        channel_raw={ch: ch_raw.get(ch, {}).get(rid, 0.0) for ch in ch_raw},
                        channel_calibrated={ch: ch_cal[ch].get(rid, 0.0) for ch in ch_cal},
                        channels_available=available_chs,
                        fused_score=float(fused),
                    )
                )

            identity_scores_list.sort(key=lambda x: x.fused_score, reverse=True)

            results.append(
                QueryResult(
                    query_image_id=q_img_id,
                    query_individual_id=identity,
                    ranked_identities=identity_scores_list,
                    channels_present=channels_present,
                    channels_absent=[
                        ch for ch in all_channels if ch not in channels_present
                    ],
                    identity_in_oof_gallery=False,
                )
            )

    if not results:
        raise CalibrationSupportError(
            "simulate_gallery_unknown_scores: could not construct any unknown "
            "trials. Verify that gallery images have embeddings in at least one "
            "channel."
        )

    logger.info(
        "simulate_gallery_unknown_scores: %d unknown trials from %d identities.",
        len(results),
        len(all_identities),
    )
    return results


# ---------------------------------------------------------------------------
# Probe-based open-set unknown simulation (identity removal)
# ---------------------------------------------------------------------------

def simulate_probe_unknown_trials(
    probe_df: pd.DataFrame,
    combined_gallery_df: pd.DataFrame,
    descriptor_mappings: Dict[str, pd.DataFrame],
    embedding_matrices: Dict[str, np.ndarray],
    calibrators: Dict[str, Calibrator],
    weights: Dict[str, float],
    all_channels: List[str],
    probe_emb_mappings: Dict[str, pd.DataFrame],
    probe_emb_matrices: Dict[str, np.ndarray],
) -> List[QueryResult]:
    """
    Simulate open-set unknown trials using actual held-out probe images.

    For each probe image q with true identity I:
      1. Build a restricted reference gallery by removing **all** crops/images
         of identity I from ``combined_gallery_df``.
      2. Score q against this restricted reference using the provided
         calibrators and fusion weights.
      3. Return a QueryResult per trial with ``simulated_unknown=True``.

    The truth identity is GUARANTEED absent from each trial's candidate set by
    construction (strict identity removal).  Callers can verify this invariant
    via the returned ``QueryResult.ranked_identities``.

    Parameters
    ----------
    probe_df : DataFrame with image_id, individual_id columns (probe images).
    combined_gallery_df : DataFrame with image_id, individual_id (reference).
        Must contain NO probe images.
    descriptor_mappings : {channel: DataFrame} for reference gallery crops.
    embedding_matrices : {channel: np.ndarray} L2-normalised reference.
    calibrators : {channel: Calibrator}
    weights : {channel: float} non-negative, summing to 1.
    all_channels : ordered list of enabled channel names.
    probe_emb_mappings : {channel: DataFrame} for probe image crops.
    probe_emb_matrices : {channel: np.ndarray} L2-normalised probe embeddings.

    Returns
    -------
    List[QueryResult] with ``simulated_unknown=True`` and
    ``identity_in_oof_gallery=False`` for every entry.  The truth identity
    is absent from ``ranked_identities`` in every trial.
    """
    probe_df = probe_df.copy()
    probe_df["image_id"] = probe_df["image_id"].astype(str)
    probe_df["individual_id"] = probe_df["individual_id"].astype(str)

    combined_gallery_df = combined_gallery_df.copy()
    combined_gallery_df["image_id"] = combined_gallery_df["image_id"].astype(str)
    combined_gallery_df["individual_id"] = combined_gallery_df["individual_id"].astype(str)

    gallery_ids_set = set(combined_gallery_df["image_id"])
    img_to_indiv = dict(
        zip(combined_gallery_df["image_id"], combined_gallery_df["individual_id"])
    )

    # Pre-build per-channel {image_id → embedding array} for probe images.
    probe_img_emb: Dict[str, Dict[str, np.ndarray]] = {}
    for ch in all_channels:
        dm = probe_emb_mappings.get(ch)
        emb = probe_emb_matrices.get(ch)
        if dm is None or dm.empty or emb is None:
            continue
        dm = dm.copy()
        dm["image_id"] = dm["image_id"].astype(str)
        probe_ids_set = set(probe_df["image_id"])
        dm_probe = dm[dm["image_id"].isin(probe_ids_set)]
        rows_per_img: Dict[str, List[int]] = {}
        for r in dm_probe.itertuples(index=False):
            rows_per_img.setdefault(str(r.image_id), []).append(int(r.embedding_row))
        probe_img_emb[ch] = {
            iid: emb[np.array(rows, dtype=int)]
            for iid, rows in rows_per_img.items()
        }

    # Pre-build per-channel gallery data (all channels, all gallery images).
    channel_gal_dm: Dict[str, pd.DataFrame] = {}
    channel_gal_matrix: Dict[str, np.ndarray] = {}
    channel_gal_indiv: Dict[str, np.ndarray] = {}

    for ch in all_channels:
        dm = descriptor_mappings.get(ch)
        emb = embedding_matrices.get(ch)
        if dm is None or dm.empty or emb is None:
            continue
        dm = dm.copy()
        dm["image_id"] = dm["image_id"].astype(str)
        gm = dm[dm["image_id"].isin(gallery_ids_set)].reset_index(drop=True)
        if gm.empty:
            continue
        rows = gm["embedding_row"].astype(int).to_numpy()
        channel_gal_dm[ch] = gm
        channel_gal_matrix[ch] = emb[rows]
        channel_gal_indiv[ch] = np.array(
            [img_to_indiv.get(str(iid), "") for iid in gm["image_id"].astype(str)]
        )

    results: List[QueryResult] = []

    for _, probe_row in probe_df.iterrows():
        q_img_id = str(probe_row["image_id"])
        truth_id = str(probe_row["individual_id"])

        # Gather query embeddings.
        query_emb_rows: Dict[str, np.ndarray] = {
            ch: probe_img_emb[ch][q_img_id]
            for ch in all_channels
            if ch in probe_img_emb and q_img_id in probe_img_emb[ch]
        }
        if not query_emb_rows:
            continue

        channels_present = list(query_emb_rows.keys())

        # Build restricted reference per channel (remove truth identity).
        rest_ch_matrix: Dict[str, np.ndarray] = {}
        rest_ch_indiv: Dict[str, np.ndarray] = {}
        rest_ch_unique_ids: Dict[str, List[str]] = {}

        for ch in channels_present:
            if ch not in channel_gal_indiv:
                continue
            mask = channel_gal_indiv[ch] != truth_id
            if not mask.any():
                continue
            rest_ch_matrix[ch] = channel_gal_matrix[ch][mask]
            rest_ch_indiv[ch] = channel_gal_indiv[ch][mask]
            rest_ch_unique_ids[ch] = sorted(
                {x for x in rest_ch_indiv[ch] if x and x != truth_id}
            )

        candidate_ids: set = set()
        for ch in channels_present:
            candidate_ids.update(rest_ch_unique_ids.get(ch, []))
        if not candidate_ids:
            continue

        # Safety check: truth identity must be absent from candidates.
        assert truth_id not in candidate_ids, (
            f"simulate_probe_unknown_trials: truth identity '{truth_id}' found in "
            f"candidate set after removal for probe '{q_img_id}'. "
            "This indicates a data integrity bug."
        )

        unique_ids = sorted(candidate_ids)

        # Renormalise weights.
        total_w = sum(weights.get(ch, 0.0) for ch in channels_present)
        if total_w <= 0:
            w_renorm = {ch: 1.0 / len(channels_present) for ch in channels_present}
        else:
            w_renorm = {ch: weights.get(ch, 0.0) / total_w for ch in channels_present}

        # Per-channel raw → calibrated scores.
        ch_raw: Dict[str, Dict[str, float]] = {}
        ch_cal: Dict[str, Dict[str, float]] = {}

        for ch in channels_present:
            if ch not in rest_ch_matrix:
                continue
            scores_dict = _cosine_identity_max(
                query_emb_rows[ch],
                rest_ch_matrix[ch],
                rest_ch_indiv[ch],
                unique_ids,
            )
            ch_raw[ch] = scores_dict
            cal = calibrators.get(ch)
            if cal is not None:
                raw_arr = np.array(
                    [scores_dict.get(rid, 0.0) for rid in unique_ids],
                    dtype=np.float64,
                )
                cal_arr = cal.transform(raw_arr)
                ch_cal[ch] = {rid: float(cal_arr[i]) for i, rid in enumerate(unique_ids)}
            else:
                ch_cal[ch] = scores_dict

        available_chs = [ch for ch in channels_present if ch in ch_cal]
        if not available_chs:
            continue

        identity_scores_list: List[IdentityScore] = []
        for rid in unique_ids:
            fused = sum(
                w_renorm.get(ch, 0.0) * ch_cal[ch].get(rid, 0.0)
                for ch in available_chs
            )
            identity_scores_list.append(
                IdentityScore(
                    individual_id=rid,
                    channel_raw={ch: ch_raw.get(ch, {}).get(rid, 0.0) for ch in ch_raw},
                    channel_calibrated={ch: ch_cal[ch].get(rid, 0.0) for ch in ch_cal},
                    channels_available=available_chs,
                    fused_score=float(fused),
                )
            )

        identity_scores_list.sort(key=lambda x: x.fused_score, reverse=True)

        results.append(
            QueryResult(
                query_image_id=q_img_id,
                query_individual_id=truth_id,
                ranked_identities=identity_scores_list,
                channels_present=channels_present,
                channels_absent=[ch for ch in all_channels if ch not in channels_present],
                unknown_query=True,
                simulated_unknown=True,
                identity_in_oof_gallery=False,
                probe_type="simulated_unknown",
            )
        )

    logger.info(
        "simulate_probe_unknown_trials: %d unknown trials from %d probe images.",
        len(results),
        len(probe_df),
    )
    return results


# ---------------------------------------------------------------------------
# Calibration flatness diagnostic
# ---------------------------------------------------------------------------

def check_calibration_flatness(
    calibrators: Dict[str, "Calibrator"],
    sample_scores: Optional[np.ndarray] = None,
    n_sample: int = 200,
) -> Dict[str, dict]:
    """
    Inspect each calibrator for nearly-flat isotonic output that may reduce
    ranking quality vs raw cosine similarity.

    A nearly-flat transform outputs fewer unique values than the number of
    input points, compressing score differences and degrading ranking.

    Parameters
    ----------
    calibrators : {channel: Calibrator} – fitted calibrators.
    sample_scores : optional 1-D array of raw scores in [0, 1] to probe.
        Defaults to a uniform grid of ``n_sample`` points.
    n_sample : number of probe points (default 200).

    Returns
    -------
    Dict[channel → {method, n_unique_outputs, fraction_unique,
                    min_output, max_output, output_range,
                    nearly_flat, warning}]
    """
    if sample_scores is None:
        sample_scores = np.linspace(0.0, 1.0, n_sample)
    sample_scores = np.asarray(sample_scores, dtype=np.float64)

    diagnostics: Dict[str, dict] = {}
    for ch, cal in calibrators.items():
        if cal is None:
            continue
        try:
            out = cal.transform(sample_scores)
        except Exception as exc:
            diagnostics[ch] = {"error": str(exc)}
            continue

        n_unique = int(np.unique(out).shape[0])
        frac_unique = n_unique / len(sample_scores)
        out_min = float(out.min())
        out_max = float(out.max())
        out_range = out_max - out_min
        # Flag as nearly flat if fewer than 10% unique values OR range < 0.05.
        nearly_flat = (frac_unique < 0.10) or (out_range < 0.05)
        warning = (
            f"Nearly-flat isotonic: {n_unique}/{len(sample_scores)} unique outputs, "
            f"range [{out_min:.3f}, {out_max:.3f}]. "
            "Ranking may not improve over raw cosine. "
            "Consider recalibrating or using Platt scaling."
        ) if nearly_flat else None

        diagnostics[ch] = {
            "method": getattr(cal, "method", "unknown"),
            "n_unique_outputs": n_unique,
            "n_sample": len(sample_scores),
            "fraction_unique": round(frac_unique, 4),
            "min_output": round(out_min, 4),
            "max_output": round(out_max, 4),
            "output_range": round(out_range, 4),
            "nearly_flat": nearly_flat,
            "warning": warning,
        }
        if nearly_flat:
            logger.warning("Calibration flatness [%s]: %s", ch, warning)

    return diagnostics

def estimate_unknown_threshold(
    known_oof_results: List[QueryResult],
    unknown_oof_results: List[QueryResult],
    criterion: str = "equal_far_frr",
) -> Tuple[float, Dict]:
    """
    Estimate an unknown-rejection threshold from OOF predictions.

    Parameters
    ----------
    known_oof_results : QueryResult list with fusion weights already applied
        (fused_score populated).  Each entry is a gallery OOF pseudo-query
        whose identity **was present** in the OOF rest-gallery fold
        (``identity_in_oof_gallery=True`` or pre-filtered by caller).
        The top-1 fused_score is the system's best-match confidence for a
        known individual.
    unknown_oof_results : QueryResult list with fusion weights already applied.
        Each entry is a simulated unknown trial produced by identity removal
        (``identity_in_oof_gallery=False``).  The top-1 fused_score is the
        best wrong-identity confidence for a truly unknown individual.
    criterion : "equal_far_frr" – select threshold at the equal-error-rate
        (FAR ≈ FRR).

    Returns
    -------
    threshold : float
    diagnostics : dict  – includes criterion, n_known, n_unknown,
        far_at_threshold, frr_at_threshold, provenance.

    Raises
    ------
    CalibrationSupportError
        Hard error if either distribution has no support (empty list or all
        QueryResults have empty ranked_identities).  No silent 0.5 fallback.
    """
    known_scores = [
        qr.ranked_identities[0].fused_score
        for qr in known_oof_results
        if qr.ranked_identities
    ]
    unknown_scores = [
        qr.ranked_identities[0].fused_score
        for qr in unknown_oof_results
        if qr.ranked_identities
    ]

    if not known_scores:
        raise CalibrationSupportError(
            "estimate_unknown_threshold: known distribution has no support "
            "(no known OOF queries with ranked identities). "
            "Cannot estimate a meaningful threshold. "
            "Ensure gallery has multiple sessions per identity and that "
            "fusion weights have been applied before calling this function."
        )
    if not unknown_scores:
        raise CalibrationSupportError(
            "estimate_unknown_threshold: unknown distribution has no support "
            "(no simulated unknown trials with ranked identities). "
            "Cannot estimate a meaningful threshold. "
            "Call simulate_gallery_unknown_scores to generate unknown trials."
        )

    known_arr = np.array(known_scores, dtype=np.float64)
    unk_arr = np.array(unknown_scores, dtype=np.float64)

    # Grid search for equal-error-rate threshold.
    grid = np.linspace(
        min(known_arr.min(), unk_arr.min()),
        max(known_arr.max(), unk_arr.max()),
        num=500,
    )
    best_t = float(np.median(np.concatenate([known_arr, unk_arr])))
    best_diff = np.inf

    for t in grid:
        far = float((unk_arr >= t).mean())    # false accept rate (unknowns accepted)
        frr = float((known_arr < t).mean())   # false reject rate (knowns rejected)
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_t = float(t)

    far_at_t = float((unk_arr >= best_t).mean())
    frr_at_t = float((known_arr < best_t).mean())

    diagnostics = {
        "criterion": criterion,
        "threshold": round(best_t, 6),
        "n_known": len(known_scores),
        "n_unknown": len(unknown_scores),
        "far_at_threshold": round(far_at_t, 6),
        "frr_at_threshold": round(frr_at_t, 6),
        "provenance": (
            "gallery_oof_known_identity_present + "
            "gallery_identity_removal_simulated_unknown"
        ),
    }
    logger.info(
        "Unknown threshold: t=%.4f FAR=%.3f FRR=%.3f (n_known=%d n_unknown=%d)",
        best_t, far_at_t, frr_at_t, len(known_scores), len(unknown_scores),
    )
    return best_t, diagnostics
