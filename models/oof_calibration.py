# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Grouped out-of-fold (OOF) calibration on gallery/reference artifacts.

Algorithm
---------
Calibration data must not use the fixed query/probe images.  Instead we
treat each gallery session as one "fold": images from that session become
pseudo-queries and the gallery images from all **other** sessions form the
temporary reference.

For each pseudo-query image q:
  - Exclude self (same image_id / same crop_id) from reference.
  - Exclude all reference crops that share q's session_id.
  - Build per-channel, identity-level scores:
      body channels  – one query body vector vs. one reference body vector.
                       Score(q, identity I) = max cosine over (q_body × I_body).
      ear channels   – 0..2 query ear vectors vs. 0..2 reference ear vectors.
                       Score(q, identity I) = max cosine over all (q_ear_j × I_ear_k).
  - Include the positive identity and a deterministic set of hard/top-K
    negatives (top-scoring wrong-identity references).

Fold support contract
---------------------
Each fold is included only when it contributes both at least one positive
and at least one negative pair. Folds that cannot provide this (e.g. an
identity with only one session in the gallery) are recorded in the
diagnostic but excluded from the fitting pool.

Aggregate support contract
--------------------------
After collecting OOF pairs, the following are **hard errors** (raise
CalibrationSupportError):
  - A channel is enabled but the aggregated positive count is zero.
  - A channel is enabled but the aggregated negative count is zero.
  - No session fold yielded any positive pairs at all.

If the aggregate positive count is below MIN_POSITIVE_PAIRS_FOR_ISOTONIC,
the Calibrator falls back to Platt scaling and documents the reason.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How many hard negatives to collect per pseudo-query per channel.
# Deterministic: take the top HARD_NEG_K wrong-identity references by score.
HARD_NEG_K: int = 5

# Minimum gallery images in the rest-set for a fold to be scored at all.
MIN_REST_SIZE: int = 2


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

class CalibrationSupportError(RuntimeError):
    """Hard error: a channel lacks the minimum support required to calibrate."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FoldDiagnostic:
    session_id: str
    n_pseudo_queries: int
    n_pos_pairs: int
    n_neg_pairs: int
    included: bool
    exclusion_reason: Optional[str] = None


@dataclass
class ChannelOOFResult:
    channel: str
    scores: List[float] = field(default_factory=list)
    labels: List[float] = field(default_factory=list)
    fold_diagnostics: List[FoldDiagnostic] = field(default_factory=list)
    n_skipped_folds: int = 0
    n_included_folds: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_identity_max(
    query_rows: np.ndarray,
    ref_matrix: np.ndarray,
    ref_ids: np.ndarray,
    unique_ref_ids: List[str],
) -> Dict[str, float]:
    """
    Compute per-identity maximum cosine similarity.

    Parameters
    ----------
    query_rows : shape (n_q, D)  – L2-normalised query embeddings.
    ref_matrix : shape (n_r, D)  – L2-normalised reference embeddings.
    ref_ids    : shape (n_r,)    – individual_id for each row in ref_matrix.
    unique_ref_ids : ordered list of unique individual_ids in ref_matrix.

    Returns
    -------
    {individual_id: max_cosine_similarity}
    """
    if query_rows.shape[0] == 0 or ref_matrix.shape[0] == 0:
        return {rid: 0.0 for rid in unique_ref_ids}

    # sim[i, j] = cosine(query_i, ref_j)  (embeddings are L2-normalised)
    sim = query_rows @ ref_matrix.T  # (n_q, n_r)

    identity_scores: Dict[str, float] = {}
    for rid in unique_ref_ids:
        mask = ref_ids == rid
        if not mask.any():
            identity_scores[rid] = 0.0
            continue
        # Max over (query crops × reference crops of this identity)
        identity_scores[rid] = float(sim[:, mask].max())

    return identity_scores


def _build_image_to_embedding_rows(
    descriptor_mapping: pd.DataFrame,
    gallery_image_ids: List[str],
    embedding_matrix: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Return {image_id: embedding_rows_ndarray} for images in gallery_image_ids.

    If an image has no crops in the mapping its key is absent.
    """
    image_to_rows: Dict[str, List[int]] = {}
    for row in descriptor_mapping.itertuples(index=False):
        iid = str(row.image_id)
        if iid in gallery_image_ids:
            image_to_rows.setdefault(iid, []).append(int(row.embedding_row))

    result: Dict[str, np.ndarray] = {}
    for iid, rows in image_to_rows.items():
        result[iid] = embedding_matrix[np.array(rows, dtype=int)]
    return result


# ---------------------------------------------------------------------------
# Core OOF scoring
# ---------------------------------------------------------------------------

def compute_oof_scores(
    gallery_image_df: pd.DataFrame,
    descriptor_mappings: Dict[str, pd.DataFrame],
    embedding_matrices: Dict[str, np.ndarray],
    hard_neg_k: int = HARD_NEG_K,
) -> Dict[str, ChannelOOFResult]:
    """
    Compute grouped OOF (score, label) pairs for each descriptor channel.

    Parameters
    ----------
    gallery_image_df : DataFrame with columns:
        image_id, individual_id, session_id.
        Rows must be the gallery split only (probe rows must be excluded).
    descriptor_mappings : {channel: DataFrame} with normalised descriptor
        mapping schema (embedding_row, image_id, individual_id, crop_kind).
        Body channels have crop_kind == 'body'; ear channels have 'ear'.
    embedding_matrices : {channel: np.ndarray} L2-normalised.
    hard_neg_k : number of hard/top negatives per pseudo-query per channel.

    Returns
    -------
    {channel: ChannelOOFResult}
    """
    required_cols = {"image_id", "individual_id", "session_id"}
    missing = required_cols - set(gallery_image_df.columns)
    if missing:
        raise ValueError(
            f"gallery_image_df is missing required columns: {sorted(missing)}"
        )

    # Work with string IDs consistently.
    gallery_image_df = gallery_image_df.copy()
    gallery_image_df["image_id"] = gallery_image_df["image_id"].astype(str)
    gallery_image_df["individual_id"] = gallery_image_df["individual_id"].astype(str)
    gallery_image_df["session_id"] = gallery_image_df["session_id"].astype(str)

    gallery_image_ids_set = set(gallery_image_df["image_id"])
    channels = list(descriptor_mappings.keys())

    if not channels:
        raise CalibrationSupportError("No descriptor channels provided.")

    # Validate that each channel is enabled (has mappings and embeddings).
    for ch in channels:
        if ch not in embedding_matrices or embedding_matrices[ch] is None:
            raise CalibrationSupportError(
                f"Channel '{ch}' is enabled but has no embedding matrix. "
                "Either disable the channel or provide its embeddings."
            )
        dm = descriptor_mappings[ch]
        if dm is None or dm.empty:
            raise CalibrationSupportError(
                f"Channel '{ch}' is enabled but has an empty descriptor mapping."
            )

    # Restrict each mapping to gallery images only and verify.
    gallery_mappings: Dict[str, pd.DataFrame] = {}
    gallery_emb_rows: Dict[str, Dict[str, np.ndarray]] = {}  # ch -> {image_id: rows}

    for ch, dm in descriptor_mappings.items():
        dm = dm.copy()
        dm["image_id"] = dm["image_id"].astype(str)
        dm["individual_id"] = dm["individual_id"].astype(str)
        gm = dm[dm["image_id"].isin(gallery_image_ids_set)].reset_index(drop=True)
        gallery_mappings[ch] = gm
        gallery_image_ids_list = list(gallery_image_ids_set)
        gallery_emb_rows[ch] = _build_image_to_embedding_rows(
            gm, gallery_image_ids_list, embedding_matrices[ch]
        )

    # Session-based OOF folds.
    sessions = sorted(gallery_image_df["session_id"].unique())
    logger.info(
        "OOF calibration: %d gallery images, %d sessions, %d channels.",
        len(gallery_image_df),
        len(sessions),
        len(channels),
    )

    results: Dict[str, ChannelOOFResult] = {ch: ChannelOOFResult(channel=ch) for ch in channels}

    for session_id in sessions:
        # -----------------------------------------------------------------
        # Partition into pseudo-query and rest.
        # -----------------------------------------------------------------
        pseudo_query_df = gallery_image_df[
            gallery_image_df["session_id"] == session_id
        ]
        rest_df = gallery_image_df[
            gallery_image_df["session_id"] != session_id
        ]

        if len(rest_df) < MIN_REST_SIZE:
            for ch in channels:
                results[ch].fold_diagnostics.append(
                    FoldDiagnostic(
                        session_id=session_id,
                        n_pseudo_queries=len(pseudo_query_df),
                        n_pos_pairs=0,
                        n_neg_pairs=0,
                        included=False,
                        exclusion_reason="rest_too_small",
                    )
                )
                results[ch].n_skipped_folds += 1
            continue

        rest_image_ids = set(rest_df["image_id"])
        rest_identity_arr = np.array(rest_df["individual_id"].tolist())
        unique_rest_ids = sorted(rest_df["individual_id"].unique())

        # -----------------------------------------------------------------
        # Score per channel.
        # -----------------------------------------------------------------
        for ch in channels:
            ch_result = results[ch]
            emb_matrix = embedding_matrices[ch]
            rest_gm = gallery_mappings[ch][
                gallery_mappings[ch]["image_id"].isin(rest_image_ids)
            ]

            if rest_gm.empty:
                ch_result.fold_diagnostics.append(
                    FoldDiagnostic(
                        session_id=session_id,
                        n_pseudo_queries=len(pseudo_query_df),
                        n_pos_pairs=0,
                        n_neg_pairs=0,
                        included=False,
                        exclusion_reason=f"channel_{ch}_no_rest_crops",
                    )
                )
                ch_result.n_skipped_folds += 1
                continue

            # Build reference per-identity embedding matrices for this fold.
            rest_image_ids_in_ch = set(rest_gm["image_id"].astype(str))
            rest_df_in_ch = rest_df[rest_df["image_id"].isin(rest_image_ids_in_ch)]
            unique_rest_ids_in_ch = sorted(rest_df_in_ch["individual_id"].unique())

            if not unique_rest_ids_in_ch:
                ch_result.fold_diagnostics.append(
                    FoldDiagnostic(
                        session_id=session_id,
                        n_pseudo_queries=len(pseudo_query_df),
                        n_pos_pairs=0,
                        n_neg_pairs=0,
                        included=False,
                        exclusion_reason=f"channel_{ch}_no_rest_identities",
                    )
                )
                ch_result.n_skipped_folds += 1
                continue

            rest_emb_rows = rest_gm["embedding_row"].astype(int).to_numpy()
            rest_emb_matrix = emb_matrix[rest_emb_rows]  # (n_r_crops, D)
            rest_emb_ids = rest_gm["image_id"].astype(str).to_numpy()

            # Map each rest crop row to its individual_id via image_id.
            image_to_indiv = dict(
                zip(rest_df["image_id"].astype(str), rest_df["individual_id"].astype(str))
            )
            rest_indiv_arr = np.array(
                [image_to_indiv.get(iid, "") for iid in rest_emb_ids]
            )

            # Score pseudo-query images.
            fold_scores: List[float] = []
            fold_labels: List[float] = []

            for _, q_row in pseudo_query_df.iterrows():
                q_image_id = str(q_row["image_id"])
                q_indiv = str(q_row["individual_id"])

                # Get query embedding rows for this channel.
                q_emb_rows = gallery_emb_rows[ch].get(q_image_id)
                if q_emb_rows is None or len(q_emb_rows) == 0:
                    continue

                # Compute identity-level scores.
                identity_scores = _cosine_identity_max(
                    q_emb_rows,
                    rest_emb_matrix,
                    rest_indiv_arr,
                    unique_rest_ids_in_ch,
                )

                # Positive score.
                if q_indiv not in identity_scores:
                    continue
                pos_score = identity_scores[q_indiv]

                # Negative scores (other identities).
                neg_id_scores = [
                    (rid, s) for rid, s in identity_scores.items() if rid != q_indiv
                ]
                if not neg_id_scores:
                    continue

                # Deterministic hard negatives: top HARD_NEG_K by score.
                neg_id_scores.sort(key=lambda x: x[1], reverse=True)
                hard_negatives = neg_id_scores[:hard_neg_k]

                fold_scores.append(pos_score)
                fold_labels.append(1.0)
                for _, neg_score in hard_negatives:
                    fold_scores.append(neg_score)
                    fold_labels.append(0.0)

            n_pos = int(sum(lbl == 1.0 for lbl in fold_labels))
            n_neg = int(sum(lbl == 0.0 for lbl in fold_labels))

            if n_pos < 1 or n_neg < 1:
                ch_result.fold_diagnostics.append(
                    FoldDiagnostic(
                        session_id=session_id,
                        n_pseudo_queries=len(pseudo_query_df),
                        n_pos_pairs=n_pos,
                        n_neg_pairs=n_neg,
                        included=False,
                        exclusion_reason=(
                            "no_positive_pairs" if n_pos < 1 else "no_negative_pairs"
                        ),
                    )
                )
                ch_result.n_skipped_folds += 1
            else:
                ch_result.scores.extend(fold_scores)
                ch_result.labels.extend(fold_labels)
                ch_result.fold_diagnostics.append(
                    FoldDiagnostic(
                        session_id=session_id,
                        n_pseudo_queries=len(pseudo_query_df),
                        n_pos_pairs=n_pos,
                        n_neg_pairs=n_neg,
                        included=True,
                    )
                )
                ch_result.n_included_folds += 1

    # -----------------------------------------------------------------
    # Aggregate support validation (hard errors for enabled channels).
    # -----------------------------------------------------------------
    for ch, ch_result in results.items():
        n_pos_total = int(sum(lbl == 1.0 for lbl in ch_result.labels))
        n_neg_total = int(sum(lbl == 0.0 for lbl in ch_result.labels))
        n_total = len(ch_result.labels)

        if n_total == 0:
            raise CalibrationSupportError(
                f"Channel '{ch}': no OOF pairs collected across {len(sessions)} sessions. "
                f"Skipped folds: {ch_result.n_skipped_folds}. "
                "This is a hard error — check that the gallery has multiple sessions per identity."
            )
        if n_pos_total == 0:
            raise CalibrationSupportError(
                f"Channel '{ch}': aggregate OOF positive count is zero "
                f"({n_total} total pairs, all negative). "
                "Cannot fit a meaningful calibrator."
            )
        if n_neg_total == 0:
            raise CalibrationSupportError(
                f"Channel '{ch}': aggregate OOF negative count is zero "
                f"({n_total} total pairs, all positive). "
                "Cannot fit a meaningful calibrator."
            )

        logger.info(
            "OOF channel '%s': %d pairs (%d pos, %d neg) from %d folds "
            "(%d included, %d skipped).",
            ch,
            n_total,
            n_pos_total,
            n_neg_total,
            len(sessions),
            ch_result.n_included_folds,
            ch_result.n_skipped_folds,
        )

    return results


# ---------------------------------------------------------------------------
# ROC AUC helper (no sklearn dependency)
# ---------------------------------------------------------------------------

def roc_auc(scores: List[float], labels: List[float]) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s)[::-1]
    y_sorted = y[order]
    tpr = np.cumsum(y_sorted) / y_sorted.sum()
    fpr_inc = (1 - y_sorted) / (len(y_sorted) - y_sorted.sum())
    return float(np.dot(tpr, fpr_inc))
