# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Projection-head metric adaptation for ear_miewid embeddings.

Architecture
------------
  Input  : L2-normalised MiewID embedding (dim=2152 by default).
  Head   : optional Dropout → Linear(in_dim, out_dim) → L2 normalise.
  Output : L2-normalised embedding in `out_dim`-dimensional space.

Training supports:
  - Batch-hard triplet loss.
  - Supervised contrastive loss (SupCon).
  - Optional ArcFace classification head (training only; not exported).

Data utilities:
  - build_inner_split: deterministic session-based train/val split.
  - PxKDataset / PxKSampler: identity-balanced P×K mini-batches.

Retrieval metrics:
  - retrieval_map_top1: inner-validation mAP/top1 without model downloads.

Adoption gate:
  - AdoptionGate: decides whether projected metrics improve enough to adopt.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    Small L2-normalised projection head.

    Input: (B, in_dim) – already L2-normalised embeddings.
    Output: (B, out_dim) – L2-normalised projected embeddings.
    """

    def __init__(
        self,
        in_dim: int = 2152,
        out_dim: int = 512,
        dropout: float = 0.0,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        layers: list[nn.Module] = []
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        if hidden_dim is not None and hidden_dim > 0:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=True))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(hidden_dim, out_dim, bias=True))
        else:
            linear = nn.Linear(in_dim, out_dim, bias=True)
            if in_dim == out_dim:
                with torch.no_grad():
                    linear.weight.copy_(torch.eye(in_dim))
                    linear.bias.zero_()
            layers.append(linear)

        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised projected embeddings."""
        out = self.proj(x)
        return F.normalize(out, p=2, dim=1)

    def parameter_fingerprint(self) -> str:
        """SHA256 of the serialised state-dict for checkpoint checks."""
        buf = bytearray()
        for name, param in sorted(self.state_dict().items()):
            buf += name.encode()
            buf += param.detach().cpu().numpy().tobytes()
        return hashlib.sha256(bytes(buf)).hexdigest()


# ---------------------------------------------------------------------------
# ArcFace classification head (training only)
# ---------------------------------------------------------------------------

class ArcFaceHead(nn.Module):
    """
    ArcFace angular margin classification head.

    Only used during training; never exported.

    Parameters
    ----------
    in_dim  : Dimension of the L2-normalised embedding.
    n_classes : Number of training identities.
    s         : Scaling factor (typical: 30-64).
    m         : Additive angular margin in radians (typical: 0.5).
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        s: float = 32.0,
        m: float = 0.50,
    ) -> None:
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(n_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)

        self._cos_m = math.cos(m)
        self._sin_m = math.sin(m)
        self._th = math.cos(math.pi - m)
        self._mm = math.sin(math.pi - m) * m

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return classification logits with ArcFace margin."""
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cos_theta = F.linear(x, w_norm)            # (B, C)
        sin_theta = torch.clamp(
            torch.sqrt(torch.clamp(1.0 - cos_theta ** 2, min=0.0)), min=1e-7
        )
        phi = cos_theta * self._cos_m - sin_theta * self._sin_m

        # Stable: only apply margin to the target class
        phi = torch.where(cos_theta > self._th, phi, cos_theta - self._mm)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        return output * self.s


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
    squared: bool = False,
) -> torch.Tensor:
    """
    Batch-hard triplet loss (Hermans et al., 2017).

    For each anchor, mines the hardest positive (max distance same-class)
    and the hardest negative (min distance different-class) in the batch.

    Parameters
    ----------
    embeddings : (B, D) L2-normalised.
    labels     : (B,) integer class labels.
    margin     : Triplet margin.
    squared    : Use squared Euclidean if True, Euclidean otherwise.

    Returns
    -------
    Scalar mean triplet loss.
    """
    # Pairwise squared Euclidean distances
    # For L2-normalised vectors: dist^2 = 2 - 2*dot
    dot = torch.mm(embeddings, embeddings.t())
    sq_dist = torch.clamp(2.0 - 2.0 * dot, min=0.0)
    if not squared:
        dist = torch.sqrt(sq_dist + 1e-12)
    else:
        dist = sq_dist

    # Masks
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)   # (B, B)
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    pos_mask = labels_eq & ~eye   # same class, not self
    neg_mask = ~labels_eq         # different class

    # Batch-hard positive: hardest (farthest) positive per anchor
    # Fill non-positive positions with 0 so max works correctly
    pos_dist = dist * pos_mask.float()
    hardest_pos, _ = pos_dist.max(dim=1)   # (B,)

    # Batch-hard negative: hardest (closest) negative per anchor
    # Fill non-negative positions with large value
    max_dist = dist.max().detach() + 1.0
    neg_dist = dist + max_dist * (~neg_mask).float()
    hardest_neg, _ = neg_dist.min(dim=1)   # (B,)

    triplet_loss = F.relu(hardest_pos - hardest_neg + margin)

    # Only average over anchors that have at least one positive
    has_pos = pos_mask.any(dim=1)
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    return triplet_loss[has_pos].mean()


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised contrastive loss (Khosla et al., 2020).

    Parameters
    ----------
    embeddings  : (B, D) L2-normalised.
    labels      : (B,) integer class labels.
    temperature : Logit scaling temperature.

    Returns
    -------
    Scalar mean SupCon loss.
    """
    B = embeddings.shape[0]
    device = embeddings.device

    # Pairwise cosine similarities
    sim = torch.mm(embeddings, embeddings.t()) / temperature  # (B, B)

    # Remove self-similarity from denominator
    eye = torch.eye(B, dtype=torch.bool, device=device)
    sim_no_self = sim.masked_fill(eye, float("-inf"))

    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
    pos_mask = labels_eq & ~eye

    # Anchors that have at least one positive in the batch
    has_pos = pos_mask.any(dim=1)
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    log_denom = torch.logsumexp(sim_no_self, dim=1)   # (B,)

    # For each anchor, mean log-prob over positives
    n_pos = pos_mask.sum(dim=1).float().clamp(min=1.0)
    pos_sum = (sim * pos_mask.float()).sum(dim=1) - (pos_mask.float() * temperature * 0.0)
    # Proper SupCon: sum of (log_softmax at each positive)
    # = sum_j[ pos_mask[i,j] * (sim[i,j] - log_denom[i]) ]
    log_prob = (sim - log_denom.unsqueeze(1)) * pos_mask.float()
    loss_per_anchor = -log_prob.sum(dim=1) / n_pos

    return loss_per_anchor[has_pos].mean()


# ---------------------------------------------------------------------------
# Inner train/val split (session-based, no leakage)
# ---------------------------------------------------------------------------

@dataclass
class InnerSplitResult:
    """Result of session-based inner train/val split for one experiment."""

    # image_ids (not crop_ids) for train and val sets
    train_image_ids: List[str]
    val_image_ids: List[str]

    # Identities that have only one session (train-only)
    train_only_identities: List[str]

    # Mapping: identity_id → session reserved for validation
    val_sessions_by_identity: Dict[str, str]

    # Mapping: identity_id → train sessions
    train_sessions_by_identity: Dict[str, List[str]]

    # Set of forbidden image_ids (probe + held-out) for hard-fail check
    forbidden_image_ids: frozenset = field(default_factory=frozenset)


def build_inner_split(
    ref_mapping: pd.DataFrame,
    splits_df: pd.DataFrame,
    seed: int = 42,
) -> InnerSplitResult:
    """
    Deterministic session-based inner train/val split for gallery identities.

    Rules
    -----
    1. Select only rows with split == 'gallery'.
    2. For each gallery identity with ≥2 distinct sessions, sort session IDs
       lexicographically and reserve the last one for inner validation.
    3. Identities with exactly 1 session are train-only (reported).
    4. Hard-fail if any selected image_id appears in the splits parquet with
       split ∈ {probe, held_out_probe, held_out_gallery}.
    5. Hard-fail if any individual_id from held_out_gallery or held_out_probe
       appears among the training identities.

    Parameters
    ----------
    ref_mapping : Reference descriptor mapping with image_id and individual_id.
    splits_df   : Full splits parquet (all splits).
    seed        : Unused (determinism achieved through lexicographic sort),
                  kept in signature for manifest documentation.

    Returns
    -------
    InnerSplitResult
    """
    # Step 1: merge mapping with splits to get split/session_id per image
    merge_cols = ["image_id", "split", "session_id", "individual_id"]
    avail_cols = [c for c in merge_cols if c in splits_df.columns]
    image_meta = (
        splits_df[avail_cols]
        .drop_duplicates(subset="image_id")
        .rename(columns={"individual_id": "split_individual_id"})
    )
    merged = ref_mapping.merge(image_meta, on="image_id", how="left")
    merged["split"] = merged["split"].fillna("unknown")

    # Forbidden image_ids
    forbidden_splits = {"probe", "held_out_probe", "held_out_gallery"}
    forbidden_image_ids = frozenset(
        splits_df.loc[splits_df["split"].isin(forbidden_splits), "image_id"].astype(str)
    )
    # Forbidden individual_ids (held-out identities must never enter training)
    forbidden_ind_splits = {"held_out_gallery", "held_out_probe"}
    forbidden_individual_ids = frozenset(
        splits_df.loc[
            splits_df["split"].isin(forbidden_ind_splits), "individual_id"
        ]
        .dropna()
        .astype(str)
    )

    # Step 2: filter to gallery rows only
    gallery_rows = merged[merged["split"] == "gallery"].copy()
    if gallery_rows.empty:
        raise ValueError("No gallery rows found in the merged mapping+splits.")

    # Hard fail: no forbidden image_ids in training pool
    overlap = set(gallery_rows["image_id"].astype(str)) & forbidden_image_ids
    if overlap:
        raise RuntimeError(
            f"SAFETY VIOLATION: {len(overlap)} gallery-split image_ids are also "
            f"in probe/held-out splits. This should never happen. "
            f"First offenders: {sorted(overlap)[:5]}"
        )

    # Hard fail: no forbidden individual_ids
    gallery_individuals = set(gallery_rows["individual_id"].astype(str))
    ind_overlap = gallery_individuals & forbidden_individual_ids
    if ind_overlap:
        raise RuntimeError(
            f"SAFETY VIOLATION: {len(ind_overlap)} held-out individual_ids "
            f"appear in the gallery training pool: {sorted(ind_overlap)[:5]}"
        )

    # Ensure session_id is available
    if "session_id" not in gallery_rows.columns or gallery_rows["session_id"].isna().all():
        raise ValueError(
            "session_id is missing or all-NaN in merged gallery rows. "
            "Check that the splits parquet contains session_id."
        )
    gallery_rows["session_id"] = gallery_rows["session_id"].fillna("session_unknown")

    # Step 3: per-identity session assignment
    train_image_ids: List[str] = []
    val_image_ids: List[str] = []
    train_only_identities: List[str] = []
    val_sessions_by_identity: Dict[str, str] = {}
    train_sessions_by_identity: Dict[str, List[str]] = {}

    for ind_id, grp in gallery_rows.groupby("individual_id"):
        ind_id = str(ind_id)
        sessions = sorted(grp["session_id"].unique().tolist())

        if len(sessions) >= 2:
            val_session = sessions[-1]     # lexicographically last → deterministic
            train_sessions = sessions[:-1]
        else:
            val_session = None
            train_sessions = sessions
            train_only_identities.append(ind_id)

        val_sessions_by_identity[ind_id] = val_session  # type: ignore[assignment]
        train_sessions_by_identity[ind_id] = train_sessions

        is_val = grp["session_id"] == val_session if val_session else pd.Series(False, index=grp.index)
        train_image_ids.extend(grp.loc[~is_val, "image_id"].astype(str).tolist())
        val_image_ids.extend(grp.loc[is_val, "image_id"].astype(str).tolist())

    if train_only_identities:
        logger.warning(
            "%d training-only identities (single session, no val fold): %s",
            len(train_only_identities),
            train_only_identities,
        )

    logger.info(
        "Inner split: %d train images, %d val images, %d val identities, "
        "%d train-only identities",
        len(train_image_ids),
        len(val_image_ids),
        len(val_sessions_by_identity) - len(train_only_identities),
        len(train_only_identities),
    )

    return InnerSplitResult(
        train_image_ids=train_image_ids,
        val_image_ids=val_image_ids,
        train_only_identities=train_only_identities,
        val_sessions_by_identity=val_sessions_by_identity,
        train_sessions_by_identity=train_sessions_by_identity,
        forbidden_image_ids=forbidden_image_ids,
    )


# ---------------------------------------------------------------------------
# P×K dataset and sampler
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingRecord:
    """A single embedding row with its identity label."""

    image_id: str
    individual_id: str
    label: int              # integer class index
    emb_row: int            # row index into the embedding matrix


class PxKDataset(Dataset):
    """
    Dataset of (embedding_row, int_label) pairs for metric learning.

    Parameters
    ----------
    embedding_matrix : (N, D) numpy array of L2-normalised embeddings.
    records          : List of EmbeddingRecord objects.
    """

    def __init__(
        self,
        embedding_matrix: np.ndarray,
        records: List[EmbeddingRecord],
    ) -> None:
        self.matrix = embedding_matrix
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        rec = self.records[idx]
        emb = torch.from_numpy(self.matrix[rec.emb_row]).float()
        return emb, rec.label, rec.individual_id


class PxKSampler(Sampler):
    """
    Identity-balanced P×K batch sampler.

    Each batch contains exactly P identities with K samples each.
    Identities are sampled without replacement per epoch; samples within
    each identity are sampled with replacement when there are fewer than K.

    Parameters
    ----------
    label_to_indices : dict mapping integer label → list of dataset indices.
    P                : Number of identities per batch.
    K                : Number of samples per identity per batch.
    seed             : Random seed for reproducibility.
    drop_last        : Drop the last incomplete batch if True.
    """

    def __init__(
        self,
        label_to_indices: Dict[int, List[int]],
        P: int,
        K: int,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        self.label_to_indices = label_to_indices
        self.P = P
        self.K = K
        self.seed = seed
        self.drop_last = drop_last
        self.labels = list(label_to_indices.keys())
        if len(self.labels) < P:
            raise ValueError(
                f"PxKSampler requires at least P={P} distinct identities, "
                f"but only {len(self.labels)} are available."
            )
        n_batches = len(self.labels) // P
        self._len = n_batches      # number of batches (for batch_sampler interface)

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return self._len

    def __iter__(self):
        rng = random.Random(self.seed)
        shuffled_labels = self.labels.copy()
        rng.shuffle(shuffled_labels)

        n_batches = len(shuffled_labels) // self.P
        for b in range(n_batches):
            batch_labels = shuffled_labels[b * self.P: (b + 1) * self.P]
            batch_indices: List[int] = []
            for lbl in batch_labels:
                pool = self.label_to_indices[lbl]
                if len(pool) >= self.K:
                    chosen = rng.sample(pool, self.K)
                else:
                    chosen = [rng.choice(pool) for _ in range(self.K)]
                batch_indices.extend(chosen)
            yield batch_indices     # yield full batch list (batch_sampler contract)


def build_px_k_dataset(
    embedding_matrix: np.ndarray,
    ref_mapping: pd.DataFrame,
    image_ids: List[str],
) -> Tuple[PxKDataset, Dict[int, List[int]], Dict[str, int]]:
    """
    Build a PxKDataset and supporting structures from a list of image_ids.

    Only rows in ref_mapping whose image_id appears in image_ids are included.
    Multiple crops per image are included (all rows matching the image_id).

    Returns
    -------
    dataset          : PxKDataset
    label_to_indices : dict mapping int label → list of dataset indices
    identity_to_label: dict mapping individual_id → int label
    """
    image_id_set = set(image_ids)
    rows = ref_mapping[ref_mapping["image_id"].astype(str).isin(image_id_set)].copy()

    if rows.empty:
        raise ValueError("No rows in ref_mapping match the provided image_ids.")

    identities = sorted(rows["individual_id"].astype(str).unique().tolist())
    identity_to_label: Dict[str, int] = {iid: i for i, iid in enumerate(identities)}

    records: List[EmbeddingRecord] = []
    for _, row in rows.iterrows():
        ind_id = str(row["individual_id"])
        records.append(
            EmbeddingRecord(
                image_id=str(row["image_id"]),
                individual_id=ind_id,
                label=identity_to_label[ind_id],
                emb_row=int(row["embedding_row"]),
            )
        )

    label_to_indices: Dict[int, List[int]] = {}
    for idx, rec in enumerate(records):
        label_to_indices.setdefault(rec.label, []).append(idx)

    dataset = PxKDataset(embedding_matrix, records)
    return dataset, label_to_indices, identity_to_label


# ---------------------------------------------------------------------------
# Retrieval metrics (inner validation)
# ---------------------------------------------------------------------------

def retrieval_map_top1(
    query_embs: np.ndarray,
    query_labels: np.ndarray,
    ref_embs: np.ndarray,
    ref_labels: np.ndarray,
    top_k: int = 20,
) -> Tuple[float, float]:
    """
    Compute identity-level retrieval mAP and top-1 accuracy.

    Parameters
    ----------
    query_embs  : (Q, D) L2-normalised query embeddings.
    query_labels: (Q,) integer identity labels for queries.
    ref_embs    : (R, D) L2-normalised reference embeddings.
    ref_labels  : (R,) integer identity labels for references.
    top_k       : Number of top candidates to consider for AP.

    Returns
    -------
    (mAP, top1) floats in [0, 1].
    """
    # Cosine similarity = dot product for L2-normalised vectors
    sim = query_embs @ ref_embs.T           # (Q, R)
    ranked = np.argsort(-sim, axis=1)       # (Q, R) descending

    aps: List[float] = []
    top1_correct = 0

    for qi in range(len(query_embs)):
        q_label = query_labels[qi]
        ranked_labels = ref_labels[ranked[qi, :top_k]]

        # Top-1
        if len(ranked_labels) > 0 and ranked_labels[0] == q_label:
            top1_correct += 1

        # AP: precision at each relevant hit position
        hits = (ranked_labels == q_label).astype(float)
        n_rel = hits.sum()
        if n_rel == 0:
            # Query has no match in reference; skip for mAP
            continue
        cum_hits = np.cumsum(hits)
        positions = np.arange(1, len(hits) + 1)
        precision_at_k = cum_hits / positions
        ap = (precision_at_k * hits).sum() / n_rel
        aps.append(float(ap))

    mAP = float(np.mean(aps)) if aps else 0.0
    top1 = float(top1_correct / len(query_embs)) if len(query_embs) > 0 else 0.0
    return mAP, top1


def compute_projected_embeddings(
    model: ProjectionHead,
    embedding_matrix: np.ndarray,
    batch_size: int = 512,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Apply projection head to a full embedding matrix, return L2-norm'd result.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        for start in range(0, len(embedding_matrix), batch_size):
            chunk = torch.from_numpy(
                embedding_matrix[start: start + batch_size]
            ).float().to(device)
            out = model(chunk)
            rows.append(out.cpu().numpy())

    return np.concatenate(rows, axis=0).astype(np.float32)


def get_labels_for_image_ids(
    image_ids: List[str],
    ref_mapping: pd.DataFrame,
    identity_to_label: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (row_indices, int_labels) arrays for a set of image_ids.

    Multiple crops per image are included.
    """
    image_id_set = set(image_ids)
    rows = ref_mapping[ref_mapping["image_id"].astype(str).isin(image_id_set)]
    row_indices = rows["embedding_row"].to_numpy(dtype=np.int64)
    labels = np.array(
        [identity_to_label[str(r)] for r in rows["individual_id"]], dtype=np.int64
    )
    return row_indices, labels


# ---------------------------------------------------------------------------
# Adoption gate
# ---------------------------------------------------------------------------

@dataclass
class AdoptionGateResult:
    """Decision from the adoption gate."""

    adopted: bool
    baseline_map: float
    projected_map: float
    baseline_top1: float
    projected_top1: float
    map_delta: float
    top1_delta: float
    min_map_delta: float
    min_top1_delta: float
    reason: str


def adoption_gate(
    baseline_map: float,
    projected_map: float,
    baseline_top1: float,
    projected_top1: float,
    min_map_delta: float = 0.005,
    min_top1_delta: float = 0.005,
    instability_threshold: float = -0.05,
) -> AdoptionGateResult:
    """
    Accept the projected model if metrics improve by documented minima.

    Reject if:
      - mAP improvement is below min_map_delta, AND top1 improvement is
        below min_top1_delta.
      - Either mAP or top1 dropped by more than |instability_threshold|.

    Parameters
    ----------
    baseline_map        : mAP before projection.
    projected_map       : mAP after projection.
    baseline_top1       : Top-1 before projection.
    projected_top1      : Top-1 after projection.
    min_map_delta       : Minimum mAP improvement for adoption.
    min_top1_delta      : Minimum top-1 improvement for adoption (secondary).
    instability_threshold: Maximum allowed drop (negative value is a drop).

    Returns
    -------
    AdoptionGateResult with adopted=True/False and diagnostics.
    """
    map_delta = projected_map - baseline_map
    top1_delta = projected_top1 - baseline_top1

    # Instability check
    if map_delta < instability_threshold or top1_delta < instability_threshold:
        return AdoptionGateResult(
            adopted=False,
            baseline_map=baseline_map,
            projected_map=projected_map,
            baseline_top1=baseline_top1,
            projected_top1=projected_top1,
            map_delta=map_delta,
            top1_delta=top1_delta,
            min_map_delta=min_map_delta,
            min_top1_delta=min_top1_delta,
            reason=(
                f"REJECTED (instability): map_delta={map_delta:+.4f} or "
                f"top1_delta={top1_delta:+.4f} below instability_threshold="
                f"{instability_threshold:+.4f}"
            ),
        )

    # Improvement check: either mAP or top1 must meet its threshold
    map_ok = map_delta >= min_map_delta
    top1_ok = top1_delta >= min_top1_delta

    if map_ok or top1_ok:
        return AdoptionGateResult(
            adopted=True,
            baseline_map=baseline_map,
            projected_map=projected_map,
            baseline_top1=baseline_top1,
            projected_top1=projected_top1,
            map_delta=map_delta,
            top1_delta=top1_delta,
            min_map_delta=min_map_delta,
            min_top1_delta=min_top1_delta,
            reason=(
                f"ADOPTED: map_delta={map_delta:+.4f} (min={min_map_delta:+.4f}), "
                f"top1_delta={top1_delta:+.4f} (min={min_top1_delta:+.4f})"
            ),
        )

    return AdoptionGateResult(
        adopted=False,
        baseline_map=baseline_map,
        projected_map=projected_map,
        baseline_top1=baseline_top1,
        projected_top1=projected_top1,
        map_delta=map_delta,
        top1_delta=top1_delta,
        min_map_delta=min_map_delta,
        min_top1_delta=min_top1_delta,
        reason=(
            f"REJECTED (insufficient improvement): "
            f"map_delta={map_delta:+.4f} < min={min_map_delta:+.4f}, "
            f"top1_delta={top1_delta:+.4f} < min={min_top1_delta:+.4f}"
        ),
    )


# ---------------------------------------------------------------------------
# Backbone unfreezing scaffold (NOT IMPLEMENTED)
# ---------------------------------------------------------------------------
# This section documents the interface for a future Stage 2: unfreezing the
# final MiewID backbone stage after the projection head has converged.
#
# CONTRACT (when implemented):
#   - Stage 1 MUST complete and the projection head checkpoint adopted before
#     Stage 2 is considered.
#   - Only the LAST encoder block(s) should be unfrozen; embedding backbone
#     weights must be loaded from the published conservationxlabs/miewid-msv3
#     checkpoint, not from any synthetic or untrained initialisation.
#   - The full fine-tuned model must be evaluated end-to-end (image → embedding
#     → projection head) before any adoption decision.
#   - The interface MUST preserve the same safety contracts:
#       * No probe/held-out images in training.
#       * Session-disjoint inner split.
#       * Adoption gate on held-out-free inner validation.
#   - This file intentionally does NOT implement a fake or placeholder backbone
#     fine-tune. Do not add code here unless you have real backbone weights
#     and intend real training.
#
# SUGGESTED CONFIG (for future use, not active):
#
# BACKBONE_FINETUNE_CONFIG = {
#     "stage": 2,
#     "backbone_model_id": "conservationxlabs/miewid-msv3",
#     "unfreeze_last_n_blocks": 1,
#     "stage1_checkpoint_required": True,
#     "lr_backbone": 1e-5,
#     "lr_projection": 1e-4,
#     "epochs": 20,
#     "early_stop_patience": 5,
#     "note": "NOT IMPLEMENTED – scaffold only. Implement after Stage 1 adoption.",
# }
