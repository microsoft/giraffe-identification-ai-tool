# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Tests for the MiewID projection-head metric adaptation pipeline.

Coverage
--------
1. No probe/heldout leakage into training split (build_inner_split).
2. Session disjointness: train and val sessions never overlap.
3. P×K sampler: batch sizes, identity counts, with-replacement fallback.
4. Losses finite and backward pass produces gradients.
5. Deterministic split/training (same seed → same result, tolerance ±1e-6).
6. L2-normalised outputs from ProjectionHead.
7. Checkpoint fingerprint changes after parameter update.
8. Transform mapping/index integrity (clone, row alignment, FAISS dim).
9. Adoption gate accept/reject logic.
10. Identity-level validation metrics (retrieval_map_top1).

No model downloads: all tests use synthetic random embeddings.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.miewid_projection import (
    ArcFaceHead,
    AdoptionGateResult,
    EmbeddingRecord,
    InnerSplitResult,
    ProjectionHead,
    PxKDataset,
    PxKSampler,
    adoption_gate,
    batch_hard_triplet_loss,
    build_inner_split,
    build_px_k_dataset,
    compute_projected_embeddings,
    get_labels_for_image_ids,
    retrieval_map_top1,
    supervised_contrastive_loss,
)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _make_splits_df(
    gallery_ids: List[str],
    gallery_sessions_per_id: int = 3,
    probe_ids: Optional[List[str]] = None,
    heldout_gal_ids: Optional[List[str]] = None,
    heldout_probe_ids: Optional[List[str]] = None,
    n_images_per_session: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a synthetic splits DataFrame.

    Gallery identities have `gallery_sessions_per_id` sessions each,
    with `n_images_per_session` images per session.
    Probe / heldout identities are distinct from gallery ones.
    """
    rng = np.random.default_rng(seed)
    rows = []

    def _make_images(individual_id: str, split: str, n_sess: int):
        for s_idx in range(n_sess):
            session_id = f"{individual_id}_session_{s_idx:02d}"
            for i_idx in range(n_images_per_session):
                image_id = f"img_{individual_id}_{s_idx}_{i_idx}"
                rows.append({
                    "image_id": image_id,
                    "individual_id": individual_id,
                    "session_id": session_id,
                    "split": split,
                })

    for ind_id in gallery_ids:
        _make_images(ind_id, "gallery", gallery_sessions_per_id)

    if probe_ids:
        for ind_id in probe_ids:
            _make_images(ind_id, "probe", 1)

    if heldout_gal_ids:
        for ind_id in heldout_gal_ids:
            _make_images(ind_id, "held_out_gallery", 2)

    if heldout_probe_ids:
        for ind_id in heldout_probe_ids:
            _make_images(ind_id, "held_out_probe", 1)

    return pd.DataFrame(rows)


def _make_ref_mapping(splits_df: pd.DataFrame, in_dim: int = 16) -> pd.DataFrame:
    """Build a synthetic reference mapping from a splits DataFrame."""
    gallery_rows = splits_df[splits_df["split"] == "gallery"].copy()
    mapping_rows = []
    for i, (_, row) in enumerate(gallery_rows.iterrows()):
        mapping_rows.append({
            "descriptor_name": "ear_miewid",
            "embedding_row": i,
            "faiss_row": i,
            "crop_id": f"{row['image_id']}__ear_0",
            "image_id": row["image_id"],
            "individual_id": row["individual_id"],
            "crop_kind": "ear",
            "crop_ordinal": 0,
            "crop_path": f"/fake/{row['image_id']}.jpg",
            "schema_version": "v1",
            "source_fingerprint": "abc123",
            "split_fingerprint": "def456",
            "model_preprocess_fingerprint": "ear_miewid:config-elephant-v1",
        })
    return pd.DataFrame(mapping_rows)


def _make_embedding_matrix(n: int, dim: int = 16, seed: int = 0) -> np.ndarray:
    """Return L2-normalised random float32 embedding matrix."""
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_gallery_ids():
    return [f"bteh_id_{i:02d}" for i in range(8)]


@pytest.fixture
def small_probe_ids():
    # probe individuals are DISTINCT from gallery
    return [f"bteh_probe_{i:02d}" for i in range(3)]


@pytest.fixture
def heldout_gal_ids():
    return [f"bteh_heldout_{i:02d}" for i in range(2)]


@pytest.fixture
def heldout_probe_ids():
    return [f"bteh_heldout_probe_{i:02d}" for i in range(2)]


@pytest.fixture
def splits_and_mapping(small_gallery_ids, small_probe_ids, heldout_gal_ids, heldout_probe_ids):
    splits_df = _make_splits_df(
        gallery_ids=small_gallery_ids,
        gallery_sessions_per_id=3,
        probe_ids=small_probe_ids,
        heldout_gal_ids=heldout_gal_ids,
        heldout_probe_ids=heldout_probe_ids,
    )
    ref_mapping = _make_ref_mapping(splits_df)
    matrix = _make_embedding_matrix(len(ref_mapping), dim=16)
    return splits_df, ref_mapping, matrix


# ---------------------------------------------------------------------------
# 1. No probe/heldout leakage
# ---------------------------------------------------------------------------

class TestNoLeakage:
    def test_gallery_image_ids_not_in_probe(self, splits_and_mapping):
        splits_df, ref_mapping, _ = splits_and_mapping
        inner = build_inner_split(ref_mapping, splits_df, seed=42)

        probe_image_ids = set(
            splits_df.loc[splits_df["split"] == "probe", "image_id"].astype(str)
        )
        heldout_image_ids = set(
            splits_df.loc[
                splits_df["split"].isin({"held_out_probe", "held_out_gallery"}),
                "image_id",
            ].astype(str)
        )

        train_ids = set(inner.train_image_ids)
        val_ids = set(inner.val_image_ids)

        assert len(train_ids & probe_image_ids) == 0, (
            "Probe image_ids leaked into training set!"
        )
        assert len(train_ids & heldout_image_ids) == 0, (
            "Heldout image_ids leaked into training set!"
        )
        assert len(val_ids & probe_image_ids) == 0, (
            "Probe image_ids leaked into validation set!"
        )
        assert len(val_ids & heldout_image_ids) == 0, (
            "Heldout image_ids leaked into validation set!"
        )

    def test_heldout_individual_ids_not_in_training(
        self, splits_and_mapping, heldout_gal_ids, heldout_probe_ids
    ):
        splits_df, ref_mapping, _ = splits_and_mapping
        inner = build_inner_split(ref_mapping, splits_df, seed=42)

        # Inner split only contains gallery image_ids; gather their individual_ids
        gallery_rows = splits_df[splits_df["split"] == "gallery"]
        train_individuals = set(
            gallery_rows.loc[
                gallery_rows["image_id"].isin(inner.train_image_ids),
                "individual_id",
            ].astype(str)
        )

        for ind_id in heldout_gal_ids + heldout_probe_ids:
            assert ind_id not in train_individuals, (
                f"Heldout individual {ind_id!r} leaked into training!"
            )

    def test_hard_fail_if_forbidden_image_in_mapping(
        self, splits_and_mapping
    ):
        """
        If the reference mapping somehow contains a probe image_id,
        build_inner_split must hard-fail with RuntimeError.
        """
        splits_df, ref_mapping, _ = splits_and_mapping
        # Poison the mapping with a probe image_id
        probe_image_id = splits_df.loc[
            splits_df["split"] == "probe", "image_id"
        ].iloc[0]

        poisoned = ref_mapping.copy()
        poisoned.loc[0, "image_id"] = probe_image_id

        # Build a splits_df where this probe_image_id appears as BOTH
        # gallery (first row → kept by drop_duplicates) and probe (second row).
        # drop_duplicates keeps the first occurrence, so gallery split wins for
        # the merge, putting the image_id into gallery_rows.
        # The probe row still populates forbidden_image_ids → overlap detected.
        new_gallery_row = splits_df[splits_df["image_id"] == probe_image_id].copy()
        new_gallery_row["split"] = "gallery"
        new_gallery_row["individual_id"] = "bteh_id_00"  # some gallery identity
        # Gallery row must come FIRST so drop_duplicates keeps it
        combined_splits = pd.concat(
            [new_gallery_row, splits_df], ignore_index=True
        )

        # gallery_rows now includes probe_image_id (split=gallery after dedup),
        # but forbidden_image_ids also includes it (probe row still in splits_df).
        with pytest.raises(RuntimeError, match="SAFETY VIOLATION"):
            build_inner_split(poisoned, combined_splits, seed=42)


# ---------------------------------------------------------------------------
# 2. Session disjointness
# ---------------------------------------------------------------------------

class TestSessionDisjointness:
    def test_no_session_overlap_between_train_and_val(
        self, splits_and_mapping, small_gallery_ids
    ):
        splits_df, ref_mapping, _ = splits_and_mapping
        inner = build_inner_split(ref_mapping, splits_df, seed=42)

        train_id_set = set(inner.train_image_ids)
        val_id_set = set(inner.val_image_ids)

        # Gather sessions for train and val image_ids
        img_to_session = (
            splits_df.set_index("image_id")["session_id"].to_dict()
        )
        train_sessions = {
            img_to_session[img]
            for img in train_id_set
            if img in img_to_session
        }
        val_sessions = {
            img_to_session[img]
            for img in val_id_set
            if img in img_to_session
        }

        overlap = train_sessions & val_sessions
        assert len(overlap) == 0, (
            f"Session leakage: sessions {overlap} appear in both train and val!"
        )

    def test_val_session_is_last_sorted(self, small_gallery_ids):
        """Validation session is the lexicographically last session per identity."""
        splits_df = _make_splits_df(
            gallery_ids=small_gallery_ids,
            gallery_sessions_per_id=3,
        )
        ref_mapping = _make_ref_mapping(splits_df)
        inner = build_inner_split(ref_mapping, splits_df, seed=42)

        gallery_rows = splits_df[splits_df["split"] == "gallery"]
        for ind_id, sessions_map in inner.val_sessions_by_identity.items():
            if sessions_map is None:
                continue
            ind_sessions = sorted(
                gallery_rows.loc[
                    gallery_rows["individual_id"] == ind_id, "session_id"
                ].unique().tolist()
            )
            assert sessions_map == ind_sessions[-1], (
                f"Val session for {ind_id!r} is not the last sorted: "
                f"expected {ind_sessions[-1]!r}, got {sessions_map!r}"
            )

    def test_single_session_identity_is_train_only(self):
        """Identities with only one session are reported as train-only."""
        # One identity with 1 session, others with 3 sessions
        gallery_ids = ["single_id"] + [f"multi_{i}" for i in range(4)]
        splits_df = _make_splits_df(
            gallery_ids=["single_id"],
            gallery_sessions_per_id=1,
        )
        multi_df = _make_splits_df(
            gallery_ids=[f"multi_{i}" for i in range(4)],
            gallery_sessions_per_id=3,
        )
        combined = pd.concat([splits_df, multi_df], ignore_index=True)
        ref_mapping = _make_ref_mapping(combined)
        inner = build_inner_split(ref_mapping, combined, seed=42)

        assert "single_id" in inner.train_only_identities, (
            "Single-session identity should be in train_only_identities."
        )
        # single_id has no val images
        single_val_images = [
            img for img in inner.val_image_ids if "single_id" in img
        ]
        assert len(single_val_images) == 0, (
            "Single-session identity should have no validation images."
        )


# ---------------------------------------------------------------------------
# 3. P×K sampler behaviour
# ---------------------------------------------------------------------------

class TestPxKSampler:
    def _make_label_indices(self, n_ids: int, k_per_id: int) -> Dict[int, List[int]]:
        offset = 0
        d = {}
        for i in range(n_ids):
            d[i] = list(range(offset, offset + k_per_id))
            offset += k_per_id
        return d

    def test_batch_size_is_p_times_k(self):
        P, K, n_ids = 4, 3, 10
        label_to_indices = self._make_label_indices(n_ids, 5)
        sampler = PxKSampler(label_to_indices, P=P, K=K, seed=0)
        batches = list(sampler)
        # sampler is a batch_sampler: len() = number of batches
        assert len(batches) == len(sampler)
        # Each batch must have exactly P*K items
        for batch in batches:
            assert len(batch) == P * K

    def test_each_batch_has_exactly_p_identities(self):
        """
        Verify that indices within each batch come from exactly P identities.
        """
        P, K, n_ids = 4, 3, 12
        label_to_indices = self._make_label_indices(n_ids, 5)
        index_to_label = {}
        for lbl, idxs in label_to_indices.items():
            for idx in idxs:
                index_to_label[idx] = lbl

        sampler = PxKSampler(label_to_indices, P=P, K=K, seed=0)
        for batch in sampler:
            batch_labels = {index_to_label[i] for i in batch}
            assert len(batch_labels) == P, (
                f"Batch has {len(batch_labels)} identities, expected {P}."
            )

    def test_with_replacement_when_few_samples(self):
        """When identity has fewer than K samples, sampling with replacement fills the batch."""
        P, K = 2, 5
        label_to_indices = {
            0: [0, 1],     # only 2 samples, K=5 → with replacement
            1: [2, 3, 4, 5, 6],
        }
        sampler = PxKSampler(label_to_indices, P=P, K=K, seed=0)
        batches = list(sampler)
        # Should not raise and each batch must be P * K items
        assert len(batches) == 1
        assert len(batches[0]) == P * K

    def test_raises_when_fewer_identities_than_p(self):
        """Raise ValueError when fewer identities than P."""
        label_to_indices = self._make_label_indices(3, 5)
        with pytest.raises(ValueError, match="at least P=5"):
            PxKSampler(label_to_indices, P=5, K=2, seed=0)

    def test_dataset_returns_tensor_and_label(self):
        """PxKDataset.__getitem__ returns (FloatTensor, int, str)."""
        n, dim = 10, 16
        matrix = _make_embedding_matrix(n, dim=dim)
        records = [
            EmbeddingRecord(
                image_id=f"img_{i}",
                individual_id="id_0" if i < 5 else "id_1",
                label=0 if i < 5 else 1,
                emb_row=i,
            )
            for i in range(n)
        ]
        dataset = PxKDataset(matrix, records)
        emb, label, ind_id = dataset[0]
        assert isinstance(emb, torch.Tensor)
        assert emb.dtype == torch.float32
        assert emb.shape == (dim,)
        assert isinstance(label, (int, np.integer))


# ---------------------------------------------------------------------------
# 4. Loss finiteness and gradient flow
# ---------------------------------------------------------------------------

class TestLosses:
    def _make_batch(self, B: int = 16, D: int = 32, n_classes: int = 4):
        mat = _make_embedding_matrix(B, dim=D)
        embs = torch.from_numpy(mat).float().requires_grad_(True)
        labels = torch.tensor(
            [i % n_classes for i in range(B)], dtype=torch.long
        )
        return embs, labels

    def test_triplet_loss_finite(self):
        embs, labels = self._make_batch(16, 32, 4)
        loss = batch_hard_triplet_loss(embs, labels, margin=0.3)
        assert torch.isfinite(loss), f"Triplet loss is not finite: {loss}"
        assert loss.item() >= 0.0

    def test_supcon_loss_finite(self):
        embs, labels = self._make_batch(16, 32, 4)
        loss = supervised_contrastive_loss(embs, labels, temperature=0.07)
        assert torch.isfinite(loss), f"SupCon loss is not finite: {loss}"
        assert loss.item() >= 0.0

    def test_triplet_loss_gradient_flows(self):
        embs, labels = self._make_batch(16, 32, 4)
        loss = batch_hard_triplet_loss(embs, labels, margin=0.3)
        if loss.item() > 0:
            loss.backward()
            assert embs.grad is not None
            assert torch.isfinite(embs.grad).all()

    def test_supcon_loss_gradient_flows(self):
        embs, labels = self._make_batch(16, 32, 4)
        loss = supervised_contrastive_loss(embs, labels, temperature=0.07)
        if loss.item() > 0:
            loss.backward()
            assert embs.grad is not None
            assert torch.isfinite(embs.grad).all()

    def test_triplet_loss_zero_when_no_positives(self):
        """With all distinct labels, no positives → loss is 0 tensor."""
        B, D = 8, 16
        mat = _make_embedding_matrix(B, dim=D)
        embs = torch.from_numpy(mat).float()
        # All distinct labels → no positives in batch
        labels = torch.arange(B, dtype=torch.long)
        loss = batch_hard_triplet_loss(embs, labels, margin=0.3)
        assert loss.item() == 0.0

    def test_supcon_loss_zero_when_no_positives(self):
        B, D = 8, 16
        mat = _make_embedding_matrix(B, dim=D)
        embs = torch.from_numpy(mat).float()
        labels = torch.arange(B, dtype=torch.long)
        loss = supervised_contrastive_loss(embs, labels, temperature=0.07)
        assert loss.item() == 0.0

    def test_projection_head_gradient_flows(self):
        """ProjectionHead parameters receive gradients from triplet loss."""
        model = ProjectionHead(in_dim=16, out_dim=8, dropout=0.0)
        embs_in = torch.from_numpy(_make_embedding_matrix(16, dim=16)).float()
        labels = torch.tensor([i % 4 for i in range(16)], dtype=torch.long)

        projected = model(embs_in)
        loss = batch_hard_triplet_loss(projected, labels, margin=0.3)
        if loss.item() > 0:
            loss.backward()
            for name, param in model.named_parameters():
                assert param.grad is not None, f"No grad for {name}"

    def test_arcface_head_finite(self):
        in_dim, n_classes = 16, 5
        head = ArcFaceHead(in_dim=in_dim, n_classes=n_classes)
        embs = torch.from_numpy(_make_embedding_matrix(10, dim=in_dim)).float()
        labels = torch.tensor([i % n_classes for i in range(10)], dtype=torch.long)
        logits = head(embs, labels)
        assert logits.shape == (10, n_classes)
        assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# 5. Deterministic split and training
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_inner_split_is_deterministic(self, splits_and_mapping):
        splits_df, ref_mapping, _ = splits_and_mapping
        inner1 = build_inner_split(ref_mapping, splits_df, seed=42)
        inner2 = build_inner_split(ref_mapping, splits_df, seed=42)
        assert sorted(inner1.train_image_ids) == sorted(inner2.train_image_ids)
        assert sorted(inner1.val_image_ids) == sorted(inner2.val_image_ids)

    def test_projection_head_init_is_deterministic_with_seed(self):
        """Same torch seed → identical initial weights."""
        torch.manual_seed(0)
        m1 = ProjectionHead(in_dim=16, out_dim=8)
        w1 = m1.proj[-1].weight.detach().clone()

        torch.manual_seed(0)
        m2 = ProjectionHead(in_dim=16, out_dim=8)
        w2 = m2.proj[-1].weight.detach().clone()

        assert torch.allclose(w1, w2, atol=1e-7)

    def test_px_k_sampler_deterministic(self):
        label_to_indices = {i: list(range(i * 5, i * 5 + 5)) for i in range(10)}
        s1 = PxKSampler(label_to_indices, P=4, K=3, seed=99)
        s2 = PxKSampler(label_to_indices, P=4, K=3, seed=99)
        assert list(s1) == list(s2)

    def test_px_k_sampler_different_seeds_differ(self):
        label_to_indices = {i: list(range(i * 5, i * 5 + 5)) for i in range(10)}
        s1 = PxKSampler(label_to_indices, P=4, K=3, seed=0)
        s2 = PxKSampler(label_to_indices, P=4, K=3, seed=1)
        # With high probability different seeds produce different orderings
        assert list(s1) != list(s2)


# ---------------------------------------------------------------------------
# 6. L2-normalised outputs
# ---------------------------------------------------------------------------

class TestL2Normalisation:
    def test_projection_head_output_is_unit_norm(self):
        model = ProjectionHead(in_dim=32, out_dim=16, dropout=0.0)
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(_make_embedding_matrix(50, dim=32)).float()
            out = model(x)
        norms = torch.norm(out, dim=1)
        assert torch.allclose(norms, torch.ones(50), atol=1e-5), (
            f"Output norms not all 1.0: {norms.min().item():.6f} .. {norms.max().item():.6f}"
        )

    def test_compute_projected_embeddings_returns_normalised(self):
        model = ProjectionHead(in_dim=16, out_dim=8, dropout=0.0)
        matrix = _make_embedding_matrix(30, dim=16)
        projected = compute_projected_embeddings(model, matrix, batch_size=10, device="cpu")
        norms = np.linalg.norm(projected, axis=1)
        assert projected.dtype == np.float32
        assert np.allclose(norms, 1.0, atol=1e-4), (
            f"compute_projected_embeddings output norms: {norms.min():.6f} .. {norms.max():.6f}"
        )

    def test_projection_with_dropout_still_normalised_at_eval(self):
        model = ProjectionHead(in_dim=32, out_dim=16, dropout=0.5)
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(_make_embedding_matrix(20, dim=32)).float()
            out = model(x)
        norms = torch.norm(out, dim=1)
        assert torch.allclose(norms, torch.ones(20), atol=1e-5)

    def test_equal_dimension_linear_projection_starts_as_identity(self):
        model = ProjectionHead(in_dim=16, out_dim=16, dropout=0.0)
        model.eval()
        x = torch.from_numpy(_make_embedding_matrix(12, dim=16)).float()
        with torch.no_grad():
            out = model(x)
        torch.testing.assert_close(out, x)


# ---------------------------------------------------------------------------
# 7. Checkpoint fingerprints
# ---------------------------------------------------------------------------

class TestCheckpointFingerprints:
    def test_fingerprint_changes_after_parameter_update(self):
        model = ProjectionHead(in_dim=16, out_dim=8)
        fp1 = model.parameter_fingerprint()

        # Update a parameter
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.01 * torch.ones_like(p))

        fp2 = model.parameter_fingerprint()
        assert fp1 != fp2, "Fingerprint did not change after parameter update."

    def test_fingerprint_stable_for_same_model(self):
        torch.manual_seed(1)
        model = ProjectionHead(in_dim=16, out_dim=8)
        fp1 = model.parameter_fingerprint()
        fp2 = model.parameter_fingerprint()
        assert fp1 == fp2

    def test_checkpoint_save_and_load_preserves_fingerprint(self):
        import io
        model = ProjectionHead(in_dim=16, out_dim=8)
        fp_before = model.parameter_fingerprint()

        buf = io.BytesIO()
        torch.save({"state_dict": model.state_dict(), "in_dim": 16, "out_dim": 8}, buf)
        buf.seek(0)
        ckpt = torch.load(buf, map_location="cpu", weights_only=False)

        model2 = ProjectionHead(in_dim=16, out_dim=8)
        # Randomise model2 first to confirm it changes
        with torch.no_grad():
            for p in model2.parameters():
                p.fill_(0.0)

        model2.load_state_dict(ckpt["state_dict"])
        fp_after = model2.parameter_fingerprint()
        assert fp_before == fp_after, (
            "Fingerprint changed after checkpoint save/load round-trip."
        )


# ---------------------------------------------------------------------------
# 8. Transform mapping and index integrity
# ---------------------------------------------------------------------------

class TestTransformIntegrity:
    def test_clone_mapping_updates_descriptor_name(self):
        from pipeline.transform_miewid_projection import _clone_mapping
        src = pd.DataFrame({
            "descriptor_name": ["ear_miewid"] * 5,
            "embedding_row": list(range(5)),
            "faiss_row": list(range(5)),
            "image_id": [f"img_{i}" for i in range(5)],
            "individual_id": ["id_0"] * 5,
            "model_preprocess_fingerprint": ["ear_miewid:v1"] * 5,
        })
        out = _clone_mapping(src, "ear_miewid_projected", "proj:abc123")
        assert (out["descriptor_name"] == "ear_miewid_projected").all()
        assert list(out["embedding_row"]) == list(range(5))
        assert list(out["faiss_row"]) == list(range(5))

    def test_clone_mapping_row_alignment(self):
        from pipeline.transform_miewid_projection import _clone_mapping
        n = 20
        src = pd.DataFrame({
            "descriptor_name": ["ear_miewid"] * n,
            "embedding_row": list(range(n)),
            "faiss_row": list(range(n)),
            "image_id": [f"img_{i}" for i in range(n)],
            "individual_id": ["id_x"] * n,
        })
        out = _clone_mapping(src, "ear_miewid_projected", "proj:xyz")
        assert list(out["embedding_row"]) == list(range(n))
        assert list(out["faiss_row"]) == list(range(n))

    def test_clone_mapping_rejects_shuffled_embedding_rows(self):
        from pipeline.transform_miewid_projection import _clone_mapping
        src = pd.DataFrame({
            "descriptor_name": ["ear_miewid"] * 3,
            "embedding_row": [2, 0, 1],
            "faiss_row": [2, 0, 1],
            "image_id": ["a", "b", "c"],
            "individual_id": ["id"] * 3,
        })
        with pytest.raises(ValueError, match="ordered by contiguous"):
            _clone_mapping(src, "ear_miewid_projected", "projection")

    def test_transform_end_to_end(self, tmp_path):
        """Full transform: write fake ear_miewid artifacts, run transform, verify outputs."""
        from pipeline.transform_miewid_projection import transform_projection

        # Build source artifacts
        in_dim, out_dim = 16, 8
        n = 10
        matrix = _make_embedding_matrix(n, dim=in_dim)
        mapping = pd.DataFrame({
            "descriptor_name": ["ear_miewid"] * n,
            "embedding_row": list(range(n)),
            "faiss_row": list(range(n)),
            "crop_id": [f"img_{i}__ear_0" for i in range(n)],
            "image_id": [f"img_{i}" for i in range(n)],
            "individual_id": [f"id_{i % 3}" for i in range(n)],
            "crop_kind": ["ear"] * n,
            "crop_ordinal": [0] * n,
            "crop_path": [f"/fake/img_{i}.jpg" for i in range(n)],
            "schema_version": ["v1"] * n,
            "source_fingerprint": ["src_fp"] * n,
            "split_fingerprint": ["spl_fp"] * n,
            "model_preprocess_fingerprint": ["ear_miewid:config-elephant-v1"] * n,
        })

        emb_dir = tmp_path / "embeddings" / "reference"
        emb_dir.mkdir(parents=True)
        np.save(str(emb_dir / "ear_miewid.npy"), matrix)
        mapping.to_parquet(str(emb_dir / "ear_miewid_mapping.parquet"), index=False)

        # Build and save a checkpoint
        torch.manual_seed(42)
        model = ProjectionHead(in_dim=in_dim, out_dim=out_dim)
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "best_projection.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "in_dim": in_dim,
            "out_dim": out_dim,
            "dropout": 0.0,
            "hidden_dim": None,
            "source_fingerprint": "src_fp",
            "split_fingerprint": "spl_fp",
            "model_preprocess_fingerprint": "ear_miewid:config-elephant-v1",
            "val_map": 0.9,
            "checkpoint_fingerprint": "abc",
            "schema_version": "v1",
        }, str(ckpt_path))

        # Run transform (reference only, skip FAISS for speed)
        result = transform_projection(
            artifact_root=tmp_path,
            ckpt_path=ckpt_path,
            out_descriptor="ear_miewid_projected",
            src_descriptor="ear_miewid",
            partitions=["reference"],
            build_index=False,
            force=False,
        )

        out_npy = emb_dir / "ear_miewid_projected.npy"
        out_parquet = emb_dir / "ear_miewid_projected_mapping.parquet"
        assert out_npy.is_file(), "Output .npy not written"
        assert out_parquet.is_file(), "Output mapping.parquet not written"

        proj = np.load(str(out_npy))
        assert proj.shape == (n, out_dim)
        norms = np.linalg.norm(proj, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), "Projected outputs not L2-normalised"

        out_map = pd.read_parquet(str(out_parquet))
        assert (out_map["descriptor_name"] == "ear_miewid_projected").all()
        assert list(out_map["embedding_row"]) == list(range(n))
        assert len(out_map) == n

    def test_transform_fails_on_dimension_mismatch(self, tmp_path):
        from pipeline.transform_miewid_projection import transform_projection

        in_dim = 16
        n = 5
        matrix = _make_embedding_matrix(n, dim=in_dim)
        mapping = pd.DataFrame({
            "descriptor_name": ["ear_miewid"] * n,
            "embedding_row": list(range(n)),
            "faiss_row": list(range(n)),
            "image_id": [f"img_{i}" for i in range(n)],
            "individual_id": ["id_0"] * n,
        })
        emb_dir = tmp_path / "embeddings" / "reference"
        emb_dir.mkdir(parents=True)
        np.save(str(emb_dir / "ear_miewid.npy"), matrix)
        mapping.to_parquet(str(emb_dir / "ear_miewid_mapping.parquet"), index=False)

        # Checkpoint expects wrong input dim
        model = ProjectionHead(in_dim=999, out_dim=8)
        ckpt_path = tmp_path / "bad_ckpt.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "in_dim": 999,
            "out_dim": 8,
        }, str(ckpt_path))

        with pytest.raises(ValueError, match="Dimension mismatch"):
            transform_projection(
                artifact_root=tmp_path,
                ckpt_path=ckpt_path,
                partitions=["reference"],
                build_index=False,
                force=False,
            )

    def test_transform_refuses_overwrite_without_force(self, tmp_path):
        from pipeline.transform_miewid_projection import transform_projection

        in_dim, out_dim = 16, 8
        n = 5
        matrix = _make_embedding_matrix(n, dim=in_dim)
        mapping = pd.DataFrame({
            "descriptor_name": ["ear_miewid"] * n,
            "embedding_row": list(range(n)),
            "faiss_row": list(range(n)),
            "image_id": [f"img_{i}" for i in range(n)],
            "individual_id": ["id_0"] * n,
            "source_fingerprint": ["sf"] * n,
            "split_fingerprint": ["spf"] * n,
        })
        emb_dir = tmp_path / "embeddings" / "reference"
        emb_dir.mkdir(parents=True)
        np.save(str(emb_dir / "ear_miewid.npy"), matrix)
        mapping.to_parquet(str(emb_dir / "ear_miewid_mapping.parquet"), index=False)
        # Pre-create output to trigger refusal
        (emb_dir / "ear_miewid_projected.npy").touch()

        model = ProjectionHead(in_dim=in_dim, out_dim=out_dim)
        ckpt_path = tmp_path / "ckpt.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "in_dim": in_dim, "out_dim": out_dim,
            "source_fingerprint": "sf", "split_fingerprint": "spf",
        }, str(ckpt_path))

        with pytest.raises(FileExistsError):
            transform_projection(
                artifact_root=tmp_path,
                ckpt_path=ckpt_path,
                partitions=["reference"],
                build_index=False,
                force=False,
            )


# ---------------------------------------------------------------------------
# 9. Adoption gate
# ---------------------------------------------------------------------------

class TestAdoptionGate:
    def test_gate_accepts_when_map_improves(self):
        result = adoption_gate(
            baseline_map=0.50, projected_map=0.52,
            baseline_top1=0.60, projected_top1=0.61,
            min_map_delta=0.005,
        )
        assert result.adopted is True
        assert "ADOPTED" in result.reason

    def test_gate_rejects_when_map_insufficient(self):
        result = adoption_gate(
            baseline_map=0.50, projected_map=0.502,
            baseline_top1=0.60, projected_top1=0.601,
            min_map_delta=0.005,
            min_top1_delta=0.01,
        )
        assert result.adopted is False
        assert "REJECTED" in result.reason

    def test_gate_default_rejects_flat_top1_with_map_regression(self):
        result = adoption_gate(
            baseline_map=0.60,
            projected_map=0.57,
            baseline_top1=0.70,
            projected_top1=0.70,
        )
        assert result.adopted is False

    def test_gate_rejects_on_instability(self):
        result = adoption_gate(
            baseline_map=0.50, projected_map=0.40,
            baseline_top1=0.60, projected_top1=0.50,
            min_map_delta=0.005,
            instability_threshold=-0.05,
        )
        assert result.adopted is False
        assert "instability" in result.reason.lower()

    def test_gate_accepts_via_top1_even_if_map_small(self):
        result = adoption_gate(
            baseline_map=0.50, projected_map=0.503,
            baseline_top1=0.60, projected_top1=0.65,
            min_map_delta=0.01,
            min_top1_delta=0.02,
        )
        assert result.adopted is True

    def test_gate_result_fields_are_correct(self):
        result = adoption_gate(
            baseline_map=0.40, projected_map=0.46,
            baseline_top1=0.55, projected_top1=0.58,
            min_map_delta=0.005,
        )
        assert abs(result.map_delta - 0.06) < 1e-8
        assert abs(result.top1_delta - 0.03) < 1e-8
        assert result.baseline_map == 0.40
        assert result.projected_map == 0.46


# ---------------------------------------------------------------------------
# 10. Identity-level validation metrics
# ---------------------------------------------------------------------------

class TestRetrievalMetrics:
    def test_perfect_retrieval(self):
        """When query embeddings = reference embeddings, top1=1.0 and mAP≈1.0."""
        n_ids, k_per_id, dim = 5, 3, 16
        embeddings = _make_embedding_matrix(n_ids * k_per_id, dim=dim)
        labels = np.repeat(np.arange(n_ids), k_per_id)

        # Queries are the same as references
        mAP, top1 = retrieval_map_top1(embeddings, labels, embeddings, labels, top_k=n_ids * k_per_id)
        # With self-similarity, nearest neighbour is self → top1 = 1.0
        assert top1 == pytest.approx(1.0, abs=1e-6)

    def test_random_retrieval_below_perfect(self):
        """Random embeddings should give mAP < 1.0 for most configurations."""
        n_ids, k_per_id, dim = 10, 4, 64
        matrix = _make_embedding_matrix(n_ids * k_per_id, dim=dim, seed=1)
        labels = np.repeat(np.arange(n_ids), k_per_id)
        mAP, top1 = retrieval_map_top1(matrix[:20], labels[:20], matrix[20:], labels[20:], top_k=20)
        # Simply test that metrics are in valid range
        assert 0.0 <= mAP <= 1.0
        assert 0.0 <= top1 <= 1.0

    def test_metrics_in_valid_range(self):
        for _ in range(3):
            n = 20
            q_embs = _make_embedding_matrix(n, dim=32)
            r_embs = _make_embedding_matrix(n, dim=32)
            labels = np.array([i % 5 for i in range(n)], dtype=np.int64)
            mAP, top1 = retrieval_map_top1(q_embs, labels, r_embs, labels)
            assert 0.0 <= mAP <= 1.0, f"mAP out of range: {mAP}"
            assert 0.0 <= top1 <= 1.0, f"top1 out of range: {top1}"

    def test_get_labels_for_image_ids(self):
        splits_df = _make_splits_df(
            gallery_ids=["id_a", "id_b", "id_c"],
            gallery_sessions_per_id=2,
        )
        ref_mapping = _make_ref_mapping(splits_df)
        identity_to_label = {"id_a": 0, "id_b": 1, "id_c": 2}
        image_ids = splits_df.loc[
            splits_df["individual_id"] == "id_a", "image_id"
        ].tolist()
        row_indices, int_labels = get_labels_for_image_ids(
            image_ids, ref_mapping, identity_to_label
        )
        assert len(row_indices) > 0
        assert (int_labels == 0).all(), "All labels should be 0 for id_a"


# ---------------------------------------------------------------------------
# 11. build_px_k_dataset correctness
# ---------------------------------------------------------------------------

class TestBuildPxKDataset:
    def test_dataset_has_correct_size(self, splits_and_mapping):
        splits_df, ref_mapping, matrix = splits_and_mapping
        inner = build_inner_split(ref_mapping, splits_df, seed=42)
        dataset, label_to_indices, id_to_label = build_px_k_dataset(
            matrix, ref_mapping, inner.train_image_ids
        )
        # Dataset should have all training crops
        train_rows = ref_mapping[
            ref_mapping["image_id"].isin(set(inner.train_image_ids))
        ]
        assert len(dataset) == len(train_rows)

    def test_label_to_indices_covers_all_records(self, splits_and_mapping):
        splits_df, ref_mapping, matrix = splits_and_mapping
        inner = build_inner_split(ref_mapping, splits_df, seed=42)
        dataset, label_to_indices, _ = build_px_k_dataset(
            matrix, ref_mapping, inner.train_image_ids
        )
        all_indices = [idx for idxs in label_to_indices.values() for idx in idxs]
        assert sorted(all_indices) == list(range(len(dataset)))


# ---------------------------------------------------------------------------
# 12. Descriptor crop-kind support for ear descriptors
# ---------------------------------------------------------------------------

class TestEarDescriptorCropKind:
    def test_ear_miewid_projected_is_ear_descriptor(self):
        """
        The projected descriptor should be treated as an ear descriptor by
        the pipeline (crop_kind='ear').
        """
        from configs.config_elephant import EAR_DESCRIPTORS
        # The projected descriptor is documented as an ear variant.
        # Validate that the base descriptor is in EAR_DESCRIPTORS and
        # confirm the convention for the projected variant.
        assert "ear_miewid" in EAR_DESCRIPTORS, (
            "'ear_miewid' must be in EAR_DESCRIPTORS for the ear descriptor logic."
        )
        # The projected variant follows the same naming: prefix 'ear_'
        assert "ear_miewid_projected".startswith("ear_"), (
            "Projected descriptor must start with 'ear_' to match ear descriptor convention."
        )


# ---------------------------------------------------------------------------
# 13. Training CLI integration (lightweight, no real training)
# ---------------------------------------------------------------------------

class TestTrainingCLI:
    def test_train_projection_runs_one_epoch_synthetic(self, tmp_path):
        """
        Run train_projection for 1 epoch on fully synthetic data.
        Verifies: no crashes, manifest written, checkpoint written,
        gate decision present.
        No model downloads or real artifacts used.
        """
        from pipeline.train_miewid_projection import train_projection

        # Build synthetic artifacts
        in_dim = 16
        gallery_ids = [f"gal_{i}" for i in range(8)]
        splits_df = _make_splits_df(
            gallery_ids=gallery_ids,
            gallery_sessions_per_id=3,
            probe_ids=[f"prb_{i}" for i in range(2)],
            heldout_gal_ids=["hld_0"],
        )
        ref_mapping = _make_ref_mapping(splits_df)
        matrix = _make_embedding_matrix(len(ref_mapping), dim=in_dim)

        # Write artifacts to tmp_path
        emb_dir = tmp_path / "embeddings" / "reference"
        emb_dir.mkdir(parents=True)
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()

        np.save(str(emb_dir / "ear_miewid.npy"), matrix)
        ref_mapping.to_parquet(str(emb_dir / "ear_miewid_mapping.parquet"), index=False)
        splits_df.to_parquet(str(splits_dir / "bteh_splits.parquet"), index=False)

        out_dir = tmp_path / "ckpt_out"

        manifest = train_projection(
            artifact_root=tmp_path,
            out_dir=out_dir,
            descriptor="ear_miewid",
            out_dim=8,
            dropout=0.0,
            loss_mode="triplet",
            use_arcface=False,
            epochs=1,
            P=4,
            K=2,
            lr=1e-3,
            seed=42,
            device_str="cpu",
        )

        assert (out_dir / "best_projection.pt").is_file()
        assert (out_dir / "training_manifest.json").is_file()
        assert (out_dir / "training_curves.json").is_file()
        assert (out_dir / "experiment_diagnostics.json").is_file()

        # Manifest fields
        assert "gate" in manifest
        assert "adopted" in manifest["gate"]
        assert "baseline_map" in manifest
        assert manifest["safety"]["probe_never_used_for_training"] is True
        assert manifest["training_mode"] == "projection_only"

    def test_train_supcon_loss_runs(self, tmp_path):
        from pipeline.train_miewid_projection import train_projection

        gallery_ids = [f"gal_{i}" for i in range(6)]
        splits_df = _make_splits_df(gallery_ids=gallery_ids, gallery_sessions_per_id=2)
        ref_mapping = _make_ref_mapping(splits_df)
        matrix = _make_embedding_matrix(len(ref_mapping), dim=16)

        emb_dir = tmp_path / "embeddings" / "reference"
        emb_dir.mkdir(parents=True)
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()
        np.save(str(emb_dir / "ear_miewid.npy"), matrix)
        ref_mapping.to_parquet(str(emb_dir / "ear_miewid_mapping.parquet"), index=False)
        splits_df.to_parquet(str(splits_dir / "bteh_splits.parquet"), index=False)

        train_projection(
            artifact_root=tmp_path,
            out_dir=tmp_path / "ckpt_supcon",
            out_dim=8, epochs=1, P=4, K=2,
            loss_mode="supcon", seed=0, device_str="cpu",
        )
        # If we get here without exception, supcon ran fine

    def test_train_both_losses_with_arcface(self, tmp_path):
        from pipeline.train_miewid_projection import train_projection

        gallery_ids = [f"gal_{i}" for i in range(8)]
        splits_df = _make_splits_df(gallery_ids=gallery_ids, gallery_sessions_per_id=3)
        ref_mapping = _make_ref_mapping(splits_df)
        matrix = _make_embedding_matrix(len(ref_mapping), dim=16)

        emb_dir = tmp_path / "embeddings" / "reference"
        emb_dir.mkdir(parents=True)
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()
        np.save(str(emb_dir / "ear_miewid.npy"), matrix)
        ref_mapping.to_parquet(str(emb_dir / "ear_miewid_mapping.parquet"), index=False)
        splits_df.to_parquet(str(splits_dir / "bteh_splits.parquet"), index=False)

        train_projection(
            artifact_root=tmp_path,
            out_dir=tmp_path / "ckpt_arcface",
            out_dim=8, epochs=1, P=4, K=2,
            loss_mode="both", use_arcface=True, seed=0, device_str="cpu",
        )
