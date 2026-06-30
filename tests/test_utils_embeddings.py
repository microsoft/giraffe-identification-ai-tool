import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from utils.utils_embeddings import (
    l2_normalize,
    cosine_topk,
    save_embeddings,
    load_embeddings_matrix,
)


def test_l2_normalize_unit_norm():
    rng = np.random.default_rng(seed=42)
    matrix = rng.random((20, 64)).astype(np.float32)
    normed = l2_normalize(matrix)
    row_norms = np.linalg.norm(normed, axis=1)
    np.testing.assert_allclose(row_norms, np.ones(20), atol=1e-6)


def test_l2_normalize_zero_row():
    rng = np.random.default_rng(seed=42)
    matrix = rng.random((10, 32)).astype(np.float32)
    matrix[3] = 0.0
    normed = l2_normalize(matrix)
    assert np.all(np.isfinite(normed)), "l2_normalize produced NaN or inf for zero row"
    np.testing.assert_array_equal(normed[3], np.zeros(32, dtype=np.float32))


def test_cosine_topk_shape():
    rng = np.random.default_rng(seed=42)
    query_vecs = l2_normalize(rng.random((5, 64)).astype(np.float32))
    ref_vecs = l2_normalize(rng.random((20, 64)).astype(np.float32))
    k = 3

    all_dists = []
    all_idxs = []
    for i in range(query_vecs.shape[0]):
        dists, idxs = cosine_topk(query_vecs[i], ref_vecs, k)
        all_dists.append(dists)
        all_idxs.append(idxs)

    assert len(all_dists) == 5
    assert len(all_idxs) == 5
    for d, ix in zip(all_dists, all_idxs):
        assert d.shape == (k,), f"distances shape mismatch: {d.shape}"
        assert ix.shape == (k,), f"indices shape mismatch: {ix.shape}"


def test_cosine_topk_self_similarity():
    rng = np.random.default_rng(seed=42)
    matrix = l2_normalize(rng.random((15, 64)).astype(np.float32))

    for i in range(matrix.shape[0]):
        dists, idxs = cosine_topk(matrix[i], matrix, k=1)
        assert idxs[0] == i, f"top-1 for row {i} was {idxs[0]}, expected {i}"


def test_save_load_embeddings_roundtrip(tmp_path):
    rng = np.random.default_rng(seed=42)
    matrix = rng.random((25, 128)).astype(np.float32)

    save_embeddings(matrix, str(tmp_path), "reference", "megadescriptor")
    loaded = load_embeddings_matrix(str(tmp_path), "reference", "megadescriptor")

    np.testing.assert_allclose(matrix, loaded, rtol=1e-6)
