import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("data_root_abs_path", "/tmp")
os.environ.setdefault("container_name", "test_container")

import numpy as np
import faiss
import pytest

from utils.utils_embeddings import l2_normalize
from utils.utils_matching import build_and_save_global_index, load_global_index, build_global_index


def test_build_and_load_index(tmp_path):
    rng = np.random.default_rng(seed=42)
    matrix = rng.random((50, 128)).astype(np.float32)
    matrix = l2_normalize(matrix)

    build_and_save_global_index(matrix, "test_desc", str(tmp_path))
    index = load_global_index("test_desc", str(tmp_path))

    query = rng.random(128).astype(np.float32)
    query = query / np.linalg.norm(query)

    distances, indices = index.search(query.reshape(1, -1), 5)

    assert indices.shape == (1, 5)
    assert np.all(indices[0] >= 0)
    assert np.all(indices[0] <= 49)


def test_index_flat_ip_cosine():
    rng = np.random.default_rng(seed=42)
    matrix = rng.random((30, 64)).astype(np.float32)
    matrix = l2_normalize(matrix)

    index = build_global_index(matrix)

    for row_idx in range(5):
        query = matrix[row_idx].reshape(1, -1)
        distances, indices = index.search(query, 1)
        assert indices[0][0] == row_idx, (
            f"Expected top-1 to be self (row {row_idx}), got {indices[0][0]}"
        )
