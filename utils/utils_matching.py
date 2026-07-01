# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
import time
import faiss
import pickle
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_matching import faiss_index_dir


def train_faiss(all_descriptors_train):
    print('\nTraining faiss index started ...')
    start_time = time.time()

    all_descriptors_train_normalized = normalize(all_descriptors_train)

    faiss_index = faiss.IndexHNSWFlat(all_descriptors_train.shape[1], 16)
    faiss_index.add(all_descriptors_train_normalized)

    print('\nTraining time for faiss {:.6f} seconds'.format(time.time() - start_time))
    return faiss_index


def write_faiss(faiss_index, all_descriptors_train, all_labels_train, all_serials_train, faiss_index_dir, subdir=None, activate=False):
    if subdir is not None:
        faiss_index_dir = os.path.join(faiss_index_dir, subdir)
    os.makedirs(faiss_index_dir, exist_ok=True)
    print('\nWriting faiss index started ...')
    print('\nPath to write faiss index: ', faiss_index_dir)
    start_time = time.time()
    faiss.write_index(faiss_index, os.path.join(faiss_index_dir, 'faiss_index.index'))
    print('\nWriting time for faiss {:.6f} seconds'.format(time.time() - start_time))

    print('\nWriting supplemental data for faiss index started ...')
    start_time = time.time()
    with open(os.path.join(faiss_index_dir, 'all_descriptors_train.pkl'), 'wb') as file:
        pickle.dump(all_descriptors_train, file)
    with open(os.path.join(faiss_index_dir, 'all_labels_train.pkl'), 'wb') as file:
        pickle.dump(all_labels_train, file)
    with open(os.path.join(faiss_index_dir, 'all_serials_train.pkl'), 'wb') as file:
        pickle.dump(all_serials_train, file)
    print('\nWriting time for supplemental data for faiss index {:.6f} seconds'.format(time.time() - start_time))


def read_faiss():
    print('\nReading faiss index started ...')
    print('\nPath to read faiss index: ', faiss_index_dir)
    start_time = time.time()
    faiss_index = faiss.read_index(os.path.join(faiss_index_dir, 'faiss_index.index'))
    print('\nReading time for faiss {:.6f} seconds'.format(time.time() - start_time))

    print('\nReading supplemental data for faiss index started ...')
    start_time = time.time()
    with open(os.path.join(faiss_index_dir, 'all_descriptors_train.pkl'), 'rb') as file:
        all_descriptors_train = pickle.load(file)
    with open(os.path.join(faiss_index_dir, 'all_labels_train.pkl'), 'rb') as file:
        all_labels_train = pickle.load(file)
    with open(os.path.join(faiss_index_dir, 'all_serials_train.pkl'), 'rb') as file:
        all_serials_train = pickle.load(file)
    print('\nReading time for supplemental data for faiss index {:.6f} seconds'.format(time.time() - start_time))

    return faiss_index, all_descriptors_train, all_labels_train, all_serials_train


def load_trained_faiss_ref(faiss_index_dir):
    filenames = [
        'all_descriptors_train.pkl',
        'all_labels_train.pkl',
        'all_serials_train.pkl',
        'faiss_index.index',
    ]
    if all(os.path.isfile(os.path.join(faiss_index_dir, f)) for f in filenames):
        return read_faiss()
    else:
        print('index not available to load.')
        sys.exit()


def normalize(all_descriptors_train):
    norms = np.linalg.norm(all_descriptors_train, axis=1, keepdims=True)
    all_descriptors_train_normalized = all_descriptors_train / norms
    return all_descriptors_train_normalized.astype(np.float32)


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0


def replace_negatives_with_unique_values(array, target_value=-1):
    some_large_number = 10000000

    target_indices = np.where(array == target_value)[0]
    existing_values = set(array[array != target_value])

    replacement_values = set(
        range(
            max(existing_values, default=0) + some_large_number,
            max(existing_values, default=0) + some_large_number + len(target_indices),
        )
    )
    unique_replacements = iter(replacement_values)

    new_array = array.copy()
    for index in target_indices:
        new_array[index] = next(unique_replacements)

    return new_array


def run_union_find(col1, col2):
    uf = UnionFind()

    for val in np.concatenate([col1, col2]):
        uf.add(val)

    for v1, v2 in zip(col1, col2):
        uf.union(v1, v2)

    value_to_new_id = {}
    for val in np.concatenate([col1, col2]):
        root = uf.find(val)
        if root not in value_to_new_id:
            value_to_new_id[root] = len(value_to_new_id)
        value_to_new_id[val] = value_to_new_id[root]

    return value_to_new_id


# ---------------------------------------------------------------------------
# Embedding-based FAISS helpers (elephant pipeline)
# ---------------------------------------------------------------------------

def build_global_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build an IndexFlatIP from L2-normalized embeddings.

    Inner product on L2-normalised vectors equals cosine similarity, so
    higher scores mean better matches.
    """
    embeddings = embeddings.astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def build_and_save_global_index(
    embeddings: np.ndarray, desc_name: str, faiss_index_dir: str
) -> faiss.IndexFlatIP:
    """Build and persist an IndexFlatIP for a single descriptor."""
    os.makedirs(faiss_index_dir, exist_ok=True)
    index = build_global_index(embeddings)
    out_path = os.path.join(faiss_index_dir, f"{desc_name}.index")
    faiss.write_index(index, out_path)
    print(f"FAISS index for '{desc_name}' saved to {out_path} ({index.ntotal} vectors).")
    return index


def load_global_index(desc_name: str, faiss_index_dir: str) -> faiss.IndexFlatIP:
    """Load a persisted IndexFlatIP for a single descriptor."""
    index_path = os.path.join(faiss_index_dir, f"{desc_name}.index")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    index = faiss.read_index(index_path)
    print(f"FAISS index for '{desc_name}' loaded from {index_path} ({index.ntotal} vectors).")
    return index
