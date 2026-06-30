import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("data_root_abs_path", "/tmp")
os.environ.setdefault("container_name", "test_container")

import numpy as np
import pytest

from utils.utils_matching import UnionFind, replace_negatives_with_unique_values, run_union_find
from utils.helpers_matching import mint_new_individual_id
from configs.config_elephant import NEW_ID_PREFIX


def test_union_find_basic():
    uf = UnionFind()
    for x in range(4):
        uf.add(x)

    uf.union(0, 1)
    uf.union(1, 2)

    assert uf.find(0) == uf.find(2), "0 and 2 should be in the same component"
    assert uf.find(3) != uf.find(0), "3 should be isolated from 0"


def test_replace_negatives_with_unique_values():
    array = np.array([5, -1, 3, -1, 7, -1], dtype=np.int64)
    result = replace_negatives_with_unique_values(array)

    neg_indices = np.where(array == -1)[0]
    replaced = result[neg_indices]

    assert len(set(replaced)) == len(replaced), "replaced values are not all unique"
    assert np.all(result[array != -1] == array[array != -1]), "non-negative values changed"
    assert np.all(result != -1), "some -1 values remain unreplaced"


def test_run_union_find():
    col1 = np.array([0, 1, 3])
    col2 = np.array([1, 2, 4])

    mapping = run_union_find(col1, col2)

    assert mapping[0] == mapping[1] == mapping[2], "0, 1, 2 should share a cluster id"
    assert mapping[3] == mapping[4], "3 and 4 should share a cluster id"
    assert mapping[0] != mapping[3], "cluster of {0,1,2} must differ from {3,4}"


def test_mint_new_individual_id():
    id1 = mint_new_individual_id()
    id2 = mint_new_individual_id()

    assert id1 != id2, "mint_new_individual_id should produce unique IDs"
    assert id1.startswith(NEW_ID_PREFIX), f"ID '{id1}' does not start with '{NEW_ID_PREFIX}'"
    assert id2.startswith(NEW_ID_PREFIX), f"ID '{id2}' does not start with '{NEW_ID_PREFIX}'"
