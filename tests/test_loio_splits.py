import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("data_root_abs_path", "/tmp")
os.environ.setdefault("container_name", "test_container")

import numpy as np
import pandas as pd
import pytest

from utils.helpers_matching import make_loio_splits


def _make_metadata(n_individuals=5, images_per_individual=3, seed=42):
    rng = np.random.default_rng(seed=seed)
    rows = []
    for i in range(n_individuals):
        ind_id = f"eleph_{i:03d}"
        for j in range(images_per_individual):
            rows.append({
                "individual_id": ind_id,
                "image_id": f"img_{i:03d}_{j:03d}",
            })
    return pd.DataFrame(rows)


def _make_metadata_with_sessions(n_individuals=5, images_per_individual=3, seed=42):
    rng = np.random.default_rng(seed=seed)
    df = _make_metadata(n_individuals, images_per_individual, seed)
    n_sessions = n_individuals * images_per_individual // 2
    sessions = [f"session_{k % n_sessions}" for k in range(len(df))]
    df["session"] = sessions
    return df


def test_loio_no_individual_leakage():
    df = _make_metadata(n_individuals=5, images_per_individual=3)
    for train, probe in make_loio_splits(df, id_col="individual_id"):
        probe_ids = set(probe["individual_id"].unique())
        train_ids = set(train["individual_id"].unique())
        assert probe_ids.isdisjoint(train_ids), (
            f"Individual(s) {probe_ids & train_ids} appear in both probe and train"
        )


def test_loio_session_leakage_guard():
    df = _make_metadata_with_sessions(n_individuals=5, images_per_individual=3)
    for train, probe in make_loio_splits(df, id_col="individual_id", session_col="session"):
        probe_sessions = set(probe["session"].dropna())
        train_sessions = set(train["session"].dropna())
        shared = probe_sessions & train_sessions
        assert len(shared) == 0, (
            f"Sessions {shared} appear in both probe and train sets"
        )


def test_loio_yields_all_individuals():
    n_individuals = 5
    df = _make_metadata(n_individuals=n_individuals, images_per_individual=3)
    folds = list(make_loio_splits(df, id_col="individual_id"))
    assert len(folds) == n_individuals, (
        f"Expected {n_individuals} folds, got {len(folds)}"
    )


def test_loio_probe_covers_all_images_of_individual():
    df = _make_metadata(n_individuals=5, images_per_individual=3)
    for train, probe in make_loio_splits(df, id_col="individual_id"):
        assert len(probe["individual_id"].unique()) == 1
        held_out_id = probe["individual_id"].iloc[0]
        all_images_of_individual = set(
            df[df["individual_id"] == held_out_id]["image_id"]
        )
        probe_images = set(probe["image_id"])
        assert probe_images == all_images_of_individual, (
            f"Probe for {held_out_id} missing images: "
            f"{all_images_of_individual - probe_images}"
        )
