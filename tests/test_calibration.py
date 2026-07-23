import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from models.calibration import Calibrator
from configs.config_elephant import MIN_POSITIVE_PAIRS_FOR_ISOTONIC


def _make_scores_labels(rng, n_pos, n_neg):
    pos_scores = rng.uniform(0.5, 1.0, size=n_pos)
    neg_scores = rng.uniform(0.0, 0.5, size=n_neg)
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    return scores, labels


def test_isotonic_fit_transform_monotone():
    rng = np.random.default_rng(seed=42)
    n_pos = MIN_POSITIVE_PAIRS_FOR_ISOTONIC + 50
    scores, labels = _make_scores_labels(rng, n_pos, n_pos)

    cal = Calibrator()
    cal.fit(scores, labels)

    assert cal.method == "isotonic"

    grid = np.linspace(0.0, 1.0, 200)
    out = cal.transform(grid)

    diffs = np.diff(out)
    assert np.all(diffs >= -1e-6), "transform output is not monotonically non-decreasing"


def test_platt_fallback_fit_transform():
    """Low-support calibration now uses Platt (logistic) scaling, not temperature.

    temperature expit(score/T) was replaced because it cannot represent
    probabilities < 0.5 for positive cosine scores (score > 0 → prob > 0.5).
    Platt scaling with a fitted intercept covers the full [0, 1] range.
    """
    rng = np.random.default_rng(seed=42)
    n_pos = MIN_POSITIVE_PAIRS_FOR_ISOTONIC - 1
    scores, labels = _make_scores_labels(rng, n_pos, n_pos)

    cal = Calibrator()
    cal.fit(scores, labels)

    # Method must now be 'platt', not 'temperature'.
    assert cal.method == Calibrator.PLATT, (
        f"Expected 'platt' fallback; got '{cal.method}'. "
        "The temperature fallback was replaced because expit(s/T) cannot "
        "represent probs < 0.5 for positive cosine scores."
    )

    test_scores = rng.uniform(-2.0, 2.0, size=100)
    out = cal.transform(test_scores)

    assert np.all(out >= 0.0), "Platt output contains values < 0"
    assert np.all(out <= 1.0), "Platt output contains values > 1"


def test_explicit_platt_with_high_support():
    scores = np.linspace(0.1, 0.9, 500)
    labels = np.concatenate([np.zeros(250), np.ones(250)])
    cal = Calibrator().fit(scores, labels, method="platt")
    assert cal.method == Calibrator.PLATT
    transformed = cal.transform(np.array([0.2, 0.8]))
    assert transformed[0] < transformed[1]


def test_calibrator_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(seed=42)
    n_pos = MIN_POSITIVE_PAIRS_FOR_ISOTONIC + 10
    scores, labels = _make_scores_labels(rng, n_pos, n_pos)

    cal = Calibrator()
    cal.fit(scores, labels)

    save_path = str(tmp_path / "calibrator.pkl")
    cal.save(save_path)

    loaded = Calibrator().load(save_path)

    test_scores = rng.uniform(0.0, 1.0, size=50)
    original_out = cal.transform(test_scores)
    loaded_out = loaded.transform(test_scores)

    np.testing.assert_allclose(original_out, loaded_out, rtol=1e-5)


def test_calibrator_output_range():
    rng = np.random.default_rng(seed=42)
    n_pos = MIN_POSITIVE_PAIRS_FOR_ISOTONIC + 20
    scores, labels = _make_scores_labels(rng, n_pos, n_pos)

    cal = Calibrator()
    cal.fit(scores, labels)

    random_scores = rng.uniform(-5.0, 5.0, size=500)
    out = cal.transform(random_scores)

    assert np.all(out >= 0.0), "output below 0"
    assert np.all(out <= 1.0), "output above 1"
