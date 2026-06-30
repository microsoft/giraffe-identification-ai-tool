# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

import os
import sys
import logging
import pickle
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit  # numerically stable sigmoid
from sklearn.isotonic import IsotonicRegression

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_elephant import MIN_POSITIVE_PAIRS_FOR_ISOTONIC

logger = logging.getLogger(__name__)


def _log_loss_temperature(T: float, scores: np.ndarray, is_same: np.ndarray) -> float:
    probs = expit(scores / T)
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return -np.mean(is_same * np.log(probs) + (1 - is_same) * np.log(1 - probs))


class Calibrator:
    """
    Maps raw similarity scores to calibrated probabilities in [0, 1].
    Auto-selects isotonic regression or temperature scaling depending on
    the number of positive pairs available.
    """

    def __init__(self):
        self._method: str | None = None
        self._iso: IsotonicRegression | None = None
        self._temperature: float | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, scores: np.ndarray, is_same: np.ndarray) -> "Calibrator":
        scores  = np.asarray(scores,  dtype=np.float64)
        is_same = np.asarray(is_same, dtype=np.float64)

        n_positive = int(is_same.sum())

        if n_positive >= MIN_POSITIVE_PAIRS_FOR_ISOTONIC:
            self._method = "isotonic"
            logger.info(
                "Calibrator: using isotonic regression (%d positive pairs >= threshold %d).",
                n_positive,
                MIN_POSITIVE_PAIRS_FOR_ISOTONIC,
            )
            self._iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
            self._iso.fit(scores, is_same)
        else:
            self._method = "temperature"
            logger.info(
                "Calibrator: using temperature scaling (%d positive pairs < threshold %d).",
                n_positive,
                MIN_POSITIVE_PAIRS_FOR_ISOTONIC,
            )
            result = minimize_scalar(
                _log_loss_temperature,
                args=(scores, is_same),
                bounds=(1e-3, 100.0),
                method="bounded",
            )
            self._temperature = float(result.x)
            logger.info("Calibrator: optimal temperature = %.4f", self._temperature)

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)

        if self._method == "isotonic":
            return self._iso.predict(scores).astype(np.float32)
        elif self._method == "temperature":
            return expit(scores / self._temperature).astype(np.float32)
        else:
            raise RuntimeError("Calibrator has not been fitted yet.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("Calibrator saved to %s", path)

    def load(self, path: str) -> "Calibrator":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        self._method      = obj._method
        self._iso         = obj._iso
        self._temperature = obj._temperature
        return self

    # ------------------------------------------------------------------
    # Property
    # ------------------------------------------------------------------

    @property
    def method(self) -> str:
        if self._method is None:
            raise RuntimeError("Calibrator has not been fitted yet.")
        return self._method
