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
from sklearn.linear_model import LogisticRegression

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_elephant import MIN_POSITIVE_PAIRS_FOR_ISOTONIC

logger = logging.getLogger(__name__)


# NOTE: The legacy temperature scaling expit(score/T) cannot represent
# probabilities below 0.5 for positive cosine scores (score > 0 always gives
# expit(score/T) > 0.5 when T > 0).  For datasets where same-individual pairs
# have moderate cosine overlap with negative pairs, the calibrated probability
# must be free to fall below 0.5.  Platt scaling fits an affine-monotonic
# sigmoid  sigmoid(a*score + b)  with a fitted intercept, covering the full
# [0, 1] range.  The legacy temperature method remains loadable for backward
# compatibility with existing pickles but is no longer fitted by default.


def _log_loss_temperature(T: float, scores: np.ndarray, is_same: np.ndarray) -> float:
    """Legacy temperature loss — kept for backward compatibility only."""
    probs = expit(scores / T)
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return -np.mean(is_same * np.log(probs) + (1 - is_same) * np.log(1 - probs))


class Calibrator:
    """
    Maps raw similarity scores to calibrated probabilities in [0, 1].

    Selection logic (controlled by ``fit``):
    - **isotonic**: used when positive-pair count >= MIN_POSITIVE_PAIRS_FOR_ISOTONIC.
      Fits sklearn IsotonicRegression on the full score/label set.
    - **platt**: used as the low-support fallback.  Fits logistic regression
      with an intercept (Platt scaling) so the mapping is affine-monotonic and
      can represent probabilities across the full [0, 1] range — including
      below 0.5 for positive cosine scores.
    - **temperature** (legacy, load-only): the old expit(score/T) mapping;
      still deserialised correctly from existing pkl files but never fitted
      by new code.  Cannot represent probs < 0.5 for score > 0.

    The ``fit_reason`` attribute records why the selected method was chosen,
    together with support counts, for inclusion in the calibration manifest.
    """

    ISOTONIC = "isotonic"
    PLATT = "platt"
    TEMPERATURE = "temperature"  # legacy load-only

    def __init__(self):
        self._method: str | None = None
        self._iso: IsotonicRegression | None = None
        self._platt: LogisticRegression | None = None
        # Legacy temperature field — retained so old pickles load correctly.
        self._temperature: float | None = None
        self.fit_reason: str | None = None
        self.fit_n_positive: int | None = None
        self.fit_n_negative: int | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        scores: np.ndarray,
        is_same: np.ndarray,
        method: str = "auto",
    ) -> "Calibrator":
        scores  = np.asarray(scores,  dtype=np.float64)
        is_same = np.asarray(is_same, dtype=np.float64)

        n_positive = int(is_same.sum())
        n_negative = int((1 - is_same).sum())
        total = len(is_same)

        if total == 0:
            raise ValueError("Calibrator.fit: empty score/label arrays.")
        if n_positive == 0:
            raise ValueError(
                "Calibrator.fit: no positive pairs (is_same all zeros). "
                "Cannot fit a calibrator with zero positive support."
            )
        if n_negative == 0:
            raise ValueError(
                "Calibrator.fit: no negative pairs (is_same all ones). "
                "Cannot fit a calibrator with zero negative support."
            )

        self.fit_n_positive = n_positive
        self.fit_n_negative = n_negative

        if method not in {"auto", self.ISOTONIC, self.PLATT}:
            raise ValueError(f"Unknown calibration method: {method!r}")

        use_isotonic = (
            method == self.ISOTONIC
            or (
                method == "auto"
                and n_positive >= MIN_POSITIVE_PAIRS_FOR_ISOTONIC
            )
        )
        if use_isotonic:
            self._method = self.ISOTONIC
            self.fit_reason = (
                f"isotonic: {n_positive} positive pairs >= threshold {MIN_POSITIVE_PAIRS_FOR_ISOTONIC}"
            )
            logger.info(
                "Calibrator: using isotonic regression (%d positive pairs >= threshold %d).",
                n_positive,
                MIN_POSITIVE_PAIRS_FOR_ISOTONIC,
            )
            self._iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
            self._iso.fit(scores, is_same)
        else:
            self._method = self.PLATT
            reason = (
                "explicitly selected"
                if method == self.PLATT
                else f"{n_positive} positive pairs < threshold {MIN_POSITIVE_PAIRS_FOR_ISOTONIC}"
            )
            self.fit_reason = (
                f"platt: {reason}; affine logistic scaling supports the full "
                "[0,1] probability range"
            )
            logger.info(
                "Calibrator: using Platt (logistic) scaling (%d positive pairs < threshold %d). "
                "Reason: temperature expit(s/T) cannot represent probs < 0.5 for s > 0.",
                n_positive,
                MIN_POSITIVE_PAIRS_FOR_ISOTONIC,
            )
            # Use high C (low regularisation) so the fit is close to unregularised Platt.
            # solver='lbfgs' converges reliably on 1-D feature with intercept.
            self._platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            self._platt.fit(scores.reshape(-1, 1), is_same)

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)

        if self._method == self.ISOTONIC:
            return self._iso.predict(scores).astype(np.float32)
        elif self._method == self.PLATT:
            probs = self._platt.predict_proba(scores.reshape(-1, 1))[:, 1]
            return probs.astype(np.float32)
        elif self._method == self.TEMPERATURE:
            # Legacy: load-only path for old pickles.
            return expit(scores / self._temperature).astype(np.float32)
        else:
            raise RuntimeError("Calibrator has not been fitted yet.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("Calibrator saved to %s", path)

    def load(self, path: str) -> "Calibrator":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        self._method      = obj._method
        self._iso         = getattr(obj, "_iso", None)
        self._platt       = getattr(obj, "_platt", None)
        # Legacy temperature field.
        self._temperature = getattr(obj, "_temperature", None)
        self.fit_reason   = getattr(obj, "fit_reason", None)
        self.fit_n_positive = getattr(obj, "fit_n_positive", None)
        self.fit_n_negative = getattr(obj, "fit_n_negative", None)
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def method(self) -> str:
        if self._method is None:
            raise RuntimeError("Calibrator has not been fitted yet.")
        return self._method
