# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the head-region schema extension and detector foundation.

Coverage:
  - Schema accepts one head / rejects duplicate or ordinal > 0.
  - Ear detector behaviour is unchanged after refactor.
  - Shared GroundingDINO backend loaded once when injected.
  - Head-specific detection constraints (area, aspect, NMS).
  - Deterministic single-best head result after NMS.
  - Body-first / original-image fallback in run_head_detection.
  - Terminal-status resumability (0/1 head).
  - Unavailable model fails pre-loop (no partial records written).
  - Detector fingerprint and schema/source/split join checks.
  - Production v1 crop manifest not mutated by head detection.
"""

import importlib.util
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_bteh import ARTIFACT_SCHEMA_VERSION
from models.detector import (
    EarDetector,
    HeadDetector,
    _GroundingDINOBackend,
    _iou,
    _nms,
)
from pipeline.step_1_run_head_detection import (
    head_detector_fingerprint,
    run_head_detection,
)
from utils.artifact_schema import (
    CROP_MANIFEST_COLUMNS,
    HEAD_EXPERIMENT_MANIFEST_COLUMNS,
    VALID_CROP_KINDS,
    assert_crop_manifest_integrity,
    assert_head_experiment_manifest_integrity,
    make_crop_id,
)


# ---------------------------------------------------------------------------
# Parquet engine shim (for CI without pyarrow/fastparquet)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _parquet_engine_fallback(monkeypatch):
    if importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet"):
        return
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda frame, path, index=False: frame.to_pickle(path),
    )
    monkeypatch.setattr(pd, "read_parquet", pd.read_pickle)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_manifest(image_ids=("img1",), individual_id="elephant1"):
    return pd.DataFrame({
        "image_id": list(image_ids),
        "individual_id": [individual_id] * len(image_ids),
    })


def _head_crop_record(image_id="img1", ordinal=0, **overrides):
    record = {
        "crop_id": make_crop_id(image_id, "head", ordinal),
        "image_id": image_id,
        "individual_id": "elephant1",
        "crop_kind": "head",
        "crop_ordinal": ordinal,
        "crop_path": f"/data/{image_id}__head_{ordinal}.jpg",
        "detector_confidence": 0.75,
        "detector_box": "[10,10,50,50]",
        "detector_status": "accepted",
        "review_status": "pending",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_fingerprint": "src_fp",
        "split_fingerprint": "spl_fp",
        "source_used": "body_crop",
        "detector_fingerprint": "abcd1234abcd1234",
    }
    record.update(overrides)
    return record


def _head_df(rows):
    return pd.DataFrame(rows, columns=HEAD_EXPERIMENT_MANIFEST_COLUMNS)


def _write_source(path):
    image = np.full((100, 100, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return image


def _detection_manifest(source_path, image_id="img1", individual_id="elephant1"):
    return pd.DataFrame({
        "image_id": [image_id],
        "individual_id": [individual_id],
        "include_status": ["included"],
        "source_path": [str(source_path)],
    })


def _v1_body_manifest(body_crop_path, image_id="img1", individual_id="elephant1"):
    """Minimal production v1 body crop manifest (read-only input)."""
    return pd.DataFrame({
        "crop_id": [make_crop_id(image_id, "body", 0)],
        "image_id": [image_id],
        "individual_id": [individual_id],
        "crop_kind": ["body"],
        "crop_ordinal": [0],
        "crop_path": [str(body_crop_path)],
        "detector_confidence": [0.9],
        "detector_box": ["[0,0,100,100]"],
        "detector_status": ["accepted"],
        "review_status": ["pending"],
        "schema_version": [ARTIFACT_SCHEMA_VERSION],
        "source_fingerprint": ["src"],
        "split_fingerprint": ["spl"],
    })


# ---------------------------------------------------------------------------
# Fake GroundingDINO processor and model (no real GPU / model required)
# ---------------------------------------------------------------------------

class _FakeInputs(dict):
    input_ids = np.array([[1]])

    def to(self, _device):
        return self


class _FakeProcessor:
    def __init__(self, boxes, scores):
        self.boxes = np.asarray(boxes, dtype=np.float32)
        self.scores = np.asarray(scores, dtype=np.float32)

    def __call__(self, **_kwargs):
        return _FakeInputs()

    def post_process_grounded_object_detection(self, *_args, **_kwargs):
        return [{"boxes": self.boxes, "scores": self.scores}]


class _FakeModel:
    def __call__(self, **_kwargs):
        return object()


def _head_detector(boxes=(), scores=()):
    """Build a HeadDetector bypassing real model loading."""
    detector = HeadDetector.__new__(HeadDetector)
    detector._available = True
    detector.device = "cpu"
    detector.processor = _FakeProcessor(boxes, scores)
    detector.model = _FakeModel()
    return detector


def _ear_detector(boxes=(), scores=()):
    """Build an EarDetector bypassing real model loading."""
    detector = EarDetector.__new__(EarDetector)
    detector._available = True
    detector.device = "cpu"
    detector.processor = _FakeProcessor(boxes, scores)
    detector.model = _FakeModel()
    return detector


# Pipeline head detector that also tracks call inputs
class _PipelineHeadDetector:
    def __init__(self, found=True):
        self.found = found
        self.inputs = []
        self._available = True

    def detect_head(self, image_bgr, **_kwargs):
        self.inputs.append(image_bgr.copy())
        if not self.found:
            return None
        h, w = image_bgr.shape[:2]
        bw, bh = max(1, w // 4), max(1, h // 4)
        crop = image_bgr[:bh, :bw].copy()
        return {
            "box": [0, 0, bw, bh],
            "score": 0.80,
            "crop": crop,
            "ordinal": 0,
            "source": "grounding_dino",
        }


# ---------------------------------------------------------------------------
# 1. Schema: valid crop kinds include 'head'
# ---------------------------------------------------------------------------

class TestSchemaHeadKind:
    def test_head_in_valid_crop_kinds(self):
        assert "head" in VALID_CROP_KINDS

    def test_make_crop_id_head_ordinal_0(self):
        assert make_crop_id("img1", "head", 0) == "img1__head_0"

    def test_make_crop_id_head_ordinal_1_raises(self):
        with pytest.raises(ValueError, match="crop_ordinal"):
            make_crop_id("img1", "head", 1)

    def test_make_crop_id_head_ordinal_negative_raises(self):
        with pytest.raises(ValueError, match="crop_ordinal"):
            make_crop_id("img1", "head", -1)

    def test_make_crop_id_body_unchanged(self):
        assert make_crop_id("img1", "body", 0) == "img1__body_0"

    def test_make_crop_id_ear_unchanged(self):
        assert make_crop_id("img1", "ear", 1) == "img1__ear_1"

    def test_make_crop_id_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="crop_kind"):
            make_crop_id("img1", "tail", 0)


class TestSchemaHeadCardinality:
    def test_one_head_valid(self):
        df = _head_df([_head_crop_record()])
        assert_crop_manifest_integrity(df, _image_manifest())

    def test_two_heads_same_image_fails(self):
        records = [
            _head_crop_record(crop_id="img1__head_0"),
            _head_crop_record(crop_id="img1__head_0_dup"),
        ]
        df = _head_df(records)
        with pytest.raises(AssertionError, match="cardinality"):
            assert_crop_manifest_integrity(df, _image_manifest())

    def test_head_ordinal_nonzero_fails(self):
        with pytest.raises(ValueError, match="crop_ordinal"):
            _head_crop_record(ordinal=1)  # make_crop_id raises

    def test_existing_body_ear_records_unaffected(self):
        """Head extension does not invalidate manifests with only body+ear."""
        body_record = {
            "crop_id": "img1__body_0",
            "image_id": "img1",
            "individual_id": "elephant1",
            "crop_kind": "body",
            "crop_ordinal": 0,
            "crop_path": "/data/body.jpg",
            "detector_confidence": 0.9,
            "detector_box": None,
            "detector_status": "accepted",
            "review_status": "pending",
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "source_fingerprint": None,
            "split_fingerprint": None,
        }
        df = pd.DataFrame([body_record], columns=CROP_MANIFEST_COLUMNS)
        assert_crop_manifest_integrity(df, _image_manifest())


# ---------------------------------------------------------------------------
# 2. Head experiment manifest integrity check
# ---------------------------------------------------------------------------

class TestHeadExperimentManifestIntegrity:
    def _manifest(self, rows):
        return _head_df(rows)

    def test_valid_head_manifest_passes(self):
        assert_head_experiment_manifest_integrity(
            self._manifest([_head_crop_record()]),
            _image_manifest(),
        )

    def test_missing_source_used_column_fails(self):
        df = self._manifest([_head_crop_record()])
        df = df.drop(columns=["source_used"])
        with pytest.raises(AssertionError, match="source_used"):
            assert_head_experiment_manifest_integrity(df, _image_manifest())

    def test_missing_detector_fingerprint_column_fails(self):
        df = self._manifest([_head_crop_record()])
        df = df.drop(columns=["detector_fingerprint"])
        with pytest.raises(AssertionError, match="detector_fingerprint"):
            assert_head_experiment_manifest_integrity(df, _image_manifest())

    def test_non_head_row_fails(self):
        body_rec = dict(_head_crop_record())
        body_rec["crop_kind"] = "body"
        body_rec["crop_id"] = "img1__body_0"
        body_rec["crop_ordinal"] = 0
        df = pd.DataFrame([body_rec], columns=HEAD_EXPERIMENT_MANIFEST_COLUMNS)
        with pytest.raises(AssertionError, match="non-head"):
            assert_head_experiment_manifest_integrity(df, _image_manifest())

    def test_detector_fingerprint_mismatch_fails(self):
        assert_head_experiment_manifest_integrity(
            self._manifest([_head_crop_record(detector_fingerprint="fp_A")]),
            _image_manifest(),
            expected_detector_fingerprint="fp_A",
        )
        with pytest.raises(AssertionError, match="detector_fingerprint"):
            assert_head_experiment_manifest_integrity(
                self._manifest([_head_crop_record(detector_fingerprint="fp_A")]),
                _image_manifest(),
                expected_detector_fingerprint="fp_B",
            )


# ---------------------------------------------------------------------------
# 3. EarDetector behaviour unchanged after refactor
# ---------------------------------------------------------------------------

class TestEarDetectorUnchanged:
    image = np.full((100, 100, 3), 127, dtype=np.uint8)

    def test_detect_ears_zero_returns_empty(self):
        assert _ear_detector().detect_ears(self.image) == []

    def test_detect_ears_one_ear(self):
        ears = _ear_detector([[10, 20, 40, 60]], [0.9]).detect_ears(self.image)
        assert len(ears) == 1
        assert ears[0]["ordinal"] == 0

    def test_detect_ears_two_ears_ordered_left_to_right(self):
        ears = _ear_detector(
            [[60, 20, 90, 60], [10, 20, 40, 60]], [0.9, 0.8]
        ).detect_ears(self.image)
        assert [e["box"][0] for e in ears] == [10, 60]
        assert [e["ordinal"] for e in ears] == [0, 1]

    def test_detect_ears_area_filter(self):
        assert _ear_detector([[0, 0, 100, 100]], [0.9]).detect_ears(self.image) == []

    def test_detect_ears_nms_suppresses_overlap(self):
        ears = _ear_detector(
            [[10, 10, 50, 50], [12, 12, 52, 52]], [0.9, 0.8]
        ).detect_ears(self.image)
        assert len(ears) == 1

    def test_detect_ear_backward_compat_returns_ndarray(self):
        crop = _ear_detector([[10, 20, 40, 60]], [0.9]).detect_ear(self.image)
        assert isinstance(crop, np.ndarray) and crop.size > 0

    def test_require_available_raises(self):
        det = _ear_detector()
        det._available = False
        with pytest.raises(RuntimeError, match="unavailable"):
            det.detect_ears(self.image, require_available=True)

    def test_ear_detector_accepts_backend_kwarg(self):
        """EarDetector(backend=...) constructor path works."""
        backend = _GroundingDINOBackend.__new__(_GroundingDINOBackend)
        backend.device = "cpu"
        backend._available = True
        backend.processor = _FakeProcessor([[10, 20, 40, 60]], [0.9])
        backend.model = _FakeModel()
        det = EarDetector(backend=backend)
        assert det._available is True
        assert det.processor is backend.processor


# ---------------------------------------------------------------------------
# 4. Shared backend loaded once
# ---------------------------------------------------------------------------

class TestSharedBackend:
    def test_shared_backend_processor_is_same_object(self):
        backend = _GroundingDINOBackend.__new__(_GroundingDINOBackend)
        backend.device = "cpu"
        backend._available = True
        backend.processor = _FakeProcessor([], [])
        backend.model = _FakeModel()

        ear_det = EarDetector(backend=backend)
        head_det = HeadDetector(backend=backend)

        assert ear_det.processor is backend.processor
        assert head_det.processor is backend.processor
        assert ear_det.model is backend.model
        assert head_det.model is backend.model

    def test_shared_backend_model_is_same_object(self):
        backend = _GroundingDINOBackend.__new__(_GroundingDINOBackend)
        backend.device = "cpu"
        backend._available = True
        backend.processor = _FakeProcessor([], [])
        fake_model = _FakeModel()
        backend.model = fake_model

        ear_det = EarDetector(backend=backend)
        head_det = HeadDetector(backend=backend)

        assert ear_det.model is fake_model
        assert head_det.model is fake_model

    def test_unavailable_backend_propagates_to_both_detectors(self):
        backend = _GroundingDINOBackend.__new__(_GroundingDINOBackend)
        backend.device = "cpu"
        backend._available = False
        backend.processor = None
        backend.model = None

        ear_det = EarDetector(backend=backend)
        head_det = HeadDetector(backend=backend)

        assert ear_det._available is False
        assert head_det._available is False


# ---------------------------------------------------------------------------
# 5. HeadDetector — detection constraints and determinism
# ---------------------------------------------------------------------------

class TestHeadDetector:
    image = np.full((100, 100, 3), 127, dtype=np.uint8)

    def test_no_boxes_returns_none(self):
        assert _head_detector().detect_head(self.image) is None

    def test_valid_box_returns_dict(self):
        result = _head_detector([[20, 20, 60, 60]], [0.8]).detect_head(self.image)
        assert result is not None
        assert result["ordinal"] == 0
        assert result["source"] == "grounding_dino"
        assert isinstance(result["crop"], np.ndarray)
        assert result["crop"].size > 0

    def test_returns_at_most_one_result(self):
        # Two non-overlapping boxes → NMS keeps both, but head returns only 1.
        result = _head_detector(
            [[5, 5, 30, 30], [60, 60, 90, 90]], [0.7, 0.85]
        ).detect_head(self.image)
        assert result is not None
        assert result["ordinal"] == 0

    def test_returns_highest_score_after_nms(self):
        # Two non-overlapping boxes; lower score first in list.
        result = _head_detector(
            [[5, 5, 30, 30], [60, 60, 90, 90]], [0.70, 0.85]
        ).detect_head(self.image)
        assert result is not None
        assert abs(result["score"] - 0.85) < 1e-4

    def test_nms_suppresses_near_duplicate(self):
        # Two highly-overlapping boxes.
        result = _head_detector(
            [[10, 10, 50, 50], [12, 12, 52, 52]], [0.9, 0.8]
        ).detect_head(self.image)
        assert result is not None
        assert abs(result["score"] - 0.9) < 1e-4

    def test_area_filter_too_small(self):
        # 2×2 box on a 100×100 image → area_frac = 0.0004 < min_area_frac (0.02).
        result = _head_detector([[10, 10, 12, 12]], [0.9]).detect_head(self.image)
        assert result is None

    def test_area_filter_too_large(self):
        # Full image → area_frac = 1.0 > max_area_frac (0.70).
        result = _head_detector([[0, 0, 100, 100]], [0.9]).detect_head(self.image)
        assert result is None

    def test_aspect_filter_too_wide(self):
        # 80×10 → aspect = 8.0 > max_aspect (2.5).
        result = _head_detector([[10, 45, 90, 55]], [0.9]).detect_head(self.image)
        assert result is None

    def test_aspect_filter_too_tall(self):
        # 10×80 → aspect = 0.125 < min_aspect (0.4).
        result = _head_detector([[45, 10, 55, 90]], [0.9]).detect_head(self.image)
        assert result is None

    def test_require_available_raises_when_unavailable(self):
        det = _head_detector()
        det._available = False
        with pytest.raises(RuntimeError, match="unavailable"):
            det.detect_head(self.image, require_available=True)

    def test_unavailable_returns_none_by_default(self):
        det = _head_detector()
        det._available = False
        assert det.detect_head(self.image) is None

    def test_empty_image_returns_none(self):
        assert _head_detector([[10, 10, 50, 50]], [0.9]).detect_head(
            np.zeros((0, 0, 3), dtype=np.uint8)
        ) is None

    def test_default_constraints_differ_from_ear(self):
        """Head defaults must be independently configurable from ear defaults."""
        assert HeadDetector.DEFAULT_CONF_THRESHOLD != EarDetector.__init__.__defaults__[0]  # noqa
        # Simpler: just check the documented defaults.
        assert HeadDetector.DEFAULT_MIN_AREA_FRAC == 0.02
        assert HeadDetector.DEFAULT_MAX_AREA_FRAC == 0.70
        assert HeadDetector.DEFAULT_PAD_FRAC == 0.10


# ---------------------------------------------------------------------------
# 6. Detector fingerprint
# ---------------------------------------------------------------------------

class TestDetectorFingerprint:
    def test_fingerprint_is_16_chars(self):
        fp = head_detector_fingerprint(0.30, 0.02, 0.70, 0.40, 2.50, 0.50, 0.10)
        assert len(fp) == 16

    def test_fingerprint_changes_with_threshold(self):
        fp1 = head_detector_fingerprint(0.30, 0.02, 0.70, 0.40, 2.50, 0.50, 0.10)
        fp2 = head_detector_fingerprint(0.35, 0.02, 0.70, 0.40, 2.50, 0.50, 0.10)
        assert fp1 != fp2

    def test_fingerprint_stable(self):
        fp1 = head_detector_fingerprint(0.30, 0.02, 0.70, 0.40, 2.50, 0.50, 0.10)
        fp2 = head_detector_fingerprint(0.30, 0.02, 0.70, 0.40, 2.50, 0.50, 0.10)
        assert fp1 == fp2


# ---------------------------------------------------------------------------
# 7. run_head_detection pipeline
# ---------------------------------------------------------------------------

class TestRunHeadDetection:
    def _run(
        self,
        tmp_path,
        *,
        image_id="img1",
        found=True,
        body_crop_exists=True,
        write_source=True,
        body_manifest_supplied=True,
    ):
        source = tmp_path / "source.jpg"
        if write_source:
            _write_source(source)

        body_crop = tmp_path / "body_crop.jpg"
        if body_crop_exists:
            _write_source(body_crop)

        body_manifest = None
        if body_manifest_supplied and body_crop_exists:
            body_manifest = _v1_body_manifest(body_crop, image_id=image_id)

        detector = _PipelineHeadDetector(found=found)
        return (
            run_head_detection(
                image_manifest=_detection_manifest(source, image_id=image_id),
                body_manifest=body_manifest,
                crops_dir=str(tmp_path / "head_crops"),
                manifest_path=str(tmp_path / "head_manifest.parquet"),
                head_detector=detector,
                crop_size=64,
            ),
            detector,
        )

    def test_accepted_head_from_body_crop(self, tmp_path):
        result, det = self._run(tmp_path)
        accepted = result[result["detector_status"] == "accepted"]
        assert len(accepted) == 1
        assert accepted.iloc[0]["crop_kind"] == "head"
        assert accepted.iloc[0]["source_used"] == "body_crop"

    def test_accepted_head_from_original_when_no_body_manifest(self, tmp_path):
        result, det = self._run(tmp_path, body_manifest_supplied=False)
        accepted = result[result["detector_status"] == "accepted"]
        assert len(accepted) == 1
        assert accepted.iloc[0]["source_used"] == "original"

    def test_body_first_fallback_to_original(self, tmp_path):
        """When body manifest exists but has no accepted body crop, use original."""
        source = tmp_path / "source.jpg"
        _write_source(source)
        # Supply a body manifest with none_detected status.
        body_manifest = pd.DataFrame({
            "crop_id": [make_crop_id("img1", "body", 0)],
            "image_id": ["img1"],
            "individual_id": ["elephant1"],
            "crop_kind": ["body"],
            "crop_ordinal": [0],
            "crop_path": [str(tmp_path / "nonexistent.jpg")],
            "detector_confidence": [None],
            "detector_box": [None],
            "detector_status": ["none_detected"],
            "review_status": ["pending"],
            "schema_version": [ARTIFACT_SCHEMA_VERSION],
            "source_fingerprint": ["s"],
            "split_fingerprint": ["p"],
        })
        detector = _PipelineHeadDetector(found=True)
        result = run_head_detection(
            image_manifest=_detection_manifest(source),
            body_manifest=body_manifest,
            crops_dir=str(tmp_path / "head_crops"),
            manifest_path=str(tmp_path / "head_manifest.parquet"),
            head_detector=detector,
            crop_size=64,
        )
        accepted = result[result["detector_status"] == "accepted"]
        assert len(accepted) == 1
        assert accepted.iloc[0]["source_used"] == "original"

    def test_none_detected_when_head_not_found(self, tmp_path):
        result, _ = self._run(tmp_path, found=False)
        assert list(result["detector_status"]) == ["none_detected"]

    def test_not_applicable_when_no_image_readable(self, tmp_path):
        # Don't write the source image; supply no body manifest.
        result, _ = self._run(
            tmp_path, write_source=False, body_crop_exists=False,
            body_manifest_supplied=False
        )
        assert list(result["detector_status"]) == ["not_applicable"]

    def test_result_has_detector_fingerprint(self, tmp_path):
        result, _ = self._run(tmp_path)
        assert "detector_fingerprint" in result.columns
        assert result.iloc[0]["detector_fingerprint"] is not None
        assert len(str(result.iloc[0]["detector_fingerprint"])) == 16

    def test_result_has_source_used(self, tmp_path):
        result, _ = self._run(tmp_path)
        assert "source_used" in result.columns

    def test_individual_id_propagated(self, tmp_path):
        result, _ = self._run(tmp_path)
        accepted = result[result["detector_status"] == "accepted"]
        assert all(accepted["individual_id"] == "elephant1")

    def test_crop_file_written_on_accepted(self, tmp_path):
        result, _ = self._run(tmp_path)
        accepted = result[result["detector_status"] == "accepted"]
        assert len(accepted) == 1
        crop_path = accepted.iloc[0]["crop_path"]
        assert os.path.isfile(str(crop_path))


# ---------------------------------------------------------------------------
# 8. Terminal-status resumability
# ---------------------------------------------------------------------------

class TestHeadResumability:
    def _kwargs(self, tmp_path, found=True, body_crop_exists=True):
        source = tmp_path / "source.jpg"
        _write_source(source)
        body_crop = tmp_path / "body.jpg"
        if body_crop_exists:
            _write_source(body_crop)
        bm = _v1_body_manifest(body_crop) if body_crop_exists else None
        return dict(
            image_manifest=_detection_manifest(source),
            body_manifest=bm,
            crops_dir=str(tmp_path / "head_crops"),
            manifest_path=str(tmp_path / "manifest.parquet"),
            head_detector=_PipelineHeadDetector(found=found),
            crop_size=64,
        )

    def test_accepted_idempotent(self, tmp_path):
        kw = self._kwargs(tmp_path)
        first = run_head_detection(**kw)
        second = run_head_detection(**kw)
        pd.testing.assert_frame_equal(first, second)

    def test_none_detected_idempotent(self, tmp_path):
        kw = self._kwargs(tmp_path, found=False)
        first = run_head_detection(**kw)
        assert list(first["detector_status"]) == ["none_detected"]
        second = run_head_detection(**kw)
        pd.testing.assert_frame_equal(first, second)

    def test_none_detected_does_not_accumulate_rows(self, tmp_path):
        kw = self._kwargs(tmp_path, found=False)
        first = run_head_detection(**kw)
        second = run_head_detection(**kw)
        assert len(second) == 1

    def test_not_applicable_idempotent(self, tmp_path):
        source = tmp_path / "missing_source.jpg"
        kw = dict(
            image_manifest=_detection_manifest(source),
            body_manifest=None,
            crops_dir=str(tmp_path / "head_crops"),
            manifest_path=str(tmp_path / "manifest.parquet"),
            head_detector=_PipelineHeadDetector(),
            crop_size=64,
        )
        first = run_head_detection(**kw)
        assert list(first["detector_status"]) == ["not_applicable"]
        second = run_head_detection(**kw)
        pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------------------------------------
# 9. Unavailable model fails pre-loop
# ---------------------------------------------------------------------------

class TestHeadUnavailableModel:
    def test_unavailable_raises_before_loop(self, tmp_path):
        source = tmp_path / "source.jpg"
        _write_source(source)
        det = _PipelineHeadDetector()
        det._available = False
        with pytest.raises(RuntimeError, match="unavailable"):
            run_head_detection(
                image_manifest=_detection_manifest(source),
                body_manifest=None,
                crops_dir=str(tmp_path / "crops"),
                manifest_path=str(tmp_path / "manifest.parquet"),
                head_detector=det,
            )

    def test_unavailable_writes_no_manifest(self, tmp_path):
        source = tmp_path / "source.jpg"
        _write_source(source)
        manifest_path = str(tmp_path / "manifest.parquet")
        det = _PipelineHeadDetector()
        det._available = False
        with pytest.raises(RuntimeError):
            run_head_detection(
                image_manifest=_detection_manifest(source),
                body_manifest=None,
                crops_dir=str(tmp_path / "crops"),
                manifest_path=manifest_path,
                head_detector=det,
            )
        assert not os.path.isfile(manifest_path), (
            "No head manifest should be written when model is unavailable"
        )


# ---------------------------------------------------------------------------
# 10. Production v1 artifact not mutated
# ---------------------------------------------------------------------------

class TestNoProductionMutation:
    def test_v1_body_manifest_not_modified(self, tmp_path):
        source = tmp_path / "source.jpg"
        _write_source(source)
        body_crop = tmp_path / "body.jpg"
        _write_source(body_crop)
        v1_manifest = _v1_body_manifest(body_crop)
        v1_parquet = tmp_path / "v1_crop_manifest.parquet"
        v1_manifest.to_parquet(str(v1_parquet), index=False)

        mtime_before = os.path.getmtime(str(v1_parquet))

        run_head_detection(
            image_manifest=_detection_manifest(source),
            body_manifest=pd.read_parquet(str(v1_parquet)),
            crops_dir=str(tmp_path / "head_crops"),
            manifest_path=str(tmp_path / "head_manifest.parquet"),
            head_detector=_PipelineHeadDetector(),
            crop_size=64,
        )

        mtime_after = os.path.getmtime(str(v1_parquet))
        assert mtime_before == mtime_after, (
            "Production v1 crop manifest was modified by head detection"
        )

    def test_head_manifest_separate_from_body_manifest(self, tmp_path):
        source = tmp_path / "source.jpg"
        _write_source(source)
        body_crop = tmp_path / "body.jpg"
        _write_source(body_crop)
        head_manifest_path = str(tmp_path / "head_manifest.parquet")

        run_head_detection(
            image_manifest=_detection_manifest(source),
            body_manifest=_v1_body_manifest(body_crop),
            crops_dir=str(tmp_path / "head_crops"),
            manifest_path=head_manifest_path,
            head_detector=_PipelineHeadDetector(),
            crop_size=64,
        )

        head_df = pd.read_parquet(head_manifest_path)
        assert all(head_df["crop_kind"] == "head")


# ---------------------------------------------------------------------------
# 11. Fingerprint / join checks
# ---------------------------------------------------------------------------

class TestFingerprintJoin:
    def test_source_fingerprint_stored_in_manifest(self, tmp_path):
        source = tmp_path / "source.jpg"
        _write_source(source)
        result = run_head_detection(
            image_manifest=_detection_manifest(source),
            body_manifest=None,
            crops_dir=str(tmp_path / "head_crops"),
            manifest_path=str(tmp_path / "manifest.parquet"),
            head_detector=_PipelineHeadDetector(),
            source_fingerprint="test_src_fp",
            split_fingerprint="test_spl_fp",
            crop_size=64,
        )
        assert all(result["source_fingerprint"] == "test_src_fp")
        assert all(result["split_fingerprint"] == "test_spl_fp")

    def test_schema_version_stored_in_manifest(self, tmp_path):
        source = tmp_path / "source.jpg"
        _write_source(source)
        result = run_head_detection(
            image_manifest=_detection_manifest(source),
            body_manifest=None,
            crops_dir=str(tmp_path / "head_crops"),
            manifest_path=str(tmp_path / "manifest.parquet"),
            head_detector=_PipelineHeadDetector(),
            crop_size=64,
        )
        assert all(result["schema_version"] == ARTIFACT_SCHEMA_VERSION)

    def test_detector_fingerprint_consistent_across_accepted_rows(self, tmp_path):
        sources = [tmp_path / f"s{i}.jpg" for i in range(3)]
        for s in sources:
            _write_source(s)
        manifest = pd.DataFrame({
            "image_id": [f"img{i}" for i in range(3)],
            "individual_id": ["e1", "e2", "e3"],
            "include_status": ["included"] * 3,
            "source_path": [str(s) for s in sources],
        })
        result = run_head_detection(
            image_manifest=manifest,
            body_manifest=None,
            crops_dir=str(tmp_path / "head_crops"),
            manifest_path=str(tmp_path / "manifest.parquet"),
            head_detector=_PipelineHeadDetector(),
            crop_size=64,
        )
        accepted = result[result["detector_status"] == "accepted"]
        fps = accepted["detector_fingerprint"].unique()
        assert len(fps) == 1

    def test_resume_rejects_changed_detector_fingerprint_before_write(self, tmp_path):
        source = tmp_path / "source.jpg"
        _write_source(source)
        manifest_path = tmp_path / "manifest.parquet"
        existing = _head_df([
            _head_crop_record(
                crop_path=str(tmp_path / "existing.jpg"),
                detector_fingerprint="old-fingerprint",
            )
        ])
        existing.to_parquet(manifest_path, index=False)
        before = manifest_path.read_bytes()

        with pytest.raises(ValueError, match="different detector parameters"):
            run_head_detection(
                image_manifest=_detection_manifest(source),
                body_manifest=None,
                crops_dir=str(tmp_path / "heads"),
                manifest_path=str(manifest_path),
                head_detector=_PipelineHeadDetector(),
                detector_fingerprint="new-fingerprint",
                crop_size=64,
            )
        assert manifest_path.read_bytes() == before

    def test_review_only_pilot_rows_are_not_processed(self, tmp_path):
        pilot_source = tmp_path / "pilot.jpg"
        review_source = tmp_path / "review.jpg"
        _write_source(pilot_source)
        _write_source(review_source)
        pilot_row = _detection_manifest(pilot_source).assign(_pilot_role="pilot")
        review_row = _detection_manifest(review_source).assign(
            image_id="review",
            individual_id="unresolved",
            _pilot_role="review",
        )
        result = run_head_detection(
            image_manifest=pd.concat([pilot_row, review_row], ignore_index=True),
            body_manifest=None,
            crops_dir=str(tmp_path / "heads"),
            manifest_path=str(tmp_path / "manifest.parquet"),
            head_detector=_PipelineHeadDetector(),
            crop_size=64,
        )
        assert set(result["image_id"]) == {"img1"}

    def test_crop_id_format_head(self, tmp_path):
        source = tmp_path / "source.jpg"
        _write_source(source)
        result = run_head_detection(
            image_manifest=_detection_manifest(source),
            body_manifest=None,
            crops_dir=str(tmp_path / "head_crops"),
            manifest_path=str(tmp_path / "manifest.parquet"),
            head_detector=_PipelineHeadDetector(),
            crop_size=64,
        )
        assert result.iloc[0]["crop_id"] == "img1__head_0"
