import sys
import importlib.util
from pathlib import Path

import cv2
import faiss
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_bteh import ARTIFACT_SCHEMA_VERSION
from models.detector import EarDetector, _iou, _nms
from models.fusion import WildFusionMatcher
from pipeline.step_4b_train_calibration import run_all_pairs
from pipeline.step_1_run_detection_to_crop import run_bteh_detection
from pipeline.step_2_create_embeddings import (
    build_faiss_index,
    embed_from_crop_manifest,
)
from utils.artifact_schema import (
    CROP_MANIFEST_COLUMNS,
    DESCRIPTOR_MAPPING_COLUMNS,
    assert_crop_manifest_integrity,
    assert_descriptor_mapping_integrity,
    assert_no_cross_artifact_contamination,
    fingerprint_dataframe_columns,
    make_crop_id,
)


def test_row_fingerprint_changes_with_assignment():
    first = pd.DataFrame(
        [{"image_id": "img1", "split": "gallery"}, {"image_id": "img2", "split": "probe"}]
    )
    second = first.copy()
    second.loc[second["image_id"] == "img1", "split"] = "probe"

    assert fingerprint_dataframe_columns(
        first, ["image_id", "split"], sort_by=["image_id"]
    ) != fingerprint_dataframe_columns(
        second, ["image_id", "split"], sort_by=["image_id"]
    )


def _image_manifest(image_ids=("img1",), individual_id="elephant1"):
    return pd.DataFrame({
        "image_id": list(image_ids),
        "individual_id": [individual_id] * len(image_ids),
    })


def _crop_record(image_id="img1", kind="body", ordinal=0, **overrides):
    record = {
        "crop_id": make_crop_id(image_id, kind, ordinal),
        "image_id": image_id,
        "individual_id": "elephant1",
        "crop_kind": kind,
        "crop_ordinal": ordinal,
        "crop_path": f"/data/{image_id}__{kind}_{ordinal}.jpg",
        "detector_confidence": 0.9,
        "detector_box": "[1,2,10,20]",
        "detector_status": "accepted",
        "review_status": "pending",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_fingerprint": None,
        "split_fingerprint": None,
    }
    record.update(overrides)
    return record


def _mapping(n=2, descriptor="megadescriptor"):
    rows = []
    for index in range(n):
        rows.append(
            {
                "descriptor_name": descriptor,
                "embedding_row": index,
                "faiss_row": index,
                "crop_id": f"crop{index}",
                "image_id": f"img{index}",
                "individual_id": f"elephant{index}",
                "crop_kind": "body",
                "crop_ordinal": 0,
                "crop_path": f"/data/crop{index}.jpg",
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_fingerprint": "source",
                "split_fingerprint": "split",
                "model_preprocess_fingerprint": "model",
            }
        )
    return pd.DataFrame(rows, columns=DESCRIPTOR_MAPPING_COLUMNS)


def _unit_matrix(n=2, dim=4):
    matrix = np.zeros((n, dim), dtype=np.float32)
    for index in range(n):
        matrix[index, index % dim] = 1.0
    return matrix


class TestCropManifestIntegrity:
    def _assert_valid(self, rows):
        assert_crop_manifest_integrity(
            pd.DataFrame(rows, columns=CROP_MANIFEST_COLUMNS), _image_manifest()
        )

    def test_zero_ears_valid(self):
        self._assert_valid([_crop_record()])

    def test_one_ear_valid(self):
        self._assert_valid([_crop_record(), _crop_record(kind="ear")])

    def test_two_ears_valid(self):
        self._assert_valid(
            [_crop_record(), _crop_record(kind="ear"), _crop_record(kind="ear", ordinal=1)]
        )

    def test_missing_ear_omitted(self):
        rows = [_crop_record(), _crop_record(kind="ear", ordinal=1)]
        self._assert_valid(rows)
        assert len(rows) == 2

    def test_three_ears_fails(self):
        rows = [
            _crop_record(kind="ear", ordinal=0),
            _crop_record(kind="ear", ordinal=1),
            _crop_record(
                kind="ear", ordinal=1, crop_id="img1__ear_duplicate"
            ),
        ]
        with pytest.raises(AssertionError, match="cardinality"):
            self._assert_valid(rows)

    def test_two_bodies_fails(self):
        rows = [_crop_record(), _crop_record(crop_id="img1__body_duplicate")]
        with pytest.raises(AssertionError, match="cardinality"):
            self._assert_valid(rows)

    def test_duplicate_crop_id_fails(self):
        with pytest.raises(AssertionError, match="unique"):
            self._assert_valid([_crop_record(), _crop_record()])

    def test_missing_image_id_fails(self):
        with pytest.raises(AssertionError, match="missing from image manifest"):
            assert_crop_manifest_integrity(
                pd.DataFrame([_crop_record(image_id="unknown")]), _image_manifest()
            )

    def test_wrong_crop_kind_fails(self):
        with pytest.raises(AssertionError, match="crop_kind"):
            self._assert_valid([_crop_record(crop_kind="tail")])

    def test_schema_version_mismatch_fails(self):
        with pytest.raises(AssertionError, match="schema_version"):
            self._assert_valid([_crop_record(schema_version="v0")])


class TestDescriptorMappingIntegrity:
    def _assert_valid(self, mapping=None, matrix=None, index=None, **kwargs):
        mapping = _mapping() if mapping is None else mapping
        matrix = _unit_matrix() if matrix is None else matrix
        assert_descriptor_mapping_integrity(
            mapping,
            matrix,
            index,
            is_reference=True,
            **kwargs,
        )

    def test_valid_mapping_passes(self):
        self._assert_valid()

    def test_duplicate_crop_id_fails(self):
        mapping = _mapping()
        mapping.loc[1, "crop_id"] = mapping.loc[0, "crop_id"]
        with pytest.raises(AssertionError, match="duplicate"):
            self._assert_valid(mapping=mapping)

    def test_wrong_individual_id_fails(self):
        mapping = _mapping()
        mapping.loc[1, ["crop_id", "descriptor_name"]] = ["crop0", "miewid"]
        with pytest.raises(AssertionError, match="individual_id"):
            self._assert_valid(mapping=mapping)

    def test_noncontiguous_embedding_rows_fails(self):
        mapping = _mapping()
        mapping["embedding_row"] = [0, 2]
        with pytest.raises(AssertionError, match="contiguous"):
            self._assert_valid(mapping=mapping)

    def test_row_count_mismatch_fails(self):
        with pytest.raises(AssertionError, match="row count"):
            self._assert_valid(matrix=_unit_matrix(1))

    def test_nan_vector_fails(self):
        matrix = _unit_matrix()
        matrix[0, 0] = np.nan
        with pytest.raises(AssertionError, match="non-finite"):
            self._assert_valid(matrix=matrix)

    def test_zero_norm_vector_fails(self):
        matrix = _unit_matrix()
        matrix[0] = 0
        with pytest.raises(AssertionError, match="zero-norm"):
            self._assert_valid(matrix=matrix)

    def test_non_normalized_vector_fails(self):
        matrix = _unit_matrix()
        matrix[0, 0] = 2
        with pytest.raises(AssertionError, match="not L2-normalized"):
            self._assert_valid(matrix=matrix)

    def test_faiss_ntotal_mismatch_fails(self):
        with pytest.raises(AssertionError, match="FAISS"):
            self._assert_valid(index=build_faiss_index(_unit_matrix(1)))

    def test_mismatched_fingerprint_fails(self):
        with pytest.raises(AssertionError, match="source_fingerprint"):
            self._assert_valid(expected_source_fingerprint="other")

    def test_faiss_mapping_roundtrip(self):
        matrix = _unit_matrix()
        index = build_faiss_index(matrix)
        mapping = _mapping()
        self._assert_valid(mapping=mapping, matrix=matrix, index=index)
        _, rows = index.search(matrix[[1]], 1)
        assert mapping.iloc[rows[0, 0]]["image_id"] == "img1"


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


def _ear_detector(boxes=(), scores=()):
    detector = EarDetector.__new__(EarDetector)
    detector._available = True
    detector.device = "cpu"
    detector.processor = _FakeProcessor(boxes, scores)
    detector.model = _FakeModel()
    return detector


class TestEarDetector:
    image = np.full((100, 100, 3), 127, dtype=np.uint8)

    def test_detect_ears_zero_ears_returns_empty(self):
        assert _ear_detector().detect_ears(self.image) == []

    def test_detect_ears_one_ear(self):
        ears = _ear_detector([[10, 20, 40, 60]], [0.9]).detect_ears(self.image)
        assert len(ears) == 1
        assert ears[0]["ordinal"] == 0

    def test_detect_ears_two_ears_ordered_by_x(self):
        ears = _ear_detector(
            [[60, 20, 90, 60], [10, 20, 40, 60]], [0.9, 0.8]
        ).detect_ears(self.image)
        assert [ear["box"][0] for ear in ears] == [10, 60]
        assert [ear["ordinal"] for ear in ears] == [0, 1]

    def test_detect_ears_nms_suppresses_overlap(self):
        ears = _ear_detector(
            [[10, 10, 50, 50], [12, 12, 52, 52]], [0.9, 0.8]
        ).detect_ears(self.image)
        assert len(ears) == 1

    def test_detect_ears_area_filter(self):
        ears = _ear_detector([[0, 0, 100, 100]], [0.9]).detect_ears(self.image)
        assert ears == []

    def test_detect_ears_require_available_raises(self):
        detector = _ear_detector()
        detector._available = False
        with pytest.raises(RuntimeError, match="unavailable"):
            detector.detect_ears(self.image, require_available=True)

    def test_detect_ear_backward_compat(self):
        crop = _ear_detector([[10, 20, 40, 60]], [0.9]).detect_ear(self.image)
        assert isinstance(crop, np.ndarray)
        assert crop.size > 0


class _BodyDetector:
    def __init__(self, found=True):
        self.found = found

    def crop(self, image):
        return (image[10:90, 10:90].copy(), "left") if self.found else (None, "unknown")


class _PipelineEarDetector:
    def __init__(self, count):
        self.count = count
        self.inputs = []
        self._available = True

    def detect_ears(self, image, require_available=False):
        self.inputs.append(image.copy())
        detections = []
        boxes = [[5, 5, 25, 30], [35, 5, 55, 30]]
        for ordinal in range(self.count):
            detections.append(
                {
                    "box": boxes[ordinal],
                    "score": 0.9 - ordinal * 0.1,
                    "ordinal": ordinal,
                    "crop": image[5:30, boxes[ordinal][0] : boxes[ordinal][2]].copy(),
                }
            )
        return detections


def _detection_manifest(path, image_id="img1", individual_id="elephant1"):
    return pd.DataFrame(
        {
            "image_id": [image_id],
            "individual_id": [individual_id],
            "include_status": ["included"],
            "source_path": [str(path)],
        }
    )


def _write_source(path):
    image = np.full((100, 100, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return image


class TestMultiEarCrops:
    @pytest.fixture(autouse=True)
    def _parquet_engine_fallback(self, monkeypatch):
        if importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet"):
            return
        monkeypatch.setattr(
            pd.DataFrame,
            "to_parquet",
            lambda frame, path, index=False: frame.to_pickle(path),
        )
        monkeypatch.setattr(pd, "read_parquet", pd.read_pickle)

    def _run(self, tmp_path, body=True, ears=0):
        source = tmp_path / "source.jpg"
        _write_source(source)
        return run_bteh_detection(
            _detection_manifest(source),
            str(tmp_path / "crops"),
            str(tmp_path / "crop_manifest.parquet"),
            _BodyDetector(body),
            _PipelineEarDetector(ears),
            crop_size=64,
        )

    def test_bteh_detection_no_ears(self, tmp_path):
        result = self._run(tmp_path)
        accepted = result[result["detector_status"] == "accepted"]
        assert list(accepted["crop_kind"]) == ["body"]

    def test_bteh_detection_two_ears(self, tmp_path):
        result = self._run(tmp_path, ears=2)
        accepted = result[result["detector_status"] == "accepted"]
        assert len(accepted) == 3
        assert list(accepted[accepted["crop_kind"] == "ear"]["crop_ordinal"]) == [0, 1]

    def test_bteh_detection_missing_image_id_raises(self, tmp_path):
        with pytest.raises(ValueError, match="image_id"):
            run_bteh_detection(
                pd.DataFrame({"include_status": ["included"]}),
                str(tmp_path / "crops"),
                str(tmp_path / "manifest.parquet"),
                _BodyDetector(),
                _PipelineEarDetector(0),
            )

    def test_bteh_detection_missing_individual_id_raises(self, tmp_path):
        """Blocker 2: manifest without individual_id must be rejected."""
        source = tmp_path / "source.jpg"
        _write_source(source)
        manifest_no_id = pd.DataFrame({
            "image_id": ["img1"],
            "include_status": ["included"],
            "source_path": [str(source)],
        })
        with pytest.raises(ValueError, match="individual_id"):
            run_bteh_detection(
                manifest_no_id,
                str(tmp_path / "crops"),
                str(tmp_path / "manifest.parquet"),
                _BodyDetector(),
                _PipelineEarDetector(0),
            )

    def test_bteh_detection_idempotent(self, tmp_path):
        source = tmp_path / "source.jpg"
        _write_source(source)
        kwargs = dict(
            image_manifest=_detection_manifest(source),
            crops_dir=str(tmp_path / "crops"),
            manifest_path=str(tmp_path / "manifest.parquet"),
            detector=_BodyDetector(),
            ear_detector=_PipelineEarDetector(2),
        )
        first = run_bteh_detection(**kwargs)
        second = run_bteh_detection(**kwargs)
        pd.testing.assert_frame_equal(first, second)

    def test_zero_ear_idempotent(self, tmp_path):
        """Blocker 3: image with 0 ears is skipped on resume without accumulating rows."""
        source = tmp_path / "source.jpg"
        _write_source(source)
        kwargs = dict(
            image_manifest=_detection_manifest(source),
            crops_dir=str(tmp_path / "crops"),
            manifest_path=str(tmp_path / "manifest.parquet"),
            detector=_BodyDetector(),
            ear_detector=_PipelineEarDetector(0),
        )
        first = run_bteh_detection(**kwargs)
        # ear_0 must be none_detected, ear_1 must be not_applicable
        ear_rows = first[first["crop_kind"] == "ear"].sort_values("crop_ordinal")
        assert list(ear_rows["detector_status"]) == ["none_detected", "not_applicable"]
        second = run_bteh_detection(**kwargs)
        pd.testing.assert_frame_equal(first, second)

    def test_one_ear_idempotent(self, tmp_path):
        """Blocker 3: image with 1 ear is skipped on resume; ear_1 gets none_detected."""
        source = tmp_path / "source.jpg"
        _write_source(source)
        kwargs = dict(
            image_manifest=_detection_manifest(source),
            crops_dir=str(tmp_path / "crops"),
            manifest_path=str(tmp_path / "manifest.parquet"),
            detector=_BodyDetector(),
            ear_detector=_PipelineEarDetector(1),
        )
        first = run_bteh_detection(**kwargs)
        ear_rows = first[first["crop_kind"] == "ear"].sort_values("crop_ordinal")
        statuses = list(ear_rows["detector_status"])
        assert statuses[0] == "accepted"
        assert statuses[1] == "none_detected"
        second = run_bteh_detection(**kwargs)
        pd.testing.assert_frame_equal(first, second)

    def test_two_ear_idempotent(self, tmp_path):
        """Blocker 3: image with 2 ears is skipped on resume."""
        source = tmp_path / "source.jpg"
        _write_source(source)
        kwargs = dict(
            image_manifest=_detection_manifest(source),
            crops_dir=str(tmp_path / "crops"),
            manifest_path=str(tmp_path / "manifest.parquet"),
            detector=_BodyDetector(),
            ear_detector=_PipelineEarDetector(2),
        )
        first = run_bteh_detection(**kwargs)
        ear_rows = first[first["crop_kind"] == "ear"]
        assert list(ear_rows["detector_status"]) == ["accepted", "accepted"]
        second = run_bteh_detection(**kwargs)
        pd.testing.assert_frame_equal(first, second)

    def test_unavailable_ear_model_fails_early(self, tmp_path):
        """Blocker 4: unavailable ear model must raise before the processing loop."""
        source = tmp_path / "source.jpg"
        _write_source(source)
        unavailable = _PipelineEarDetector(0)
        unavailable._available = False
        with pytest.raises(RuntimeError, match="EarDetector model is unavailable"):
            run_bteh_detection(
                _detection_manifest(source),
                str(tmp_path / "crops"),
                str(tmp_path / "manifest.parquet"),
                _BodyDetector(),
                unavailable,
            )

    def test_unavailable_ear_model_writes_no_records(self, tmp_path):
        """Blocker 4: no partial records written when ear model is unavailable."""
        source = tmp_path / "source.jpg"
        _write_source(source)
        manifest_path = str(tmp_path / "manifest.parquet")
        unavailable = _PipelineEarDetector(0)
        unavailable._available = False
        with pytest.raises(RuntimeError):
            run_bteh_detection(
                _detection_manifest(source),
                str(tmp_path / "crops"),
                manifest_path,
                _BodyDetector(),
                unavailable,
            )
        import os
        assert not os.path.isfile(manifest_path), "No crop manifest should be written on early failure"

    def test_ear_retry_on_original_when_body_fails(self, tmp_path):
        source = tmp_path / "source.jpg"
        original = _write_source(source)
        ears = _PipelineEarDetector(1)
        result = run_bteh_detection(
            _detection_manifest(source),
            str(tmp_path / "crops"),
            str(tmp_path / "manifest.parquet"),
            _BodyDetector(False),
            ears,
        )
        accepted = result[result["detector_status"] == "accepted"]
        assert len(accepted) == 1
        assert accepted.iloc[0]["crop_kind"] == "ear"
        assert ears.inputs[0].shape == original.shape

    def test_ear_ordinal_deterministic(self, tmp_path):
        result = self._run(tmp_path, ears=2)
        ear_rows = result[result["crop_kind"] == "ear"]
        assert list(ear_rows["crop_id"]) == ["img1__ear_0", "img1__ear_1"]

    def test_individual_id_propagated_from_manifest(self, tmp_path):
        """Blocker 2: individual_id from image manifest must appear in crop records."""
        source = tmp_path / "source.jpg"
        _write_source(source)
        result = run_bteh_detection(
            _detection_manifest(source, individual_id="bteh_nellie"),
            str(tmp_path / "crops"),
            str(tmp_path / "manifest.parquet"),
            _BodyDetector(),
            _PipelineEarDetector(1),
        )
        accepted = result[result["detector_status"] == "accepted"]
        assert all(accepted["individual_id"] == "bteh_nellie")


class TestIndividualIdEnforcement:
    """Blocker 2: individual_id must be enforced at schema and embedding level."""

    def test_accepted_crop_empty_individual_id_fails_schema(self):
        """assert_crop_manifest_integrity rejects accepted crops with empty individual_id."""
        record = _crop_record(individual_id="")
        with pytest.raises(AssertionError, match="individual_id"):
            assert_crop_manifest_integrity(
                pd.DataFrame([record], columns=CROP_MANIFEST_COLUMNS),
                _image_manifest(),
            )

    def test_individual_id_mismatch_with_image_manifest_fails(self):
        """Crop individual_id disagreeing with image manifest fails integrity check."""
        manifest = _image_manifest(image_ids=["img1"], individual_id="elephant_correct")
        record = _crop_record(individual_id="elephant_wrong")
        with pytest.raises(AssertionError, match="individual_id"):
            assert_crop_manifest_integrity(
                pd.DataFrame([record], columns=CROP_MANIFEST_COLUMNS),
                manifest,
            )

    def test_non_accepted_crop_no_individual_id_ok(self):
        """Non-accepted records are not checked for individual_id by the schema."""
        record = _crop_record(individual_id="", detector_status="none_detected")
        # none_detected record — ordinal check: body is ordinal 0 which is allowed
        # No AssertionError should be raised (no accepted crops to validate)
        assert_crop_manifest_integrity(
            pd.DataFrame([record], columns=CROP_MANIFEST_COLUMNS),
            _image_manifest(),
        )

    def test_embed_from_crop_manifest_raises_on_empty_individual_id(self, tmp_path):
        """Blocker 2: embed_from_crop_manifest rejects crops with empty individual_id."""
        from pipeline.step_2_create_embeddings import embed_from_crop_manifest

        path = tmp_path / "body.jpg"
        assert cv2.imwrite(str(path), np.full((20, 20, 3), 128, np.uint8))
        records = [{
            "crop_id": "img1__body_0",
            "image_id": "img1",
            "individual_id": "",
            "crop_kind": "body",
            "crop_ordinal": 0,
            "crop_path": str(path),
            "detector_status": "accepted",
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "source_fingerprint": None,
            "split_fingerprint": None,
        }]
        crops = pd.DataFrame(records)
        with pytest.raises(ValueError, match="individual_id"):
            embed_from_crop_manifest(crops, FakeEmbedder(), "megadescriptor")

    def test_descriptor_individual_id_matches_crop(self, tmp_path):
        """Regression: descriptor mapping individual_id equals the source crop's individual_id."""
        from pipeline.step_2_create_embeddings import embed_from_crop_manifest

        path = tmp_path / "body.jpg"
        assert cv2.imwrite(str(path), np.full((20, 20, 3), 128, np.uint8))
        records = [{
            "crop_id": "img1__body_0",
            "image_id": "img1",
            "individual_id": "bteh_amara",
            "crop_kind": "body",
            "crop_ordinal": 0,
            "crop_path": str(path),
            "detector_status": "accepted",
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "source_fingerprint": None,
            "split_fingerprint": None,
        }]
        crops = pd.DataFrame(records)
        mapping, _ = embed_from_crop_manifest(crops, FakeEmbedder(), "megadescriptor")
        assert mapping.iloc[0]["individual_id"] == "bteh_amara"


class FakeEmbedder:
    def __init__(self, dim=128):
        self.dim = dim
        self.backend = "fake"

    def embed_batch(self, images, batch_size=32):
        rng = np.random.default_rng(42)
        matrix = rng.random((len(images), self.dim)).astype(np.float32)
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def _embedding_crop_manifest(tmp_path, kinds):
    records = []
    for ordinal, kind in enumerate(kinds):
        crop_ordinal = ordinal if kind == "ear" else 0
        path = tmp_path / f"{kind}_{ordinal}.jpg"
        assert cv2.imwrite(str(path), np.full((20, 20, 3), ordinal + 1, np.uint8))
        records.append(
            _crop_record(
                image_id="img1",
                kind=kind,
                ordinal=crop_ordinal,
                crop_path=str(path),
                individual_id="elephant1",
            )
        )
    return pd.DataFrame(records)


class TestSparseEmbeddings:
    def test_embed_from_crop_manifest_no_ears_produces_no_rows(self, tmp_path):
        crops = _embedding_crop_manifest(tmp_path, ["body"])
        mapping, matrix = embed_from_crop_manifest(
            crops, FakeEmbedder(), "ear_megadescriptor", is_ear=True
        )
        assert mapping.empty
        assert matrix.shape == (0, 128)

    def test_embed_from_crop_manifest_body_only(self, tmp_path):
        mapping, matrix = embed_from_crop_manifest(
            _embedding_crop_manifest(tmp_path, ["body"]),
            FakeEmbedder(),
            "megadescriptor",
        )
        assert len(mapping) == len(matrix) == 1

    def test_embed_from_crop_manifest_two_ears(self, tmp_path):
        mapping, matrix = embed_from_crop_manifest(
            _embedding_crop_manifest(tmp_path, ["ear", "ear"]),
            FakeEmbedder(),
            "ear_megadescriptor",
            is_ear=True,
        )
        assert len(mapping) == len(matrix) == 2

    def test_no_zero_vectors_in_matrix(self, tmp_path):
        _, matrix = embed_from_crop_manifest(
            _embedding_crop_manifest(tmp_path, ["ear", "ear"]),
            FakeEmbedder(),
            "ear_megadescriptor",
            is_ear=True,
        )
        assert np.all(np.linalg.norm(matrix, axis=1) > 0)

    def test_descriptor_specific_query_rows(self, tmp_path):
        crops = _embedding_crop_manifest(tmp_path, ["body", "ear"])
        body, _ = embed_from_crop_manifest(crops, FakeEmbedder(), "megadescriptor")
        ear, _ = embed_from_crop_manifest(
            crops, FakeEmbedder(), "ear_megadescriptor", is_ear=True
        )
        assert list(body["embedding_row"]) == [0]
        assert list(ear["embedding_row"]) == [0]
        assert_no_cross_artifact_contamination(body, ear)

    def test_faiss_mapping_roundtrip_sparse(self, tmp_path):
        mapping, matrix = embed_from_crop_manifest(
            _embedding_crop_manifest(tmp_path, ["ear", "ear"]),
            FakeEmbedder(),
            "ear_megadescriptor",
            is_ear=True,
        )
        index = build_faiss_index(matrix)
        _, matches = index.search(matrix[[1]], 1)
        match = mapping.iloc[matches[0, 0]]
        assert match["image_id"] == "img1"
        assert match["crop_id"] == "img1__ear_1"


class TestNMSAndIOU:
    def test_iou_no_overlap(self):
        assert _iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0

    def test_iou_full_overlap(self):
        assert _iou([0, 0, 2, 2], [0, 0, 2, 2]) == 1

    def test_iou_partial_overlap(self):
        assert _iou([0, 0, 2, 2], [1, 1, 3, 3]) == pytest.approx(1 / 7)

    def test_nms_keeps_highest_score(self):
        assert _nms([[0, 0, 2, 2], [0, 0, 2, 2]], [0.5, 0.9], 0.5) == [1]

    def test_nms_keeps_both_when_no_overlap(self):
        assert _nms([[0, 0, 1, 1], [2, 2, 3, 3]], [0.5, 0.9], 0.5) == [1, 0]


class TestNormalizedMappingConsumers:
    def test_multi_ear_shortlist_keeps_best_pair(self):
        reference_matrix = np.asarray(
            [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32
        )
        reference_mapping = pd.DataFrame(
            {
                "faiss_row": [0, 1, 2],
                "individual_id": ["elephant1", "elephant1", "elephant2"],
                "image_id": ["ref1", "ref1", "ref2"],
                "crop_path": ["ear0.jpg", "ear1.jpg", "ear0.jpg"],
            }
        )
        matcher = WildFusionMatcher(
            embedders={"ear_megadescriptor": np.asarray([[1.0, 0.0], [0.0, 1.0]])},
            faiss_indexes={
                "ear_megadescriptor": build_faiss_index(reference_matrix)
            },
            ref_meta={"ear_megadescriptor": reference_mapping},
            local_matcher=None,
            calibrators={},
            shortlist_k=3,
        )
        candidates = matcher.shortlist_from_mappings(
            {
                "ear_megadescriptor": [
                    {"embedding_row": 0, "crop_id": "query__ear_0"},
                    {"embedding_row": 1, "crop_id": "query__ear_1"},
                ]
            }
        )
        by_image = {candidate["image_id"]: candidate for candidate in candidates}
        assert by_image["ref1"]["global_sims"]["ear_megadescriptor"] == pytest.approx(1.0)
        assert by_image["ref2"]["global_sims"]["ear_megadescriptor"] == pytest.approx(1.0)

    def test_calibration_uses_descriptor_local_rows(self):
        metadata = pd.DataFrame(
            {
                "image_id": ["img0", "img1"],
                "individual_id": ["elephant", "elephant"],
            }
        )
        mapping = pd.DataFrame(
            {
                "image_id": ["img1", "img0"],
                "individual_id": ["elephant", "elephant"],
                "embedding_row": [1, 0],
                "crop_id": ["crop1", "crop0"],
            }
        )
        matrices = {
            descriptor: None
            for descriptor in (
                "megadescriptor",
                "miewid",
                "ear_megadescriptor",
                "ear_miewid",
            )
        }
        matrices["megadescriptor"] = _unit_matrix(2)
        mappings = {"megadescriptor": mapping}
        desc_data, _ = run_all_pairs(
            metadata, matrices, descriptor_mappings=mappings
        )
        assert desc_data["megadescriptor"]["labels"] == [1]
