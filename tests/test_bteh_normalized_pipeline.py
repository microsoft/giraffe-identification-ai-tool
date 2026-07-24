# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""Integration tests: canonical image manifest → crop manifest → descriptor
artifacts → normalized step_3 query records/matching.

Uses fake detectors and embedders.  No model downloads or BTEH processing.
Covers all four audit findings:
  Blocker 1 – normalized BTEH step_3 CLI is end-to-end executable.
  Blocker 2 – individual_id flows image manifest → crop → descriptor mapping.
  Blocker 3 – terminal outcomes make 0/1/2-ear images resume-safe.
  Blocker 4 – unavailable ear model fails before the loop with no records.
"""

import importlib.util
import os
import sys
from pathlib import Path

import cv2
import faiss
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_bteh import ARTIFACT_SCHEMA_VERSION
from pipeline.step_1_run_detection_to_crop import run_bteh_detection
from pipeline.step_2_create_embeddings import (
    build_bteh_descriptor_artifacts,
    build_faiss_index,
    embed_from_crop_manifest,
)
from pipeline.step_3_run_initial_matching import (
    _load_bteh_reference,
    _build_bteh_query_records,
    _group_query_records_by_image,
    run_bteh_step3_normalized,
)
from utils.artifact_schema import (
    CROP_MANIFEST_COLUMNS,
    DESCRIPTOR_MAPPING_COLUMNS,
    assert_crop_manifest_integrity,
    assert_descriptor_mapping_integrity,
    make_crop_id,
)


# ---------------------------------------------------------------------------
# Fake detectors and embedder
# ---------------------------------------------------------------------------

class _FakeBodyDetector:
    def crop(self, image):
        h, w = image.shape[:2]
        return image[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].copy(), "unknown"


class _FakeEarDetector:
    """Returns a configurable number of fake ear detections."""

    def __init__(self, n_ears: int = 2):
        self.n_ears = n_ears
        self._available = True

    def detect_ears(self, image, require_available=False):
        if not self._available:
            if require_available:
                raise RuntimeError("EarDetector model is unavailable")
            return []
        h, w = image.shape[:2]
        boxes = [[0, 0, w // 3, h // 2], [w // 2, 0, w, h // 2]]
        detections = []
        for i in range(min(self.n_ears, 2)):
            x1, y1, x2, y2 = boxes[i]
            crop = image[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else image[:1, :1].copy()
            detections.append(
                {"box": boxes[i], "score": 0.9 - i * 0.1, "ordinal": i, "crop": crop}
            )
        return detections


class _FailOnceEarDetector(_FakeEarDetector):
    def __init__(self):
        super().__init__(n_ears=1)
        self.calls = 0

    def detect_ears(self, image, require_available=False):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient detector failure")
        return super().detect_ears(image, require_available=require_available)


class _FakeEmbedder:
    def __init__(self, dim: int = 16, seed: int = 0):
        self.dim = dim
        self.backend = "fake"
        self._seed = seed

    def embed_batch(self, images, batch_size=32):
        rng = np.random.default_rng(self._seed)
        mat = rng.random((len(images), self.dim)).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norms, 1e-8)


class _MeanSideEmbedder:
    dim = 2

    def embed_batch(self, images, batch_size=32):
        return np.asarray(
            [
                [float(image[:, : image.shape[1] // 2].mean()),
                 float(image[:, image.shape[1] // 2 :].mean())]
                for image in images
            ],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Fixtures
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


def _write_image(path: Path, value: int = 128) -> None:
    assert cv2.imwrite(str(path), np.full((64, 64, 3), value, dtype=np.uint8))


def _canonical_image_manifest(image_paths: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Build a minimal canonical image manifest.

    Each entry in image_paths is (image_id, individual_id, source_path).
    """
    records = []
    for image_id, individual_id, source_path in image_paths:
        records.append(
            {
                "image_id": image_id,
                "individual_id": individual_id,
                "include_status": "included",
                "source_path": source_path,
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Synthetic pipeline helper
# ---------------------------------------------------------------------------

def _build_synthetic_pipeline(
    tmp_path: Path,
    *,
    n_reference_images: int = 4,
    n_query_images: int = 2,
    n_ears_reference: int = 2,
    n_ears_query: int = 1,
    descriptor_dim: int = 16,
    descriptor_name: str = "fake_body",
) -> dict:
    """Run the full normalized BTEH pipeline on synthetic images.

    Returns a dict with keys: ref_artifact_dir, query_artifact_dir,
    reference_mapping, query_mapping, results_df.
    """
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    # --- Reference images ---
    ref_images = []
    for i in range(n_reference_images):
        path = source_dir / f"ref_{i}.jpg"
        _write_image(path, value=i * 10 + 20)
        individual_id = f"bteh_elephant_{i % 2}"  # 2 identities
        ref_images.append((f"ref_img_{i}", individual_id, str(path)))

    # --- Query images ---
    query_images = []
    for i in range(n_query_images):
        path = source_dir / f"query_{i}.jpg"
        _write_image(path, value=i * 10 + 120)
        individual_id = f"bteh_elephant_{i % 2}"
        query_images.append((f"query_img_{i}", individual_id, str(path)))

    ref_manifest = _canonical_image_manifest(ref_images)
    query_manifest = _canonical_image_manifest(query_images)

    # --- Step 1: crop detection ---
    ref_crops_dir  = tmp_path / "crops" / "reference"
    query_crops_dir = tmp_path / "crops" / "query"

    ref_crop_manifest_path   = str(tmp_path / "ref_crop_manifest.parquet")
    query_crop_manifest_path = str(tmp_path / "query_crop_manifest.parquet")

    ref_crop_df = run_bteh_detection(
        ref_manifest,
        str(ref_crops_dir),
        ref_crop_manifest_path,
        _FakeBodyDetector(),
        _FakeEarDetector(n_ears_reference),
        crop_size=32,
        schema_version=ARTIFACT_SCHEMA_VERSION,
    )
    query_crop_df = run_bteh_detection(
        query_manifest,
        str(query_crops_dir),
        query_crop_manifest_path,
        _FakeBodyDetector(),
        _FakeEarDetector(n_ears_query),
        crop_size=32,
        schema_version=ARTIFACT_SCHEMA_VERSION,
    )

    # --- Step 2: descriptor artifacts ---
    ref_artifact_dir   = str(tmp_path / "artifacts" / "reference")
    query_artifact_dir = str(tmp_path / "artifacts" / "query")

    embedder_factory = lambda _d: _FakeEmbedder(dim=descriptor_dim, seed=42)

    ref_mapping, _ref_matrix = build_bteh_descriptor_artifacts(
        crop_manifest=ref_crop_df,
        embedder_factory=embedder_factory,
        descriptor_name=descriptor_name,
        artifact_dir=ref_artifact_dir,
        is_reference=True,
        is_ear=False,
        schema_version=ARTIFACT_SCHEMA_VERSION,
    )
    query_mapping, _query_matrix = build_bteh_descriptor_artifacts(
        crop_manifest=query_crop_df,
        embedder_factory=embedder_factory,
        descriptor_name=descriptor_name,
        artifact_dir=query_artifact_dir,
        is_reference=False,
        is_ear=False,
        schema_version=ARTIFACT_SCHEMA_VERSION,
    )

    # --- Step 3: normalized matching ---
    output_path = str(tmp_path / "results.parquet")
    results_df = run_bteh_step3_normalized(
        query_artifact_dir=query_artifact_dir,
        reference_artifact_dir=ref_artifact_dir,
        output_path=output_path,
        descriptor_names=[descriptor_name],
        skip_local=True,
        schema_version=ARTIFACT_SCHEMA_VERSION,
    )

    return {
        "ref_artifact_dir": ref_artifact_dir,
        "query_artifact_dir": query_artifact_dir,
        "reference_mapping": ref_mapping,
        "query_mapping": query_mapping,
        "results_df": results_df,
        "output_path": output_path,
        "ref_crop_df": ref_crop_df,
        "query_crop_df": query_crop_df,
    }


# ---------------------------------------------------------------------------
# Blocker 1: Normalized BTEH step_3 is end-to-end executable
# ---------------------------------------------------------------------------

class TestNormalizedStep3EndToEnd:
    """Blocker 1: step_3 normalized route runs without pickle/megadescriptor_row."""

    def test_results_parquet_is_written(self, tmp_path):
        ctx = _build_synthetic_pipeline(tmp_path)
        assert os.path.isfile(ctx["output_path"])

    def test_one_row_per_query_image(self, tmp_path):
        ctx = _build_synthetic_pipeline(tmp_path, n_query_images=3)
        assert len(ctx["results_df"]) == 3

    def test_matching_status_present(self, tmp_path):
        ctx = _build_synthetic_pipeline(tmp_path)
        assert "matching_status" in ctx["results_df"].columns
        assert set(ctx["results_df"]["matching_status"]).issubset(
            {"matched", "not_matched"}
        )

    def test_no_pickle_metadata_written(self, tmp_path):
        """Reference artifacts must not contain legacy *_meta.pkl files."""
        ctx = _build_synthetic_pipeline(tmp_path)
        pkl_files = list(Path(ctx["ref_artifact_dir"]).glob("*_meta.pkl"))
        assert pkl_files == [], f"Legacy pickle files found: {pkl_files}"

    def test_no_shared_index_parquet(self, tmp_path):
        """Normalized route must not write or read a shared wide {desc}_row index parquet."""
        ctx = _build_synthetic_pipeline(tmp_path)
        # There should be no file matching the legacy pattern query_index.parquet or
        # reference_index.parquet (the wide index).
        for art_dir in (ctx["ref_artifact_dir"], ctx["query_artifact_dir"]):
            wide_idx = [
                f for f in Path(art_dir).glob("*index*.parquet")
                if "_mapping" not in f.name
            ]
            assert wide_idx == [], f"Wide index parquet found: {wide_idx}"

    def test_reference_artifacts_validated(self, tmp_path):
        """Reference mapping + matrix + FAISS index pass integrity assertions."""
        ctx = _build_synthetic_pipeline(tmp_path)
        ref_dir = ctx["ref_artifact_dir"]
        desc = "fake_body"
        mapping = pd.read_parquet(os.path.join(ref_dir, f"{desc}_mapping.parquet"))
        matrix = np.load(os.path.join(ref_dir, f"{desc}.npy"))
        index = faiss.read_index(os.path.join(ref_dir, f"{desc}.index"))
        assert_descriptor_mapping_integrity(
            mapping, matrix, index, is_reference=True,
            schema_version=ARTIFACT_SCHEMA_VERSION,
        )

    def test_query_records_built_per_image(self, tmp_path):
        """Each query image must produce exactly one descriptor record per accepted crop."""
        ctx = _build_synthetic_pipeline(tmp_path, n_query_images=2)
        desc = "fake_body"
        q_mapping = pd.read_parquet(
            os.path.join(ctx["query_artifact_dir"], f"{desc}_mapping.parquet")
        )
        q_matrix = np.load(os.path.join(ctx["query_artifact_dir"], f"{desc}.npy"))
        records = _build_bteh_query_records(q_mapping, q_matrix, desc)
        assert len(records) == 2
        for rec in records:
            assert "embedding" in rec
            assert rec["embedding"].shape == (16,)

    def test_no_filename_stem_ids(self, tmp_path):
        """image_id in results must match canonical IDs, not filename stems."""
        ctx = _build_synthetic_pipeline(tmp_path, n_query_images=2)
        result_ids = set(ctx["results_df"]["image_id"])
        assert "query_img_0" in result_ids
        assert "query_img_1" in result_ids


# ---------------------------------------------------------------------------
# Blocker 2: individual_id flows through every stage
# ---------------------------------------------------------------------------

class TestIndividualIdFlow:
    """Blocker 2: individual_id must be present and consistent at every stage."""

    def test_crop_manifest_carries_individual_id(self, tmp_path):
        ctx = _build_synthetic_pipeline(tmp_path)
        accepted = ctx["ref_crop_df"][ctx["ref_crop_df"]["detector_status"] == "accepted"]
        assert "individual_id" in accepted.columns
        assert not accepted["individual_id"].isna().any()
        assert not accepted["individual_id"].astype(str).str.strip().eq("").any()

    def test_descriptor_mapping_carries_individual_id(self, tmp_path):
        ctx = _build_synthetic_pipeline(tmp_path)
        ref_mapping = ctx["reference_mapping"]
        assert "individual_id" in ref_mapping.columns
        assert not ref_mapping["individual_id"].isna().any()
        assert not ref_mapping["individual_id"].astype(str).str.strip().eq("").any()

    def test_individual_id_consistent_with_source_manifest(self, tmp_path):
        """Descriptor individual_id must equal the image manifest entry's individual_id."""
        ctx = _build_synthetic_pipeline(tmp_path, n_reference_images=4)
        # Rebuild the reference manifest to verify consistency.
        source_dir = tmp_path / "source"
        ref_images = [
            (f"ref_img_{i}", f"bteh_elephant_{i % 2}", str(source_dir / f"ref_{i}.jpg"))
            for i in range(4)
        ]
        manifest_id_map = {img_id: ind_id for img_id, ind_id, _ in ref_images}
        ref_mapping = ctx["reference_mapping"]
        for _, row in ref_mapping.iterrows():
            assert str(row["individual_id"]) == manifest_id_map[str(row["image_id"])], (
                f"image_id={row['image_id']} has individual_id={row['individual_id']!r} "
                f"but manifest says {manifest_id_map[str(row['image_id'])]!r}"
            )

    def test_crop_individual_id_matches_image_manifest(self, tmp_path):
        """assert_crop_manifest_integrity passes with correct individual_id."""
        ctx = _build_synthetic_pipeline(tmp_path)
        source_dir = tmp_path / "source"
        ref_images = [
            (f"ref_img_{i}", f"bteh_elephant_{i % 2}", str(source_dir / f"ref_{i}.jpg"))
            for i in range(4)
        ]
        manifest = _canonical_image_manifest(ref_images)
        # Should not raise
        assert_crop_manifest_integrity(
            ctx["ref_crop_df"], manifest, schema_version=ARTIFACT_SCHEMA_VERSION
        )

    def test_build_descriptor_artifacts_requires_individual_id(self, tmp_path):
        """build_bteh_descriptor_artifacts rejects crops with empty individual_id."""
        path = tmp_path / "body.jpg"
        _write_image(path)
        crops = pd.DataFrame(
            [
                {
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
                }
            ]
        )
        with pytest.raises(ValueError, match="individual_id"):
            build_bteh_descriptor_artifacts(
                crop_manifest=crops,
                embedder_factory=lambda _: _FakeEmbedder(),
                descriptor_name="fake_body",
                artifact_dir=str(tmp_path / "arts"),
                is_reference=True,
            )

    def test_build_descriptor_rejects_relabelled_provenance(self, tmp_path):
        path = tmp_path / "body.jpg"
        _write_image(path)
        crops = pd.DataFrame(
            [
                {
                    "crop_id": "img1__body_0",
                    "image_id": "img1",
                    "individual_id": "elephant_1",
                    "crop_kind": "body",
                    "crop_ordinal": 0,
                    "crop_path": str(path),
                    "detector_status": "accepted",
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "source_fingerprint": "source-a",
                    "split_fingerprint": "split-a",
                }
            ]
        )
        with pytest.raises(ValueError, match="source_fingerprint mismatch"):
            build_bteh_descriptor_artifacts(
                crop_manifest=crops,
                embedder_factory=lambda _: _FakeEmbedder(),
                descriptor_name="fake_body",
                artifact_dir=str(tmp_path / "arts"),
                is_reference=True,
                source_fingerprint="source-b",
                split_fingerprint="split-a",
            )

    def test_matching_rejects_model_fingerprint_mismatch(self, tmp_path):
        ctx = _build_synthetic_pipeline(tmp_path)
        ref_path = (
            Path(ctx["ref_artifact_dir"]) / "fake_body_mapping.parquet"
        )
        query_path = (
            Path(ctx["query_artifact_dir"]) / "fake_body_mapping.parquet"
        )
        reference = pd.read_parquet(ref_path)
        query = pd.read_parquet(query_path)
        reference["model_preprocess_fingerprint"] = "model-reference"
        query["model_preprocess_fingerprint"] = "model-query"
        reference.to_parquet(ref_path, index=False)
        query.to_parquet(query_path, index=False)

        with pytest.raises(
            ValueError,
            match="model_preprocess_fingerprint mismatch",
        ):
            run_bteh_step3_normalized(
                query_artifact_dir=ctx["query_artifact_dir"],
                reference_artifact_dir=ctx["ref_artifact_dir"],
                output_path=str(tmp_path / "mismatch.parquet"),
                descriptor_names=["fake_body"],
                skip_local=True,
            )


# ---------------------------------------------------------------------------
# Blocker 3: terminal outcomes and resume
# ---------------------------------------------------------------------------

class TestTerminalOutcomesResume:
    """Blocker 3: 0-, 1-, and 2-ear images are resume-safe."""

    def _run_and_rerun(self, tmp_path, n_ears):
        source = tmp_path / "source.jpg"
        _write_image(source)
        manifest = _canonical_image_manifest(
            [("img1", "bteh_nellie", str(source))]
        )
        kwargs = dict(
            image_manifest=manifest,
            crops_dir=str(tmp_path / "crops"),
            manifest_path=str(tmp_path / "manifest.parquet"),
            detector=_FakeBodyDetector(),
            ear_detector=_FakeEarDetector(n_ears),
            crop_size=32,
        )
        first = run_bteh_detection(**kwargs)
        second = run_bteh_detection(**kwargs)
        return first, second

    def test_zero_ears_all_three_slots_terminal(self, tmp_path):
        """0-ear image writes terminal records for all three slots."""
        first, _ = self._run_and_rerun(tmp_path, 0)
        statuses = dict(zip(
            first["crop_id"].tolist(),
            first["detector_status"].tolist(),
        ))
        assert statuses.get("img1__body_0") == "accepted"
        assert statuses.get("img1__ear_0") == "none_detected"
        assert statuses.get("img1__ear_1") == "not_applicable"

    def test_zero_ears_idempotent(self, tmp_path):
        first, second = self._run_and_rerun(tmp_path, 0)
        pd.testing.assert_frame_equal(first, second)

    def test_one_ear_ear1_is_none_detected(self, tmp_path):
        """1-ear image: ear_0 accepted, ear_1 none_detected."""
        first, _ = self._run_and_rerun(tmp_path, 1)
        statuses = dict(zip(first["crop_id"].tolist(), first["detector_status"].tolist()))
        assert statuses["img1__ear_0"] == "accepted"
        assert statuses["img1__ear_1"] == "none_detected"

    def test_one_ear_idempotent(self, tmp_path):
        first, second = self._run_and_rerun(tmp_path, 1)
        pd.testing.assert_frame_equal(first, second)

    def test_two_ears_idempotent(self, tmp_path):
        first, second = self._run_and_rerun(tmp_path, 2)
        pd.testing.assert_frame_equal(first, second)

    def test_two_ears_both_accepted(self, tmp_path):
        first, _ = self._run_and_rerun(tmp_path, 2)
        ear_rows = first[first["crop_kind"] == "ear"]
        assert list(ear_rows["detector_status"]) == ["accepted", "accepted"]

    def test_rows_do_not_accumulate_on_reruns(self, tmp_path):
        """Re-running must not append duplicate rows."""
        first, second = self._run_and_rerun(tmp_path, 2)
        assert len(second) == len(first)

    def test_failed_ear_slot_is_retried(self, tmp_path):
        source = tmp_path / "source.jpg"
        _write_image(source)
        manifest = _canonical_image_manifest(
            [("img1", "bteh_nellie", str(source))]
        )
        ear_detector = _FailOnceEarDetector()
        kwargs = dict(
            image_manifest=manifest,
            crops_dir=str(tmp_path / "crops"),
            manifest_path=str(tmp_path / "manifest.parquet"),
            detector=_FakeBodyDetector(),
            ear_detector=ear_detector,
            crop_size=32,
        )

        first = run_bteh_detection(**kwargs)
        first_status = first.set_index("crop_id").loc["img1__ear_0", "detector_status"]
        assert first_status == "failed"

        second = run_bteh_detection(**kwargs)
        second_statuses = second.set_index("crop_id")["detector_status"].to_dict()
        assert second_statuses["img1__ear_0"] == "accepted"
        assert second_statuses["img1__ear_1"] == "none_detected"
        assert ear_detector.calls == 2

    def test_review_audit_rows_are_not_processed(self, tmp_path):
        pilot_path = tmp_path / "pilot.jpg"
        review_path = tmp_path / "review.jpg"
        _write_image(pilot_path)
        _write_image(review_path)
        manifest = _canonical_image_manifest(
            [
                ("pilot", "bteh_nellie", str(pilot_path)),
                ("review", "unresolved", str(review_path)),
            ]
        )
        manifest["_pilot_role"] = ["pilot", "review"]

        result = run_bteh_detection(
            image_manifest=manifest,
            crops_dir=str(tmp_path / "crops"),
            manifest_path=str(tmp_path / "manifest.parquet"),
            detector=_FakeBodyDetector(),
            ear_detector=_FakeEarDetector(1),
            crop_size=32,
        )
        assert set(result["image_id"]) == {"pilot"}

    def test_unreadable_source_is_recorded_as_failed(self, tmp_path):
        missing = tmp_path / "missing.jpg"
        manifest = _canonical_image_manifest(
            [("missing", "bteh_nellie", str(missing))]
        )
        result = run_bteh_detection(
            image_manifest=manifest,
            crops_dir=str(tmp_path / "crops"),
            manifest_path=str(tmp_path / "manifest.parquet"),
            detector=_FakeBodyDetector(),
            ear_detector=_FakeEarDetector(1),
            crop_size=32,
        )
        assert set(result["crop_id"]) == {
            "missing__body_0",
            "missing__ear_0",
            "missing__ear_1",
        }
        assert (result["detector_status"] == "failed").all()

    def test_resume_rejects_stale_source_fingerprint(self, tmp_path):
        source = tmp_path / "source.jpg"
        _write_image(source)
        manifest = _canonical_image_manifest(
            [("img1", "bteh_nellie", str(source))]
        )
        kwargs = {
            "image_manifest": manifest,
            "crops_dir": str(tmp_path / "crops"),
            "manifest_path": str(tmp_path / "manifest.parquet"),
            "detector": _FakeBodyDetector(),
            "ear_detector": _FakeEarDetector(1),
            "crop_size": 32,
            "source_fingerprint": "source-a",
            "split_fingerprint": "split-a",
        }
        run_bteh_detection(**kwargs)
        kwargs["source_fingerprint"] = "source-b"
        with pytest.raises(AssertionError, match="source_fingerprint mismatch"):
            run_bteh_detection(**kwargs)

    def test_resume_regenerates_missing_accepted_crop(self, tmp_path):
        source = tmp_path / "source.jpg"
        _write_image(source)
        manifest = _canonical_image_manifest(
            [("img1", "bteh_nellie", str(source))]
        )
        kwargs = {
            "image_manifest": manifest,
            "crops_dir": str(tmp_path / "crops"),
            "manifest_path": str(tmp_path / "manifest.parquet"),
            "detector": _FakeBodyDetector(),
            "ear_detector": _FakeEarDetector(1),
            "crop_size": 32,
        }
        first = run_bteh_detection(**kwargs)
        body_path = Path(
            first[first["crop_id"] == "img1__body_0"].iloc[0]["crop_path"]
        )
        body_path.unlink()
        second = run_bteh_detection(**kwargs)
        body = second[second["crop_id"] == "img1__body_0"].iloc[0]
        assert body["detector_status"] == "accepted"
        assert body_path.is_file()


def test_matching_retains_query_without_accepted_crop(tmp_path):
    ctx = _build_synthetic_pipeline(tmp_path)
    query_crops = ctx["query_crop_df"].copy()
    missing = query_crops.iloc[0].copy()
    missing["crop_id"] = "failed_query__body_0"
    missing["image_id"] = "failed_query"
    missing["individual_id"] = "bteh_failed"
    missing["crop_kind"] = "body"
    missing["crop_ordinal"] = 0
    missing["detector_status"] = "failed"
    query_crops = pd.concat(
        [query_crops, pd.DataFrame([missing])],
        ignore_index=True,
    )
    query_crop_path = tmp_path / "query_crops.parquet"
    query_crops.to_parquet(query_crop_path, index=False)

    results = run_bteh_step3_normalized(
        query_artifact_dir=ctx["query_artifact_dir"],
        reference_artifact_dir=ctx["ref_artifact_dir"],
        output_path=str(tmp_path / "all_queries.parquet"),
        descriptor_names=["fake_body"],
        skip_local=True,
        query_crop_manifest_path=str(query_crop_path),
    )
    failed = results[results["image_id"] == "failed_query"].iloc[0]
    assert failed["individual_id"] == "bteh_failed"
    assert failed["matching_status"] == "not_matched"


def test_matching_rejects_stale_query_crop_manifest(tmp_path):
    ctx = _build_synthetic_pipeline(tmp_path)
    query_crops = ctx["query_crop_df"].copy()
    query_crops["source_fingerprint"] = "stale-source"
    query_crop_path = tmp_path / "stale_query_crops.parquet"
    query_crops.to_parquet(query_crop_path, index=False)

    with pytest.raises(
        ValueError,
        match="query crop manifest source_fingerprint mismatch",
    ):
        run_bteh_step3_normalized(
            query_artifact_dir=ctx["query_artifact_dir"],
            reference_artifact_dir=ctx["ref_artifact_dir"],
            output_path=str(tmp_path / "stale_results.parquet"),
            descriptor_names=["fake_body"],
            skip_local=True,
            query_crop_manifest_path=str(query_crop_path),
        )


def test_matching_skips_empty_optional_reference_channel(tmp_path):
    ctx = _build_synthetic_pipeline(tmp_path)
    dimension = 16
    empty_mapping = pd.DataFrame(columns=DESCRIPTOR_MAPPING_COLUMNS)
    empty_matrix = np.empty((0, dimension), dtype=np.float32)
    ref_dir = Path(ctx["ref_artifact_dir"])
    query_dir = Path(ctx["query_artifact_dir"])
    empty_mapping.to_parquet(ref_dir / "empty_mapping.parquet", index=False)
    empty_mapping.to_parquet(query_dir / "empty_mapping.parquet", index=False)
    np.save(ref_dir / "empty.npy", empty_matrix)
    np.save(query_dir / "empty.npy", empty_matrix)
    faiss.write_index(
        faiss.IndexFlatIP(dimension),
        str(ref_dir / "empty.index"),
    )

    results = run_bteh_step3_normalized(
        query_artifact_dir=ctx["query_artifact_dir"],
        reference_artifact_dir=ctx["ref_artifact_dir"],
        output_path=str(tmp_path / "empty_channel_results.parquet"),
        descriptor_names=["fake_body", "empty"],
        skip_local=True,
    )
    assert len(results) == len(ctx["results_df"])


def test_matching_rejects_accepted_query_without_descriptors(tmp_path):
    ctx = _build_synthetic_pipeline(tmp_path)
    query_crops = ctx["query_crop_df"].copy()
    missing = query_crops[
        query_crops["crop_kind"].eq("body")
        & query_crops["detector_status"].eq("accepted")
    ].iloc[0].copy()
    missing["crop_id"] = "unembedded_query__body_0"
    missing["image_id"] = "unembedded_query"
    missing["individual_id"] = "bteh_unembedded"
    query_crops = pd.concat(
        [query_crops, pd.DataFrame([missing])],
        ignore_index=True,
    )
    query_crop_path = tmp_path / "unembedded_query_crops.parquet"
    query_crops.to_parquet(query_crop_path, index=False)

    with pytest.raises(
        RuntimeError,
        match="accepted query crops have no descriptor records",
    ):
        run_bteh_step3_normalized(
            query_artifact_dir=ctx["query_artifact_dir"],
            reference_artifact_dir=ctx["ref_artifact_dir"],
            output_path=str(tmp_path / "unembedded_results.parquet"),
            descriptor_names=["fake_body"],
            skip_local=True,
            query_crop_manifest_path=str(query_crop_path),
        )


def test_matching_requires_each_applicable_descriptor_channel(tmp_path):
    ctx = _build_synthetic_pipeline(tmp_path)
    build_bteh_descriptor_artifacts(
        crop_manifest=ctx["ref_crop_df"],
        embedder_factory=lambda _: _FakeEmbedder(dim=16, seed=7),
        descriptor_name="ear_miewid",
        artifact_dir=ctx["ref_artifact_dir"],
        is_reference=True,
        is_ear=True,
    )
    query_crop_path = tmp_path / "query_crops.parquet"
    ctx["query_crop_df"].to_parquet(query_crop_path, index=False)

    with pytest.raises(
        RuntimeError,
        match="accepted query crops have no descriptor records",
    ):
        run_bteh_step3_normalized(
            query_artifact_dir=ctx["query_artifact_dir"],
            reference_artifact_dir=ctx["ref_artifact_dir"],
            output_path=str(tmp_path / "missing_ear_channel.parquet"),
            descriptor_names=["fake_body", "ear_miewid"],
            skip_local=True,
            query_crop_manifest_path=str(query_crop_path),
        )


def test_normalized_body_embedding_flips_right_viewpoint(tmp_path):
    path = tmp_path / "asymmetric.jpg"
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, 10:] = 255
    assert cv2.imwrite(str(path), image)
    crop = pd.DataFrame(
        [
            {
                "crop_id": "img1__body_0",
                "image_id": "img1",
                "individual_id": "elephant_1",
                "crop_kind": "body",
                "crop_ordinal": 0,
                "crop_path": str(path),
                "detector_status": "accepted",
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_fingerprint": "source",
                "split_fingerprint": "split",
                "viewpoint": "right",
            }
        ]
    )
    _, matrix = embed_from_crop_manifest(
        crop,
        _MeanSideEmbedder(),
        "body",
        is_ear=False,
    )
    assert matrix[0, 0] > matrix[0, 1]


# ---------------------------------------------------------------------------
# Blocker 4: early failure when ear model is unavailable
# ---------------------------------------------------------------------------

class TestEarModelUnavailableFailsEarly:
    """Blocker 4: unavailable ear model raises before any per-image processing."""

    def _unavailable_ear(self):
        detector = _FakeEarDetector(0)
        detector._available = False
        return detector

    def test_raises_runtime_error(self, tmp_path):
        source = tmp_path / "source.jpg"
        _write_image(source)
        manifest = _canonical_image_manifest([("img1", "bteh_nellie", str(source))])
        with pytest.raises(RuntimeError, match="EarDetector model is unavailable"):
            run_bteh_detection(
                manifest,
                str(tmp_path / "crops"),
                str(tmp_path / "manifest.parquet"),
                _FakeBodyDetector(),
                self._unavailable_ear(),
            )

    def test_no_partial_records_written(self, tmp_path):
        source = tmp_path / "source.jpg"
        _write_image(source)
        manifest = _canonical_image_manifest([("img1", "bteh_nellie", str(source))])
        manifest_path = str(tmp_path / "manifest.parquet")
        with pytest.raises(RuntimeError):
            run_bteh_detection(
                manifest,
                str(tmp_path / "crops"),
                manifest_path,
                _FakeBodyDetector(),
                self._unavailable_ear(),
            )
        assert not os.path.isfile(manifest_path), (
            "Crop manifest must not be written when ear model is unavailable"
        )

    def test_no_body_crop_written_either(self, tmp_path):
        """No body crops should be written — the error is pre-loop."""
        source = tmp_path / "source.jpg"
        _write_image(source)
        manifest = _canonical_image_manifest([("img1", "bteh_nellie", str(source))])
        crops_dir = tmp_path / "crops"
        with pytest.raises(RuntimeError):
            run_bteh_detection(
                manifest,
                str(crops_dir),
                str(tmp_path / "manifest.parquet"),
                _FakeBodyDetector(),
                self._unavailable_ear(),
            )
        # No crops directory should have been created
        assert not crops_dir.exists() or not any(crops_dir.rglob("*.jpg")), (
            "No crop images should be written before the pre-loop failure"
        )
