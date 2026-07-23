import pandas as pd
import pytest
import pipeline.bteh_partitions as bteh_partitions_module

from pipeline.bteh_partitions import build_crop_partitions
from pipeline.elephant_splits import _fingerprint_df
from utils.artifact_schema import CROP_MANIFEST_COLUMNS


def test_bteh_wrapper_preserves_equals_form_paths(monkeypatch):
    captured = {}

    def fake_shared_main(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(
        bteh_partitions_module,
        "_shared_main",
        fake_shared_main,
    )
    args = [
        "--crop-manifest=crops.parquet",
        "--splits=splits.parquet",
        "--output-root=output",
    ]
    assert bteh_partitions_module.main(args) == 0
    assert captured["args"] == args


def _crop(image_id):
    values = {
        "crop_id": f"{image_id}__body_0",
        "image_id": image_id,
        "individual_id": "bteh_test",
        "crop_kind": "body",
        "crop_ordinal": 0,
        "crop_path": f"/tmp/{image_id}.jpg",
        "detector_confidence": 0.9,
        "detector_box": None,
        "detector_status": "accepted",
        "review_status": "pending",
        "schema_version": "v1",
        "source_fingerprint": "source",
        "split_fingerprint": "split",
    }
    return {column: values[column] for column in CROP_MANIFEST_COLUMNS}


def test_build_crop_partitions_assigns_all_active_images():
    splits = pd.DataFrame(
        [
            {
                "image_id": "gallery",
                "split": "gallery",
                "split_protocol": "temporal",
                "fold": 0,
                "evaluable": True,
            },
            {
                "image_id": "probe",
                "split": "held_out_probe",
                "split_protocol": "unseen_identity",
                "fold": 1,
                "evaluable": True,
            },
        ]
    )
    splits["session_id"] = ["session_gallery", "session_probe"]
    crops = pd.DataFrame([_crop("gallery"), _crop("probe")])
    crops["split_fingerprint"] = _fingerprint_df(splits)
    result = build_crop_partitions(crops, splits)
    assert set(result["reference"]["image_id"]) == {"gallery"}
    assert set(result["query"]["image_id"]) == {"probe"}


def test_build_crop_partitions_fails_missing_split():
    crops = pd.DataFrame([_crop("missing")])
    splits = pd.DataFrame(
        columns=["image_id", "split", "split_protocol", "fold", "evaluable"]
    )
    with pytest.raises(ValueError, match="missing split assignments"):
        build_crop_partitions(crops, splits)


def test_build_crop_partitions_rejects_wrong_split_fingerprint():
    splits = pd.DataFrame(
        [
            {
                "image_id": "gallery",
                "split": "gallery",
                "split_protocol": "temporal",
                "fold": 0,
                "evaluable": True,
                "session_id": "session_gallery",
            }
        ]
    )
    crops = pd.DataFrame([_crop("gallery")])
    with pytest.raises(ValueError, match="does not match"):
        build_crop_partitions(crops, splits)
