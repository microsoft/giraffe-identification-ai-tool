import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from configs.config_elpephants import canonical_individual_id
from pipeline.elpephants_manifest import (
    MANIFEST_COLUMNS,
    _apply_deduplication,
    _date_match,
    generate_manifest,
    main,
    validate_manifest,
)
from pipeline.step_1_run_detection_to_crop import _source_image_path
from utils.image_manifest_schema import fingerprint_image_manifest


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 16), color=color).save(path, format="JPEG")


def _write_metadata(
    root: Path,
    class_mapping: list[tuple[str, int]],
    train: list[tuple[str, str]],
    val: list[tuple[str, str]],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "class_mapping.txt").write_text(
        "".join(f"{class_id}\t{index}\n" for class_id, index in class_mapping)
    )
    for split, rows in (("train", train), ("val", val)):
        (root / f"{split}.txt").write_text(
            "".join(f"{class_id}\t{filename}\n" for class_id, filename in rows)
        )


@pytest.fixture()
def synthetic_elpephants(tmp_path):
    root = tmp_path / "ELPephants"
    train = [
        ("00183", "00183_Alvah II left side_Jun2008.jpg"),
        ("00183", "00183_Alvah II right side_May2010.jpg"),
        ("183", "183_Alvin right side_Jan2011.jpg"),
    ]
    val = [
        ("00183", "00183_Alvah II front_7Feb2015.jpg"),
        ("183", "183_Alvin left head_18Oct2016.jpg"),
    ]
    _write_metadata(root, [("00183", 0), ("183", 1)], train, val)
    for index, (_, filename) in enumerate(train + val):
        _write_image(root / "images" / filename, (index * 30, 20, 100))
    return root


def test_canonical_id_preserves_leading_zeroes():
    assert canonical_individual_id("00183") == "elpephants_00183"
    assert canonical_individual_id("183") == "elpephants_183"
    assert canonical_individual_id("00183") != canonical_individual_id("183")


def test_generate_manifest_preserves_source_metadata(synthetic_elpephants):
    manifest = generate_manifest(synthetic_elpephants, compute_phash=False)

    assert len(manifest) == 5
    assert set(MANIFEST_COLUMNS).issubset(manifest.columns)
    assert set(manifest["source_split"]) == {"train", "val"}
    assert set(manifest["source_class_id"]) == {"00183", "183"}
    assert set(manifest["source_class_index"]) == {0, 1}
    assert set(manifest["individual_id"]) == {
        "elpephants_00183",
        "elpephants_183",
    }
    assert manifest["image_id"].is_unique
    assert validate_manifest(manifest) == []


def test_generate_manifest_parses_date_name_and_viewpoint(synthetic_elpephants):
    manifest = generate_manifest(synthetic_elpephants, compute_phash=False)
    row = manifest[
        manifest["source_relative_path"].str.endswith(
            "00183_Alvah II front_7Feb2015.jpg"
        )
    ].iloc[0]

    assert row["individual_name"] == "Alvah II"
    assert row["capture_date"] == "2015-02-07"
    assert row["year"] == "2015"
    assert row["session_id"] == "elpephants_00183_2015-02-07"
    assert row["viewpoint"] == "frontal"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("1799_Hulweh I left head_24Jne2016.jpg", "2016-06-24"),
        ("3410_Phaedra I left2_Jna2003.jpg", "2003-01"),
        ("520_Bede right head_17pr2016.jpg", "2016-04-17"),
        ("1617_Gryta I front_June 25, 2016.jpg", "2016-06-25"),
    ],
)
def test_date_match_handles_known_source_typos(filename, expected):
    _, capture_date = _date_match(filename)
    assert capture_date == expected


def test_undated_image_does_not_create_synthetic_session(tmp_path):
    root = tmp_path / "ELPephants"
    filename = "10_Ten left side.jpg"
    _write_metadata(root, [("10", 0)], [("10", filename)], [])
    _write_image(root / "images" / filename, (10, 20, 30))

    manifest = generate_manifest(root, compute_phash=False)

    assert pd.isna(manifest.iloc[0]["session_id"])
    assert manifest.iloc[0]["session_source"] == "unknown"


def test_generate_manifest_rejects_unassigned_image(synthetic_elpephants):
    _write_image(
        synthetic_elpephants / "images" / "183_Alvin_extra.jpg",
        (1, 2, 3),
    )
    with pytest.raises(ValueError, match="unassigned files"):
        generate_manifest(synthetic_elpephants, compute_phash=False)


def test_same_identity_exact_duplicates_are_deduplicated(tmp_path):
    root = tmp_path / "ELPephants"
    first = "10_Ten left_Jan2010.jpg"
    second = "10_Ten right_Feb2011.jpg"
    _write_metadata(root, [("10", 0)], [("10", first)], [("10", second)])
    _write_image(root / "images" / first, (10, 10, 10))
    (root / "images" / second).write_bytes((root / "images" / first).read_bytes())

    manifest = generate_manifest(root, compute_phash=False)

    assert set(manifest["include_status"]) == {
        "duplicate_primary",
        "excluded",
    }
    duplicate = manifest[manifest["exclusion_reason"] == "exact_duplicate"].iloc[0]
    primary = manifest[
        manifest["include_status"] == "duplicate_primary"
    ].iloc[0]
    assert duplicate["duplicate_of"] == primary["image_id"]


def test_cross_identity_exact_duplicates_require_review(tmp_path):
    root = tmp_path / "ELPephants"
    first = "10_Ten left_Jan2010.jpg"
    second = "20_Twenty right_Feb2011.jpg"
    _write_metadata(
        root,
        [("10", 0), ("20", 1)],
        [("10", first)],
        [("20", second)],
    )
    _write_image(root / "images" / first, (10, 10, 10))
    (root / "images" / second).write_bytes((root / "images" / first).read_bytes())

    manifest = generate_manifest(root, compute_phash=False)

    assert (manifest["include_status"] == "review_required").all()
    assert (
        manifest["review_reason"] == "cross_identity_exact_duplicate"
    ).all()


def test_metadata_only_image_copies_are_pixel_deduplicated(tmp_path):
    root = tmp_path / "ELPephants"
    first = "10_Ten left_Jan2010.jpg"
    second = "10_Ten left_Feb2011.jpg"
    _write_metadata(root, [("10", 0)], [("10", first)], [("10", second)])
    _write_image(root / "images" / first, (10, 20, 30))
    first_bytes = (root / "images" / first).read_bytes()
    (root / "images" / second).write_bytes(first_bytes + b"metadata-padding")

    manifest = generate_manifest(root, compute_phash=False)

    assert manifest["content_hash"].nunique() == 2
    assert manifest["pixel_hash"].nunique() == 1
    assert set(manifest["include_status"]) == {
        "duplicate_primary",
        "excluded",
    }
    duplicate = manifest[
        manifest["exclusion_reason"] == "exact_pixel_duplicate"
    ].iloc[0]
    primary = manifest[
        manifest["include_status"] == "duplicate_primary"
    ].iloc[0]
    assert duplicate["duplicate_of"] == primary["image_id"]


def test_mixed_byte_and_pixel_duplicate_component_is_fully_deduplicated(
    tmp_path,
):
    root = tmp_path / "ELPephants"
    names = [
        "10_Ten left_Jan2010.jpg",
        "10_Ten left_Feb2011.jpg",
        "10_Ten left_Mar2012.jpg",
    ]
    _write_metadata(
        root,
        [("10", 0)],
        [("10", names[0]), ("10", names[1])],
        [("10", names[2])],
    )
    _write_image(root / "images" / names[0], (10, 20, 30))
    first_bytes = (root / "images" / names[0]).read_bytes()
    (root / "images" / names[1]).write_bytes(first_bytes)
    (root / "images" / names[2]).write_bytes(
        first_bytes + b"metadata-padding"
    )

    manifest = generate_manifest(root, compute_phash=False)

    assert (manifest["include_status"] == "duplicate_primary").sum() == 1
    assert (manifest["include_status"] == "excluded").sum() == 2
    primary_id = manifest.loc[
        manifest["include_status"] == "duplicate_primary",
        "image_id",
    ].iloc[0]
    assert set(
        manifest.loc[manifest["include_status"] == "excluded", "duplicate_of"]
    ) == {primary_id}


def test_cli_writes_manifest_and_sidecar(synthetic_elpephants, tmp_path):
    output = tmp_path / "artifacts" / "manifest.parquet"
    assert (
        main(
            [
                "--source-root",
                str(synthetic_elpephants),
                "--output",
                str(output),
                "--no-phash",
            ]
        )
        == 0
    )

    manifest = pd.read_parquet(output)
    sidecar = json.loads(output.with_suffix(".json").read_text())
    assert len(manifest) == 5
    assert sidecar["row_count"] == 5
    assert sidecar["source_split_counts"] == {"train": 3, "val": 2}


def test_normalized_crop_source_root_override(tmp_path):
    root = tmp_path / "ELPephants"
    relative = "images/example.jpg"
    row = pd.Series({"source_relative_path": relative})
    assert _source_image_path(row, source_root=root) == str(root / relative)


def test_manifest_fingerprint_changes_with_identity_assignment(
    synthetic_elpephants,
):
    manifest = generate_manifest(synthetic_elpephants, compute_phash=False)
    changed = manifest.copy()
    changed.loc[0, "individual_id"] = "elpephants_changed"
    assert fingerprint_image_manifest(manifest) != fingerprint_image_manifest(
        changed
    )


def test_deduplication_does_not_reactivate_corrupt_rows():
    manifest = pd.DataFrame(
        [
            {
                "image_id": "corrupt-a",
                "individual_id": "elpephants_1",
                "source_relative_path": "images/a.jpg",
                "content_hash": "same",
                "include_status": "excluded",
                "exclusion_reason": "corrupt",
                "duplicate_of": None,
                "review_flag": False,
                "review_reason": None,
            },
            {
                "image_id": "corrupt-b",
                "individual_id": "elpephants_1",
                "source_relative_path": "images/b.jpg",
                "content_hash": "same",
                "include_status": "excluded",
                "exclusion_reason": "corrupt",
                "duplicate_of": None,
                "review_flag": False,
                "review_reason": None,
            },
        ]
    )
    result = _apply_deduplication(manifest)
    assert (result["include_status"] == "excluded").all()
    assert (result["exclusion_reason"] == "corrupt").all()
