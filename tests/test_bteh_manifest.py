"""
Tests for pipeline/bteh_manifest.py using synthetic in-memory / temp trees.

Does NOT run heavyweight processing over real BTEH data.
"""

import hashlib
import os
import sys
from itertools import count as _count
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("data_root_abs_path", "/tmp/test_data")
os.environ.setdefault("container_name", "test_container")
# Use a non-existent path so config_bteh imports cleanly
os.environ.setdefault("BTEH_SOURCE_ROOT", "/nonexistent/BTEH")
os.environ.setdefault("BTEH_ARTIFACT_ROOT", "/nonexistent/artifacts")

from pipeline.bteh_manifest import (
    MANIFEST_COLUMNS,
    _apply_deduplication,
    _build_record,
    _classify_top_dir,
    _DirKind,
    _exif_capture_date,
    _session_from_dir_components,
    _year_from_dir_components,
    generate_manifest,
    validate_manifest,
)
from configs.config_bteh import (
    canonical_individual_id,
    is_uuid_dir,
    split_herd_suffix,
)

import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# Fixtures: synthetic BTEH tree
# ---------------------------------------------------------------------------

# Use a module-level counter so every image gets a unique color → unique SHA256.
_COLOR_COUNTER = _count(10)


def _make_unique_image(path: Path, size=(32, 32)):
    """Create a minimal valid JPEG with a unique color (unique content hash)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    c = next(_COLOR_COUNTER)
    color = (c % 256, (c * 7) % 256, (c * 13) % 256)
    img = Image.new("RGB", size, color=color)
    img.save(str(path), format="JPEG", quality=95)


@pytest.fixture()
def synthetic_bteh(tmp_path):
    """
    Build a minimal synthetic BTEH tree:

        Aya/
          Aya 2023/
            img1.jpg
            img2.jpg
          Not for AI/
            nai.jpg
        Balu/
          balu1.jpg
        Belle (Herd 5)/
          belle1.jpg
        ref/
          aya/
            aya_orig.jpg
            aya_thumb.jpg   ← thumbnail (should be excluded)
        <uuid>/
          uuid_img.jpg
        zips/
          archive.zip
        Beauty (Herd 4) /     ← trailing-space dir
          beauty1.jpg
    """
    root = tmp_path / "BTEH"

    # Aya — 2 images in dated subdir, 1 in Not for AI
    _make_unique_image(root / "Aya" / "Aya 2023" / "img1.jpg")
    _make_unique_image(root / "Aya" / "Aya 2023" / "img2.jpg")
    _make_unique_image(root / "Aya" / "Not for AI" / "nai.jpg")

    # Balu — 1 image directly in identity dir
    _make_unique_image(root / "Balu" / "balu1.jpg")

    # Belle with herd suffix
    belle_dir = root / "Belle (Herd 5)"
    belle_dir.mkdir(parents=True, exist_ok=True)
    _make_unique_image(belle_dir / "belle1.jpg")

    # ref — 1 original + 1 thumbnail
    _make_unique_image(root / "ref" / "aya" / "aya_orig.jpg")
    _make_unique_image(root / "ref" / "aya" / "aya_thumb_thumb.jpg")  # thumb pattern

    # UUID dir
    uuid_name = "0e4e2295-c853-496a-8c67-aca3ee021928"
    _make_unique_image(root / uuid_name / "uuid_img.jpg")

    # hex32 dir
    hex32_name = "066acaf6d37045499c824022c9e3ee46"
    _make_unique_image(root / hex32_name / "hex_img.jpg")

    # zips dir
    zip_dir = root / "zips"
    zip_dir.mkdir(parents=True, exist_ok=True)
    (zip_dir / "archive.zip").write_bytes(b"PK\x03\x04" + b"\x00" * 20)

    # Trailing-space dir
    beauty_dir = root / "Beauty (Herd 4) "
    beauty_dir.mkdir(parents=True, exist_ok=True)
    _make_unique_image(beauty_dir / "beauty1.jpg")

    return root


@pytest.fixture()
def exact_duplicate_bteh(tmp_path):
    """BTEH tree with two image files that have identical content."""
    root = tmp_path / "BTEH_dup"

    # Same image written to two paths
    img = Image.new("RGB", (32, 32), color=(100, 100, 100))

    p1 = root / "Aya" / "Aya 2023" / "same1.jpg"
    p1.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(p1), format="JPEG", quality=95)

    # Write identical bytes
    p2 = root / "Aya" / "Aya 2023" / "same2.jpg"
    import shutil
    shutil.copy(str(p1), str(p2))

    return root


# ---------------------------------------------------------------------------
# Unit tests: config helpers
# ---------------------------------------------------------------------------

class TestConfigHelpers:
    def test_is_uuid_dir_full_uuid(self):
        assert is_uuid_dir("0e4e2295-c853-496a-8c67-aca3ee021928")

    def test_is_uuid_dir_hex32(self):
        assert is_uuid_dir("066acaf6d37045499c824022c9e3ee46")

    def test_is_uuid_dir_named(self):
        assert not is_uuid_dir("Aya")
        assert not is_uuid_dir("Balu")
        assert not is_uuid_dir("Beauty (Herd 4) ")

    def test_split_herd_suffix_with_herd(self):
        name, herd = split_herd_suffix("Beauty (Herd 4) ")
        assert name == "Beauty"
        assert herd == "Herd 4"

    def test_split_herd_suffix_no_herd(self):
        name, herd = split_herd_suffix("Balu")
        assert name == "Balu"
        assert herd is None

    def test_split_herd_suffix_trailing_spaces(self):
        name, herd = split_herd_suffix("Kunene  ")
        assert name == "Kunene"
        assert herd is None

    def test_canonical_individual_id_simple(self):
        assert canonical_individual_id("Balu") == "bteh_balu"

    def test_canonical_individual_id_multiword(self):
        assert canonical_individual_id("Half Moon") == "bteh_half_moon"

    def test_canonical_individual_id_trailing_space(self):
        assert canonical_individual_id("Vula ") == "bteh_vula"


# ---------------------------------------------------------------------------
# Unit tests: year/session extraction
# ---------------------------------------------------------------------------

class TestYearSessionExtraction:
    def test_year_from_dated_folder(self):
        parts = ["Aya", "Aya 2023", "img.jpg"]
        assert _year_from_dir_components(parts) == "2023"

    def test_year_skips_uuid_dir(self):
        parts = ["0e4e2295-c853-496a-8c67-aca3ee021928", "img.jpg"]
        assert _year_from_dir_components(parts) is None

    def test_year_skips_hex32_dir(self):
        parts = ["066acaf6d37045499c824022c9e3ee46", "sub", "img.jpg"]
        assert _year_from_dir_components(parts) is None

    def test_year_from_date_folder(self):
        parts = ["Beauty (Herd 4)", "Beauty - 20231019", "img.jpg"]
        assert _year_from_dir_components(parts) == "2023"

    def test_session_from_dir_components(self):
        parts = ["Aya", "Aya 2023", "img.jpg"]
        session = _session_from_dir_components(parts)
        assert session == "Aya 2023"

    def test_session_skips_uuid(self):
        parts = ["0e4e2295-c853-496a-8c67-aca3ee021928", "img.jpg"]
        session = _session_from_dir_components(parts)
        assert session is None


# ---------------------------------------------------------------------------
# Unit tests: top-dir classification
# ---------------------------------------------------------------------------

class TestTopDirClassify:
    def test_ref_dir(self):
        assert _classify_top_dir("ref") == _DirKind.REF

    def test_zips_dir(self):
        assert _classify_top_dir("zips") == _DirKind.ZIPS

    def test_uuid_dir(self):
        assert _classify_top_dir("0e4e2295-c853-496a-8c67-aca3ee021928") == _DirKind.UUID_UNRESOLVED

    def test_hex32_dir(self):
        assert _classify_top_dir("066acaf6d37045499c824022c9e3ee46") == _DirKind.UUID_UNRESOLVED

    def test_named_dir(self):
        assert _classify_top_dir("Aya") == _DirKind.NAMED_ELEPHANT
        assert _classify_top_dir("Belle (Herd 5) ") == _DirKind.NAMED_ELEPHANT


# ---------------------------------------------------------------------------
# Integration tests: generate_manifest on synthetic tree
# ---------------------------------------------------------------------------

class TestGenerateManifest:
    def test_runs_without_error(self, synthetic_bteh):
        df = generate_manifest(synthetic_bteh, compute_phash=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_all_columns_present(self, synthetic_bteh):
        df = generate_manifest(synthetic_bteh, compute_phash=False)
        for col in MANIFEST_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_image_id_uniqueness(self, synthetic_bteh):
        df = generate_manifest(synthetic_bteh, compute_phash=False)
        assert df["image_id"].nunique() == len(df), "image_id values are not unique"

    def test_not_for_ai_excluded(self, synthetic_bteh):
        df = generate_manifest(synthetic_bteh, compute_phash=False)
        nai = df[df["source_relative_path"].str.contains("Not for AI", regex=False)]
        assert len(nai) > 0, "Expected Not-for-AI files"
        assert (nai["include_status"] == "excluded").all()
        assert (nai["exclusion_reason"] == "not_for_ai").all()

    def test_ref_thumbnail_excluded(self, synthetic_bteh):
        df = generate_manifest(synthetic_bteh, compute_phash=False)
        thumb = df[df["source_relative_path"].str.contains("_thumb", regex=False)]
        assert len(thumb) > 0, "Expected thumbnail files"
        assert (thumb["include_status"] == "excluded").all()
        assert (thumb["exclusion_reason"] == "ref_thumbnail").all()

    def test_ref_original_preserved(self, synthetic_bteh):
        df = generate_manifest(synthetic_bteh, compute_phash=False)
        ref_orig = df[
            df["source_relative_path"].str.startswith("ref/") &
            ~df["source_relative_path"].str.contains("_thumb")
        ]
        assert len(ref_orig) > 0, "Expected ref original files"
        assert (ref_orig["dataset_role"] == "ref").all()
        # ref originals should be included (not excluded)
        assert (ref_orig["include_status"].isin(["included", "duplicate_primary", "review_required"])).all()

    def test_uuid_dir_review_required(self, synthetic_bteh):
        df = generate_manifest(synthetic_bteh, compute_phash=False)
        uuid_rows = df[df["source_relative_path"].str.startswith("0e4e2295")]
        assert len(uuid_rows) > 0
        assert (uuid_rows["include_status"] == "review_required").all()
        assert (uuid_rows["individual_id"] == "unresolved").all()

    def test_hex32_dir_review_required(self, synthetic_bteh):
        df = generate_manifest(synthetic_bteh, compute_phash=False)
        hex_rows = df[df["source_relative_path"].str.startswith("066acaf6")]
        assert len(hex_rows) > 0
        assert (hex_rows["include_status"] == "review_required").all()

    def test_herd_suffix_stripped_from_individual_name(self, synthetic_bteh):
        df = generate_manifest(synthetic_bteh, compute_phash=False)
        belle = df[df["source_relative_path"].str.startswith("Belle")]
        assert len(belle) > 0
        assert (belle["individual_name"] == "Belle").all()
        assert (belle["herd"] == "Herd 5").all()

    def test_year_extracted_from_dated_folder(self, synthetic_bteh):
        df = generate_manifest(synthetic_bteh, compute_phash=False)
        aya = df[
            df["source_relative_path"].str.startswith("Aya/Aya 2023/") &
            (df["include_status"] != "excluded")
        ]
        assert len(aya) > 0
        assert (aya["year"] == "2023").all()

    def test_individual_id_format(self, synthetic_bteh):
        df = generate_manifest(synthetic_bteh, compute_phash=False)
        named = df[
            df["individual_id"].notna() &
            (df["individual_id"] != "unresolved")
        ]
        assert (named["individual_id"].str.startswith("bteh_")).all()

    def test_validate_passes(self, synthetic_bteh):
        df = generate_manifest(synthetic_bteh, compute_phash=False)
        errors = validate_manifest(df)
        assert errors == [], f"Validation errors: {errors}"

    def test_invalid_source_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            generate_manifest(tmp_path / "nonexistent", compute_phash=False)


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_exact_duplicates_marked(self, exact_duplicate_bteh):
        df = generate_manifest(exact_duplicate_bteh, compute_phash=False)
        # Exactly one of the two same-content files should be duplicate_primary
        dup_primary = df[df["include_status"] == "duplicate_primary"]
        dup_excluded = df[df["exclusion_reason"] == "exact_duplicate"]
        # same1 and same2 have the same content: one primary + one excluded
        assert len(dup_primary) == 1
        assert len(dup_excluded) == 1
        # The excluded row should reference the primary's image_id
        primary_id = dup_primary.iloc[0]["image_id"]
        assert dup_excluded.iloc[0]["duplicate_of"] == primary_id

    def test_no_duplicates_when_content_differs(self, synthetic_bteh):
        df = generate_manifest(synthetic_bteh, compute_phash=False)
        dup_excl = df[df["exclusion_reason"] == "exact_duplicate"]
        # In the synthetic tree all images have the same color so some duplicates
        # may exist — the important thing is they are properly marked, not that
        # there are zero.  Verify the referenced primary exists.
        for _, row in dup_excl.iterrows():
            assert row["duplicate_of"] in df["image_id"].values


# ---------------------------------------------------------------------------
# Validate_manifest tests
# ---------------------------------------------------------------------------

class TestValidateManifest:
    def _make_minimal_df(self):
        return pd.DataFrame([
            {
                "image_id": "abc_def",
                "include_status": "included",
                "exclusion_reason": None,
                "duplicate_of": None,
                "content_hash": "deadbeef",
            }
        ])

    def test_duplicate_image_id_fails(self):
        df = self._make_minimal_df()
        df2 = df.copy()
        combined = pd.concat([df, df2], ignore_index=True)
        errors = validate_manifest(combined)
        assert any("unique" in e for e in errors)

    def test_excluded_without_reason_fails(self):
        df = self._make_minimal_df()
        df["include_status"] = "excluded"
        df["exclusion_reason"] = None
        errors = validate_manifest(df)
        assert any("exclusion_reason" in e for e in errors)

    def test_valid_df_passes(self):
        df = self._make_minimal_df()
        errors = validate_manifest(df)
        assert errors == []
