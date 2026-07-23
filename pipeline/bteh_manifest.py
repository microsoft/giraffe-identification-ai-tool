#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Canonical BTEH image manifest generator.

Produces one row per source image in the BTEH collection.  This script is
independent of import_from_inventory.py and does not rely on the existing
inventory spreadsheet.

Key design decisions
--------------------
* image_id   : stable, collision-safe ID = sha256(normalised_source_rel_path)[:12]
               + "_" + sha256(file_content)[:12].  Two files can only share an
               image_id if they have the same normalised relative path AND the
               same content — which is impossible for distinct files.
* individual_id : derived from the top-level BTEH folder name after stripping
                  trailing whitespace and parenthetical herd suffixes; prefixed
                  with "bteh_".  UUID/hex32 top-level dirs yield
                  individual_id="unresolved" and are flagged for review.
* Deduplication : images with identical SHA-256 content hashes are grouped;
                  one representative row keeps include_status="duplicate_primary"
                  and the others get include_status="excluded" with
                  exclusion_reason="exact_duplicate" and duplicate_of set to the
                  primary image_id.
* Session / year : derived first from EXIF DateTimeOriginal or DateTime, then
                   from the first directory component whose name matches a
                   year-like or date-like pattern, then from the parent folder
                   name.  UUID/hex strings are never parsed as years.
* include_status  : one of
    included            — eligible for AI training/evaluation
    excluded            — not eligible (see exclusion_reason)
    review_required     — ambiguous; human review needed before use
    duplicate_primary   — kept representative of an exact-duplicate group
* exclusion_reason (when excluded):
    not_for_ai, ref_thumbnail, raw_file, video, archive, non_image,
    corrupt, exact_duplicate, uuid_dir_unresolved

Usage
-----
    python pipeline/bteh_manifest.py \\
        [--source-root PATH]   \\  # default: BTEH_SOURCE_ROOT env / config
        [--artifact-root PATH] \\  # default: BTEH_ARTIFACT_ROOT env / config
        [--output PATH]        \\  # default: <artifact-root>/v1/manifests/bteh_image_manifest.parquet
        [--no-hash]               # skip perceptual hash (faster, less dedup)
"""

import argparse
import hashlib
import logging
import re
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from PIL import Image, ExifTags

from configs.config_bteh import (
    ARTIFACT_SCHEMA_VERSION,
    BTEH_ARTIFACT_ROOT,
    BTEH_SOURCE_ROOT,
    MANIFEST_FILENAME,
    MANIFEST_SUBDIR,
    canonical_individual_id,
    is_uuid_dir,
    split_herd_suffix,
)
from utils.image_manifest_schema import (
    IMAGE_MANIFEST_COLUMNS,
    fingerprint_image_manifest,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
RAW_EXTENSIONS: frozenset[str] = frozenset(
    {".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".rw2", ".orf", ".srw", ".pef"}
)
VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".flv", ".3gp", ".mts", ".m2ts"}
)
ARCHIVE_EXTENSIONS: frozenset[str] = frozenset(
    {".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".tgz"}
)

# EXIF tag IDs
EXIF_DATETIME_ORIGINAL = 36867
EXIF_DATETIME = 306

# Regex to detect a year-like token (2000–2035) in a directory component name.
# Deliberately does NOT match hex digits or UUID segments.
YEAR_RE = re.compile(r"\b(20[0-2][0-9])\b")
# Separated date: 2023-10-19 or 2023_10_19
DATE_RE = re.compile(r"\b(20[0-2][0-9])[-_/](\d{2})[-_/](\d{2})\b")
# Compact date: 20231019 — 8 consecutive digits, first 4 form a year 20XX
COMPACT_DATE_RE = re.compile(r"(?<!\d)(20[0-2][0-9])\d{4}(?!\d)")

# Patterns that flag a filename as a ref thumbnail
THUMBNAIL_PATTERNS = re.compile(r"_thumb\b", re.IGNORECASE)

# Patterns that flag ambiguous identity in the filename
AMBIGUOUS_PATTERNS = re.compile(
    r"(unsure|maybe|_unsure|unknown|ambiguous|uncertain|multi)", re.IGNORECASE
)
MULTI_ELEPHANT_PATTERNS = re.compile(
    r"(herd\s*\d|multi.?eleph|group)", re.IGNORECASE
)

# Manifest schema columns (ordered)
MANIFEST_COLUMNS = IMAGE_MANIFEST_COLUMNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _compute_perceptual_hash(path: Path) -> Optional[str]:
    """Compute a simple 8×8 average-hash using only Pillow (no imagehash dep)."""
    try:
        with Image.open(path) as img:
            img = img.convert("L").resize((9, 8), Image.LANCZOS)
            pixels = list(img.getdata())
        # Build 8×8 = 64 bits by comparing adjacent columns
        bits = []
        for row in range(8):
            for col in range(8):
                bits.append(1 if pixels[row * 9 + col] > pixels[row * 9 + col + 1] else 0)
        hex_val = hex(int("".join(map(str, bits)), 2))[2:].zfill(16)
        return hex_val
    except Exception:
        return None


def _exif_capture_date(path: Path) -> Optional[datetime]:
    """Try to read EXIF DateTimeOriginal or DateTime from an image file."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
        if not exif:
            return None
        for tag_id in (EXIF_DATETIME_ORIGINAL, EXIF_DATETIME):
            raw = exif.get(tag_id)
            if raw and isinstance(raw, str):
                try:
                    return datetime.strptime(raw.strip(), "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    pass
        # Try IFD 34665 (Exif IFD) for DateTimeOriginal
        try:
            ifd = exif.get_ifd(34665)
            raw = ifd.get(EXIF_DATETIME_ORIGINAL)
            if raw and isinstance(raw, str):
                return datetime.strptime(raw.strip(), "%Y:%m:%d %H:%M:%S")
        except Exception:
            pass
    except Exception:
        pass
    return None


def _year_from_dir_components(rel_parts: list[str]) -> Optional[str]:
    """
    Extract a year from directory path components (not the filename).
    Only directory components are considered; the filename is excluded.
    UUID/hex strings are never parsed as years.
    Returns the most specific (last) year found, or None.
    """
    found = None
    for part in rel_parts[:-1]:  # exclude filename
        # Skip UUID / hex32 components
        if is_uuid_dir(part):
            continue
        # Separated date: 2023-10-19
        m = DATE_RE.search(part)
        if m:
            found = m.group(1)
            continue
        # Compact date: 20231019
        m = COMPACT_DATE_RE.search(part)
        if m:
            found = m.group(1)
            continue
        # Plain year token: 2023
        m = YEAR_RE.search(part)
        if m:
            found = m.group(1)
    return found


def _session_from_dir_components(rel_parts: list[str]) -> Optional[str]:
    """
    Build a conservative session label from directory path components only.
    Skips top-level elephant name dir (index 0) and UUID dirs.
    Returns the most-specific non-UUID subdirectory name, or None.
    """
    # rel_parts = [top_dir, sub_dir?, ..., filename]
    # Ignore top-level (elephant dir) and filename.
    middle = rel_parts[1:-1]
    for part in reversed(middle):
        if is_uuid_dir(part):
            continue
        return part.strip()
    return None


def _image_dimensions(path: Path) -> tuple[Optional[int], Optional[int]]:
    try:
        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Top-level BTEH directory classification
# ---------------------------------------------------------------------------

class _DirKind:
    NAMED_ELEPHANT = "named_elephant"
    UUID_UNRESOLVED = "uuid_unresolved"
    REF = "ref"
    ZIPS = "zips"
    SKIP = "skip"  # non-directory or metadata files


def _classify_top_dir(name: str) -> str:
    stripped = name.strip()
    if stripped.lower() == "ref":
        return _DirKind.REF
    if stripped.lower() == "zips":
        return _DirKind.ZIPS
    if is_uuid_dir(stripped):
        return _DirKind.UUID_UNRESOLVED
    return _DirKind.NAMED_ELEPHANT


# ---------------------------------------------------------------------------
# Per-file record builder
# ---------------------------------------------------------------------------

def _build_record(
    source_root: Path,
    file_path: Path,
    top_dir_name: str,
    top_dir_kind: str,
    compute_phash: bool,
) -> dict:
    rel_path = file_path.relative_to(source_root)
    rel_parts = rel_path.parts
    rel_str = rel_path.as_posix()

    norm_rel = rel_str.lower().strip("/")
    path_hash_component = _sha256_str(norm_rel)[:12]

    content_hash = _sha256_file(file_path)
    content_hash_component = content_hash[:12]
    image_id = f"{path_hash_component}_{content_hash_component}"

    phash = _compute_perceptual_hash(file_path) if compute_phash else None
    width, height = _image_dimensions(file_path)

    # --- Individual identity from top-level folder ---
    display_name, herd = split_herd_suffix(top_dir_name)
    individual_name: Optional[str] = None
    individual_id: Optional[str] = None

    if top_dir_kind == _DirKind.NAMED_ELEPHANT:
        individual_name = display_name
        individual_id = canonical_individual_id(display_name)
    elif top_dir_kind == _DirKind.REF:
        # ref/<name>/... — use the second directory component as identity
        if len(rel_parts) >= 3:
            ref_name = rel_parts[1].strip()
            individual_name = ref_name.title()
            individual_id = canonical_individual_id(ref_name)
        else:
            individual_name = None
            individual_id = "unresolved"
    elif top_dir_kind == _DirKind.UUID_UNRESOLVED:
        individual_name = None
        individual_id = "unresolved"

    # --- Inclusion status and exclusion reason ---
    include_status = "included"
    exclusion_reason = None
    dataset_role = "source"
    review_flag = False
    review_reason = None

    # Check ancestry for "Not for AI"
    lower_parts = [p.lower() for p in rel_parts]
    if "not for ai" in lower_parts:
        include_status = "excluded"
        exclusion_reason = "not_for_ai"

    # ref thumbnails
    if include_status == "included" and top_dir_kind == _DirKind.REF:
        dataset_role = "ref"
        if THUMBNAIL_PATTERNS.search(file_path.name):
            include_status = "excluded"
            exclusion_reason = "ref_thumbnail"

    # UUID unresolved
    if include_status == "included" and top_dir_kind == _DirKind.UUID_UNRESOLVED:
        include_status = "review_required"
        review_flag = True
        review_reason = "uuid_dir_unresolved"

    # Non-image file types
    ext = file_path.suffix.lower()
    if include_status == "included" and ext in RAW_EXTENSIONS:
        include_status = "excluded"
        exclusion_reason = "raw_file"
    elif include_status == "included" and ext in VIDEO_EXTENSIONS:
        include_status = "excluded"
        exclusion_reason = "video"
    elif include_status == "included" and ext in ARCHIVE_EXTENSIONS:
        include_status = "excluded"
        exclusion_reason = "archive"

    # Image corruption check (only for image-extension files that are still included)
    if include_status in ("included", "review_required") and width is None:
        if ext in IMAGE_EXTENSIONS:
            include_status = "excluded"
            exclusion_reason = "corrupt"

    # Ambiguous identity from filename annotation
    if include_status == "included":
        if AMBIGUOUS_PATTERNS.search(file_path.name):
            review_flag = True
            review_reason = (review_reason or "") + "ambiguous_identity;"
        if MULTI_ELEPHANT_PATTERNS.search(file_path.name):
            review_flag = True
            review_reason = (review_reason or "") + "possible_multi_elephant;"

    # --- Session / year provenance ---
    capture_date: Optional[str] = None
    session_source = "folder"

    if ext in IMAGE_EXTENSIONS and include_status != "excluded":
        dt = _exif_capture_date(file_path)
        if dt:
            capture_date = dt.strftime("%Y-%m-%d")
            session_source = "exif"

    year: Optional[str] = None
    if capture_date:
        year = capture_date[:4]
    else:
        year = _year_from_dir_components(list(rel_parts))

    # session_id: use capture_date if from EXIF; otherwise derive from
    # the most specific dated/named directory component.
    if session_source == "exif" and capture_date:
        # Group by individual + date to form a session
        session_id = f"{individual_id or 'unknown'}_{capture_date}"
    else:
        folder_session = _session_from_dir_components(list(rel_parts))
        if folder_session:
            # Normalise: lowercase, collapse spaces
            norm_sess = re.sub(r"\s+", "_", folder_session.lower())
            session_id = f"{individual_id or 'unknown'}_{norm_sess}"
            session_source = "folder"
        elif year:
            session_id = f"{individual_id or 'unknown'}_{year}"
            session_source = "year_folder"
        else:
            session_id = f"{individual_id or 'unknown'}_unknown"
            session_source = "unknown"

    return {
        "image_id": image_id,
        "individual_id": individual_id,
        "individual_name": individual_name,
        "herd": herd,
        "source_relative_path": rel_str,
        "content_hash": content_hash,
        "perceptual_hash": phash,
        "image_id_path_component": path_hash_component,
        "image_id_content_component": content_hash_component,
        "session_id": session_id,
        "capture_date": capture_date,
        "year": year,
        "session_source": session_source,
        "dataset_role": dataset_role,
        "include_status": include_status,
        "exclusion_reason": exclusion_reason,
        "duplicate_of": None,  # filled in deduplication pass
        "review_flag": review_flag,
        "review_reason": review_reason,
        "body_crop_status": "pending",
        "ear_detection_status": "pending",
        "image_width": width,
        "image_height": height,
    }


# ---------------------------------------------------------------------------
# Deduplication pass
# ---------------------------------------------------------------------------

def _apply_deduplication(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark exact-content duplicates deterministically.

    For each group of rows with the same content_hash:
    - Sort by source_relative_path (lexicographic) for determinism.
    - Keep the first row as "duplicate_primary".
    - Mark the rest as excluded / exact_duplicate.
    """
    df = df.copy()
    # Only deduplicate among currently included rows
    included_mask = df["include_status"].isin(["included", "review_required", "duplicate_primary"])
    included_df = df[included_mask].copy()

    dup_groups = included_df.groupby("content_hash").filter(lambda g: len(g) > 1)
    if dup_groups.empty:
        return df

    for content_hash, group in dup_groups.groupby("content_hash"):
        sorted_idx = group.sort_values("source_relative_path").index.tolist()
        primary_idx = sorted_idx[0]
        # Set primary
        df.loc[primary_idx, "include_status"] = "duplicate_primary"
        primary_id = df.loc[primary_idx, "image_id"]
        # Set duplicates
        for idx in sorted_idx[1:]:
            df.loc[idx, "include_status"] = "excluded"
            df.loc[idx, "exclusion_reason"] = "exact_duplicate"
            df.loc[idx, "duplicate_of"] = primary_id

    return df


# ---------------------------------------------------------------------------
# Integrity validation
# ---------------------------------------------------------------------------

def validate_manifest(df: pd.DataFrame) -> list[str]:
    """Return a list of integrity-error messages (empty = all OK)."""
    errors = []

    # image_id uniqueness
    dup_ids = df["image_id"][df["image_id"].duplicated()].unique()
    if len(dup_ids):
        errors.append(
            f"image_id is not unique — {len(dup_ids)} duplicate image_ids found: "
            f"{list(dup_ids[:5])}"
        )

    # Every row must have an image_id
    missing_id = df["image_id"].isna().sum()
    if missing_id:
        errors.append(f"{missing_id} rows have null image_id")

    # include_status values
    valid_statuses = {"included", "excluded", "review_required", "duplicate_primary"}
    bad_status = set(df["include_status"].dropna().unique()) - valid_statuses
    if bad_status:
        errors.append(f"Unknown include_status values: {bad_status}")

    # Excluded rows must have an exclusion_reason
    excluded = df[df["include_status"] == "excluded"]
    missing_reason = excluded["exclusion_reason"].isna().sum()
    if missing_reason:
        errors.append(
            f"{missing_reason} excluded rows have no exclusion_reason"
        )

    # exact_duplicate rows must reference a valid primary
    dup_rows = df[df["exclusion_reason"] == "exact_duplicate"]
    if not dup_rows.empty:
        valid_ids = set(df["image_id"])
        bad_refs = dup_rows[~dup_rows["duplicate_of"].isin(valid_ids)]
        if not bad_refs.empty:
            errors.append(
                f"{len(bad_refs)} exact_duplicate rows reference an unknown duplicate_of id"
            )

    return errors


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def generate_manifest(
    source_root: Path,
    compute_phash: bool = True,
) -> pd.DataFrame:
    """
    Scan *source_root* and return a manifest DataFrame.
    Does not write any files.
    """
    if not source_root.is_dir():
        raise FileNotFoundError(f"BTEH source root not found: {source_root}")

    rows = []
    top_entries = sorted(source_root.iterdir())

    for top_entry in top_entries:
        if not top_entry.is_dir():
            # Skip non-directory files at top level (e.g. elephant_name_to_path.csv)
            continue

        top_name = top_entry.name
        top_kind = _classify_top_dir(top_name)

        if top_kind == _DirKind.SKIP:
            continue
        if top_kind == _DirKind.ZIPS:
            # Archive directory: scan for archives and mark as excluded
            for f in top_entry.rglob("*"):
                if not f.is_file():
                    continue
                ext = f.suffix.lower()
                rel_str = f.relative_to(source_root).as_posix()
                norm_rel = rel_str.lower()
                path_hash_component = _sha256_str(norm_rel)[:12]
                rows.append({
                    "image_id": f"{path_hash_component}_{'0' * 12}",
                    "individual_id": None,
                    "individual_name": None,
                    "herd": None,
                    "source_relative_path": rel_str,
                    "content_hash": None,
                    "perceptual_hash": None,
                    "image_id_path_component": path_hash_component,
                    "image_id_content_component": "0" * 12,
                    "session_id": None,
                    "capture_date": None,
                    "year": None,
                    "session_source": None,
                    "dataset_role": "archive",
                    "include_status": "excluded",
                    "exclusion_reason": "archive",
                    "duplicate_of": None,
                    "review_flag": False,
                    "review_reason": None,
                    "body_crop_status": None,
                    "ear_detection_status": None,
                    "image_width": None,
                    "image_height": None,
                })
            continue

        # Walk named elephant, UUID, or ref dir
        for f in sorted(top_entry.rglob("*")):
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            # Classify non-image by extension type first
            if ext not in IMAGE_EXTENSIONS and ext not in RAW_EXTENSIONS and \
               ext not in VIDEO_EXTENSIONS and ext not in ARCHIVE_EXTENSIONS:
                # Skip misc non-image/non-media files entirely
                continue
            try:
                record = _build_record(
                    source_root, f, top_name, top_kind, compute_phash
                )
                rows.append(record)
            except Exception as exc:
                logger.warning("Failed to process %s: %s", f, exc)
                # Emit a minimal excluded row so the file is accounted for
                rel_str = f.relative_to(source_root).as_posix()
                norm_rel = rel_str.lower()
                path_hash_component = _sha256_str(norm_rel)[:12]
                rows.append({
                    "image_id": f"{path_hash_component}_{'e' * 12}",
                    "individual_id": None,
                    "individual_name": None,
                    "herd": None,
                    "source_relative_path": rel_str,
                    "content_hash": None,
                    "perceptual_hash": None,
                    "image_id_path_component": path_hash_component,
                    "image_id_content_component": "e" * 12,
                    "session_id": None,
                    "capture_date": None,
                    "year": None,
                    "session_source": None,
                    "dataset_role": "source",
                    "include_status": "excluded",
                    "exclusion_reason": "corrupt",
                    "duplicate_of": None,
                    "review_flag": False,
                    "review_reason": f"exception: {exc}",
                    "body_crop_status": None,
                    "ear_detection_status": None,
                    "image_width": None,
                    "image_height": None,
                })

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

    # Deduplication pass
    df = _apply_deduplication(df)

    # Validate
    errors = validate_manifest(df)
    if errors:
        msg = "Manifest integrity errors:\n" + "\n".join(f"  {e}" for e in errors)
        raise RuntimeError(msg)

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate canonical BTEH image manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help="Path to read-only BTEH source root (default: BTEH_SOURCE_ROOT env or config default)",
    )
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Path to writable artifact root (default: BTEH_ARTIFACT_ROOT env or config default)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output path for the manifest parquet file "
            "(default: <artifact-root>/v1/manifests/bteh_image_manifest.parquet)"
        ),
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip perceptual hash computation (faster but less precise near-duplicate detection)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load existing manifest and validate without regenerating",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    source_root = Path(args.source_root) if args.source_root else BTEH_SOURCE_ROOT
    artifact_root = Path(args.artifact_root) if args.artifact_root else BTEH_ARTIFACT_ROOT

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            artifact_root / ARTIFACT_SCHEMA_VERSION / MANIFEST_SUBDIR / MANIFEST_FILENAME
        )

    if args.validate_only:
        if not output_path.exists():
            logger.error("Manifest not found: %s", output_path)
            return 1
        df = pd.read_parquet(output_path)
        errors = validate_manifest(df)
        if errors:
            for e in errors:
                logger.error(e)
            return 1
        logger.info("Manifest valid: %d rows", len(df))
        return 0

    logger.info("Source root    : %s", source_root)
    logger.info("Output         : %s", output_path)

    df = generate_manifest(source_root, compute_phash=not args.no_hash)

    # Summary
    n_total = len(df)
    n_included = (df["include_status"] == "included").sum()
    n_excluded = (df["include_status"] == "excluded").sum()
    n_review = (df["include_status"] == "review_required").sum()
    n_dup_primary = (df["include_status"] == "duplicate_primary").sum()

    logger.info("Manifest rows  : %d", n_total)
    logger.info("  included     : %d", n_included)
    logger.info("  excl         : %d", n_excluded)
    logger.info("  review req'd : %d", n_review)
    logger.info("  dup primary  : %d", n_dup_primary)

    if args.verbose:
        excl_breakdown = (
            df[df["include_status"] == "excluded"]["exclusion_reason"]
            .value_counts()
            .to_string()
        )
        logger.debug("Exclusion breakdown:\n%s", excl_breakdown)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute fingerprint for integrity tracking
    fingerprint = fingerprint_image_manifest(df)
    logger.info("Manifest fingerprint: %s", fingerprint)

    df.to_parquet(output_path, index=False)
    logger.info("Manifest written to: %s", output_path)

    # Also write a small sidecar with fingerprint and schema version
    sidecar = output_path.with_suffix(".json")
    import json
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "manifest_fingerprint": fingerprint,
                "row_count": n_total,
                "included_count": int(n_included),
                "excluded_count": int(n_excluded),
                "review_count": int(n_review),
                "generated_at": datetime.utcnow().isoformat() + "Z",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
