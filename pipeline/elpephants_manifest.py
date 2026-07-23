#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""Build the canonical image manifest for the ELPephants benchmark dataset."""

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_elpephants import (
    ARTIFACT_SCHEMA_VERSION,
    ELPEPHANTS_ARTIFACT_ROOT,
    ELPEPHANTS_SOURCE_ROOT,
    MANIFEST_FILENAME,
    MANIFEST_SUBDIR,
    canonical_individual_id,
)
from utils.image_manifest_schema import (
    IMAGE_MANIFEST_COLUMNS,
    fingerprint_image_manifest,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
SOURCE_SPLITS = ("train", "val")
SOURCE_METADATA_COLUMNS = [
    "source_class_id",
    "source_class_index",
    "source_split",
    "viewpoint",
]
MANIFEST_COLUMNS = IMAGE_MANIFEST_COLUMNS + SOURCE_METADATA_COLUMNS

_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_MONTH_TOKEN = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DAY_FIRST_DATE_RE = re.compile(
    rf"(?i)(?P<day>\d{{1,2}})\s*(?P<month>{_MONTH_TOKEN})"
    r"\s*(?P<year>\d{2,4})"
)
_MONTH_DAY_YEAR_RE = re.compile(
    rf"(?i)(?P<month>{_MONTH_TOKEN})\s*(?P<day>\d{{1,2}})"
    r"[\s_-]+(?P<year>\d{2,4})"
)
_MONTH_YEAR_RE = re.compile(
    rf"(?i)(?P<month>{_MONTH_TOKEN})[\s_-]*(?P<year>20\d{{2}})"
)
_YEAR_RE = re.compile(r"(?<!\d)(?P<year>20\d{2})(?!\d)")
_VIEWPOINT_RE = re.compile(
    r"(?i)\b(lefts?(?:ide)?|rights?(?:ide)?|front(?:al)?|rear)"
    r"(?:\s*(?:side|head))?\b"
)
_RESIDUAL_REGION_RE = re.compile(r"(?i)\b(?:side|head)\b")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _perceptual_hash(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pixels = list(gray.getdata())
    except (OSError, ValueError):
        return None

    bits = [
        pixels[row * 9 + column] > pixels[row * 9 + column + 1]
        for row in range(8)
        for column in range(8)
    ]
    return f"{int(''.join('1' if bit else '0' for bit in bits), 2):016x}"


def _four_digit_year(value: str) -> int:
    year = int(value)
    return 2000 + year if len(value) == 2 else year


def _date_match(filename: str) -> tuple[re.Match[str] | None, str | None]:
    stem = Path(filename).stem
    for pattern in (_DAY_FIRST_DATE_RE, _MONTH_DAY_YEAR_RE, _MONTH_YEAR_RE):
        matches = list(pattern.finditer(stem))
        if not matches:
            continue
        match = matches[-1]
        year = _four_digit_year(match.group("year"))
        month = _MONTHS[match.group("month").lower()]
        day_value = match.groupdict().get("day")
        if day_value:
            try:
                parsed = datetime(year, month, int(day_value))
            except ValueError:
                return match, f"{year:04d}-{month:02d}"
            return match, parsed.strftime("%Y-%m-%d")
        return match, f"{year:04d}-{month:02d}"

    matches = list(_YEAR_RE.finditer(stem))
    if matches:
        match = matches[-1]
        return match, match.group("year")
    return None, None


def _viewpoint(filename: str) -> str:
    matches = list(
        _VIEWPOINT_RE.finditer(Path(filename).stem.replace("_", " "))
    )
    if not matches:
        return "unknown"
    value = matches[-1].group(1).lower()
    if value.startswith("left"):
        return "left"
    if value.startswith("right"):
        return "right"
    if value.startswith("front"):
        return "frontal"
    return "rear"


def _name_candidate(source_class_id: str, filename: str) -> str:
    stem = Path(filename).stem
    prefix = f"{source_class_id}_"
    if not stem.startswith(prefix):
        raise ValueError(
            f"filename {filename!r} does not start with class ID {source_class_id!r}"
        )
    value = stem[len(prefix) :].replace("_", " ")
    date_match, _ = _date_match(filename)
    if date_match:
        source_offset = len(prefix)
        relative_start = max(0, date_match.start() - source_offset)
        value = value[:relative_start]
    value = _VIEWPOINT_RE.sub(" ", value)
    value = _RESIDUAL_REGION_RE.sub(" ", value)
    return re.sub(r"\s+", " ", value).strip(" _-")


def _load_class_mapping(source_root: Path) -> dict[str, int]:
    path = source_root / "class_mapping.txt"
    if not path.is_file():
        raise FileNotFoundError(f"ELPephants class mapping not found: {path}")

    mapping: dict[str, int] = {}
    indexes: set[int] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_number}: expected class_id<TAB>index")
        class_id, raw_index = parts
        if class_id in mapping:
            raise ValueError(f"{path}:{line_number}: duplicate class ID {class_id!r}")
        index = int(raw_index)
        if index in indexes:
            raise ValueError(f"{path}:{line_number}: duplicate class index {index}")
        canonical_individual_id(class_id)
        mapping[class_id] = index
        indexes.add(index)

    if sorted(indexes) != list(range(len(indexes))):
        raise ValueError("ELPephants class indexes must be contiguous from zero")
    return mapping


def _load_source_assignments(
    source_root: Path,
    class_mapping: dict[str, int],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    assigned: set[str] = set()
    for source_split in SOURCE_SPLITS:
        path = source_root / f"{source_split}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"ELPephants split metadata not found: {path}")
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_number}: expected class_id<TAB>filename")
            class_id, filename = parts
            if class_id not in class_mapping:
                raise ValueError(
                    f"{path}:{line_number}: unknown class ID {class_id!r}"
                )
            if Path(filename).name != filename:
                raise ValueError(
                    f"{path}:{line_number}: filename must not contain directories"
                )
            if filename in assigned:
                raise ValueError(
                    f"{path}:{line_number}: image assigned more than once: {filename}"
                )
            assigned.add(filename)
            rows.append(
                {
                    "source_class_id": class_id,
                    "source_split": source_split,
                    "filename": filename,
                }
            )

    image_dir = source_root / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"ELPephants image directory not found: {image_dir}")
    actual = {
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if assigned != actual:
        missing = sorted(assigned - actual)
        unassigned = sorted(actual - assigned)
        raise ValueError(
            "ELPephants metadata/image mismatch: "
            f"{len(missing)} missing files, {len(unassigned)} unassigned files; "
            f"missing sample={missing[:5]}, unassigned sample={unassigned[:5]}"
        )
    return rows


def _canonical_names(assignments: list[dict[str, str]]) -> dict[str, str]:
    candidates: dict[str, Counter[str]] = defaultdict(Counter)
    for row in assignments:
        class_id = row["source_class_id"]
        candidate = _name_candidate(class_id, row["filename"])
        if candidate:
            candidates[class_id][candidate] += 1

    names: dict[str, str] = {}
    for class_id, counts in candidates.items():
        names[class_id] = sorted(
            counts,
            key=lambda name: (-counts[name], -len(name), name.casefold()),
        )[0]
    return names


def _apply_deduplication(manifest: pd.DataFrame) -> pd.DataFrame:
    manifest = manifest.copy()
    eligible = manifest[manifest["include_status"] == "included"]
    for _, group in eligible.groupby("content_hash", dropna=True):
        if len(group) < 2:
            continue
        if group["individual_id"].nunique() > 1:
            manifest.loc[group.index, "include_status"] = "review_required"
            manifest.loc[group.index, "review_flag"] = True
            manifest.loc[group.index, "review_reason"] = (
                "cross_identity_exact_duplicate"
            )
            continue
        indexes = group.sort_values("source_relative_path").index.tolist()
        primary_index = indexes[0]
        manifest.loc[primary_index, "include_status"] = "duplicate_primary"
        primary_id = manifest.loc[primary_index, "image_id"]
        for index in indexes[1:]:
            manifest.loc[index, "include_status"] = "excluded"
            manifest.loc[index, "exclusion_reason"] = "exact_duplicate"
            manifest.loc[index, "duplicate_of"] = primary_id
    return manifest


def validate_manifest(manifest: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing_columns = set(MANIFEST_COLUMNS) - set(manifest.columns)
    if missing_columns:
        return [f"missing manifest columns: {sorted(missing_columns)}"]
    if manifest["image_id"].isna().any() or manifest["image_id"].duplicated().any():
        errors.append("image_id must be non-null and unique")
    if manifest["source_relative_path"].duplicated().any():
        errors.append("source_relative_path must be unique")
    if not manifest["source_split"].isin(SOURCE_SPLITS).all():
        errors.append("source_split must be train or val")
    if manifest["source_class_id"].isna().any():
        errors.append("source_class_id must be non-null")
    expected_ids = manifest["source_class_id"].map(canonical_individual_id)
    if not expected_ids.equals(manifest["individual_id"]):
        errors.append("individual_id does not match source_class_id")

    valid_statuses = {
        "included",
        "excluded",
        "review_required",
        "duplicate_primary",
    }
    if not set(manifest["include_status"]).issubset(valid_statuses):
        errors.append("include_status contains unknown values")
    excluded = manifest["include_status"] == "excluded"
    if manifest.loc[excluded, "exclusion_reason"].isna().any():
        errors.append("excluded rows must have an exclusion_reason")
    duplicate_rows = manifest["exclusion_reason"] == "exact_duplicate"
    valid_ids = set(manifest["image_id"])
    if not manifest.loc[duplicate_rows, "duplicate_of"].isin(valid_ids).all():
        errors.append("exact duplicates must reference a valid primary image_id")
    return errors


def generate_manifest(
    source_root: Path,
    compute_phash: bool = True,
) -> pd.DataFrame:
    if not source_root.is_dir():
        raise FileNotFoundError(f"ELPephants source root not found: {source_root}")

    class_mapping = _load_class_mapping(source_root)
    assignments = _load_source_assignments(source_root, class_mapping)
    canonical_names = _canonical_names(assignments)
    rows: list[dict] = []

    for assignment in sorted(assignments, key=lambda row: row["filename"]):
        class_id = assignment["source_class_id"]
        filename = assignment["filename"]
        path = source_root / "images" / filename
        relative_path = path.relative_to(source_root).as_posix()
        content_hash = _sha256_file(path)
        path_hash = _sha256_text(relative_path.lower())[:12]
        content_component = content_hash[:12]
        individual_id = canonical_individual_id(class_id)
        date_match, capture_date = _date_match(filename)
        del date_match
        year = capture_date[:4] if capture_date else None
        session_value = capture_date or "unknown"

        include_status = "included"
        exclusion_reason = None
        width = None
        height = None
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
        except (OSError, ValueError):
            include_status = "excluded"
            exclusion_reason = "corrupt"

        rows.append(
            {
                "image_id": f"{path_hash}_{content_component}",
                "individual_id": individual_id,
                "individual_name": canonical_names.get(class_id, class_id),
                "herd": None,
                "source_relative_path": relative_path,
                "content_hash": content_hash,
                "perceptual_hash": (
                    _perceptual_hash(path)
                    if compute_phash and include_status != "excluded"
                    else None
                ),
                "image_id_path_component": path_hash,
                "image_id_content_component": content_component,
                "session_id": f"{individual_id}_{session_value}",
                "capture_date": capture_date,
                "year": year,
                "session_source": "filename" if capture_date else "unknown",
                "dataset_role": "source",
                "include_status": include_status,
                "exclusion_reason": exclusion_reason,
                "duplicate_of": None,
                "review_flag": False,
                "review_reason": None,
                "body_crop_status": "pending",
                "ear_detection_status": "pending",
                "image_width": width,
                "image_height": height,
                "source_class_id": class_id,
                "source_class_index": class_mapping[class_id],
                "source_split": assignment["source_split"],
                "viewpoint": _viewpoint(filename),
            }
        )

    manifest = _apply_deduplication(
        pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    )
    errors = validate_manifest(manifest)
    if errors:
        raise RuntimeError(
            "ELPephants manifest integrity errors:\n"
            + "\n".join(f"  {error}" for error in errors)
        )
    return manifest


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate the canonical ELPephants image manifest"
    )
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--no-phash",
        action="store_true",
        help="Skip perceptual hashes; exact SHA-256 deduplication still runs.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    source_root = (
        Path(args.source_root) if args.source_root else ELPEPHANTS_SOURCE_ROOT
    )
    artifact_root = (
        Path(args.artifact_root)
        if args.artifact_root
        else ELPEPHANTS_ARTIFACT_ROOT
    )
    output_path = (
        Path(args.output)
        if args.output
        else artifact_root
        / ARTIFACT_SCHEMA_VERSION
        / MANIFEST_SUBDIR
        / MANIFEST_FILENAME
    )

    if args.validate_only:
        if not output_path.is_file():
            logger.error("Manifest not found: %s", output_path)
            return 1
        errors = validate_manifest(pd.read_parquet(output_path))
        for error in errors:
            logger.error(error)
        return 1 if errors else 0

    manifest = generate_manifest(
        source_root,
        compute_phash=not args.no_phash,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output_path, index=False)

    fingerprint = fingerprint_image_manifest(manifest)
    status_counts = manifest["include_status"].value_counts()
    sidecar = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "manifest_fingerprint": fingerprint,
        "row_count": len(manifest),
        "included_count": int(status_counts.get("included", 0)),
        "excluded_count": int(status_counts.get("excluded", 0)),
        "review_count": int(status_counts.get("review_required", 0)),
        "duplicate_primary_count": int(
            status_counts.get("duplicate_primary", 0)
        ),
        "source_split_counts": {
            key: int(value)
            for key, value in manifest["source_split"].value_counts().items()
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2) + "\n"
    )
    logger.info("Manifest rows: %d", len(manifest))
    logger.info("Manifest fingerprint: %s", fingerprint)
    logger.info("Manifest written to: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
