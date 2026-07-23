# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
import pstats
import cProfile
import argparse
import json
import logging
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers_matching import print_memory_usage, log_to_file, restore_stdout
from utils.helpers_matching import load_data_dirs, load_metadata_file
from models.detector import ElephantDetector, EarDetector
from configs.config_elephant import VIEWPOINT_COL, CROP_SUBDIR, EAR_CROP_SUBDIR
from configs.config_artifacts import ARTIFACT_SCHEMA_VERSION
from utils.artifact_schema import (
    CROP_MANIFEST_COLUMNS,
    CROP_MANIFEST_DTYPES,
    TERMINAL_CROP_STATUSES,
    assert_crop_manifest_integrity,
    make_crop_id,
)


logger = logging.getLogger(__name__)
CROP_MANIFEST_FILENAME = "crop_manifest.parquet"


def _crop_output_path(path_relative_to_root: str, processed_img_dir: str) -> str:
    parts = path_relative_to_root.rsplit(".", 1)
    ext = parts[1] if len(parts) == 2 else "jpg"
    img_filename = os.path.basename(path_relative_to_root)
    stem = img_filename.rsplit(".", 1)[0]
    return os.path.join(
        processed_img_dir,
        "zoomed_version",
        f"{stem}_cropped_torso_zoomed.{ext}",
    )


def _ear_crop_output_path(path_relative_to_root: str, ear_crop_dir: str) -> str:
    parts = path_relative_to_root.rsplit(".", 1)
    ext = parts[1] if len(parts) == 2 else "jpg"
    img_filename = os.path.basename(path_relative_to_root)
    stem = img_filename.rsplit(".", 1)[0]
    return os.path.join(ear_crop_dir, f"{stem}_ear_cropped.{ext}")


def _check_if_processed_image_exists(path_relative_to_root: str, processed_img_dir: str) -> bool:
    return os.path.isfile(_crop_output_path(path_relative_to_root, processed_img_dir))


def run_detection(metadata_table, metadata_filepath, input_img_dir, processed_img_dir, detector, ear_detector, crop_size):

    zoomed_dir = os.path.join(processed_img_dir, "zoomed_version")
    ear_crop_dir = os.path.join(processed_img_dir, "ear_crops")
    os.makedirs(zoomed_dir, exist_ok=True)
    os.makedirs(ear_crop_dir, exist_ok=True)
    metadata_table["ai_found_torso"] = metadata_table["ai_found_torso"].astype(object)
    if "ai_found_ear" not in metadata_table.columns:
        metadata_table["ai_found_ear"] = np.nan
    metadata_table["ai_found_ear"] = metadata_table["ai_found_ear"].astype(object)

    for idx, row in tqdm(metadata_table.iterrows(), total=metadata_table.shape[0]):

        if idx % 100 == 0:
            metadata_table.to_csv(metadata_filepath, index=False)

        path_rel = row["path_relative_to_root"]
        full_path = os.path.join(input_img_dir, path_rel)

        if not os.path.exists(full_path):
            metadata_table.loc[idx, "ai_found_torso"] = "file_not_found"
            continue

        body_crop_exists = _check_if_processed_image_exists(path_rel, processed_img_dir)
        ear_crop_path    = _ear_crop_output_path(path_rel, ear_crop_dir)
        ear_crop_exists  = os.path.isfile(ear_crop_path)

        if body_crop_exists and ear_crop_exists:
            metadata_table.loc[idx, "ai_found_torso"] = "existing_item"
            metadata_table.loc[idx, "ai_found_ear"]   = "existing_item"
            continue

        image_bgr = cv2.imread(full_path)
        if image_bgr is None:
            metadata_table.loc[idx, "ai_found_torso"] = "file_not_found"
            continue

        if body_crop_exists:
            # Whole-body crop already saved — load it for ear detection
            crop_resized = cv2.imread(_crop_output_path(path_rel, processed_img_dir))
            metadata_table.loc[idx, "ai_found_torso"] = "existing_item"
        else:
            crop, viewpoint = detector.crop(image_bgr)
            metadata_table.loc[idx, VIEWPOINT_COL] = viewpoint

            if crop is None:
                metadata_table.loc[idx, "ai_found_torso"] = "False"
                metadata_table.loc[idx, "ai_found_ear"]   = "False"
                continue

            crop_resized = cv2.resize(crop, (crop_size, crop_size))
            out_path = _crop_output_path(path_rel, processed_img_dir)
            cv2.imwrite(out_path, crop_resized)
            metadata_table.loc[idx, "ai_found_torso"] = "True"

        # Ear detection on the whole-body crop (skip if already done)
        if not ear_crop_exists:
            ear_crop = ear_detector.detect_ear(crop_resized)
            if ear_crop is not None:
                cv2.imwrite(ear_crop_path, ear_crop)
                metadata_table.loc[idx, "ai_found_ear"] = "True"
            else:
                metadata_table.loc[idx, "ai_found_ear"] = "False"

    return metadata_table


def make_bteh_crop_paths(image_id: str, crops_dir: str) -> dict[str, str]:
    """Return canonical absolute body and ear crop paths for an image."""
    root = os.path.abspath(crops_dir)
    return {
        "body": os.path.join(root, f"{make_crop_id(image_id, 'body', 0)}.jpg"),
        "ear_0": os.path.join(root, f"{make_crop_id(image_id, 'ear', 0)}.jpg"),
        "ear_1": os.path.join(root, f"{make_crop_id(image_id, 'ear', 1)}.jpg"),
    }


def _empty_crop_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {column: pd.Series(dtype=dtype) for column, dtype in CROP_MANIFEST_DTYPES.items()}
    )


def load_or_init_crop_manifest(manifest_path: str) -> pd.DataFrame:
    """Load an existing crop manifest or return an empty normalized manifest."""
    if not os.path.isfile(manifest_path):
        return _empty_crop_manifest()
    df = pd.read_parquet(manifest_path)
    missing = [column for column in CROP_MANIFEST_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"crop manifest {manifest_path!r} is missing columns: {missing}")
    return df


def save_crop_manifest(df: pd.DataFrame, manifest_path: str) -> None:
    """Write the normalized crop manifest parquet."""
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    normalized = df.copy()
    for column, dtype in CROP_MANIFEST_DTYPES.items():
        normalized[column] = normalized[column].astype(dtype)
    normalized.to_parquet(manifest_path, index=False)


def _crop_record(
    *,
    image_id: str,
    individual_id: str,
    crop_kind: str,
    crop_ordinal: int,
    crop_path: str,
    schema_version: str,
    source_fingerprint: str | None,
    split_fingerprint: str | None = None,
    detector_status: str = "accepted",
    confidence: float | None = None,
    box: list[int] | None = None,
) -> dict:
    return {
        "crop_id": make_crop_id(image_id, crop_kind, crop_ordinal),
        "image_id": image_id,
        "individual_id": individual_id,
        "crop_kind": crop_kind,
        "crop_ordinal": crop_ordinal,
        "crop_path": os.path.abspath(crop_path),
        "detector_confidence": confidence,
        "detector_box": json.dumps(box, separators=(",", ":")) if box is not None else None,
        "detector_status": detector_status,
        "review_status": "pending",
        "schema_version": schema_version,
        "source_fingerprint": source_fingerprint,
        "split_fingerprint": split_fingerprint,
    }


def _run_body_detection(
    image_bgr: np.ndarray,
    image_id: str,
    individual_id: str,
    crops_dir: str,
    detector,
    crop_size: int,
    schema_version: str,
    source_fingerprint: str | None,
    split_fingerprint: str | None = None,
) -> dict | None:
    """Detect and save one body crop."""
    result = detector.crop(image_bgr)
    confidence = None
    box = None
    if isinstance(result, dict):
        crop = result.get("crop")
        confidence = result.get("score")
        box = result.get("box")
    elif isinstance(result, tuple):
        crop = result[0]
    else:
        crop = result
    if crop is None or crop.size == 0:
        return None

    resized = cv2.resize(crop, (crop_size, crop_size))
    path = make_bteh_crop_paths(image_id, crops_dir)["body"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, resized):
        raise OSError(f"failed to write body crop to {path}")
    return _crop_record(
        image_id=image_id,
        individual_id=individual_id,
        crop_kind="body",
        crop_ordinal=0,
        crop_path=path,
        confidence=float(confidence) if confidence is not None else None,
        box=[int(value) for value in box] if box is not None else None,
        schema_version=schema_version,
        source_fingerprint=source_fingerprint,
        split_fingerprint=split_fingerprint,
    )


def _run_ear_detection(
    image_bgr: np.ndarray,
    image_id: str,
    individual_id: str,
    crops_dir: str,
    ear_detector,
    schema_version: str,
    source_fingerprint: str | None,
    split_fingerprint: str | None = None,
) -> list[dict]:
    """Detect, save, and describe zero to two ear crops."""
    if getattr(ear_detector, "_available", True) is False:
        raise RuntimeError("EarDetector model is unavailable")
    detections = sorted(
        ear_detector.detect_ears(image_bgr),
        key=lambda detection: (
            (
                float(detection["box"][0]) + float(detection["box"][2])
            )
            / 2.0
            if detection.get("box") is not None
            else float(detection.get("ordinal", 0))
        ),
    )
    paths = make_bteh_crop_paths(image_id, crops_dir)
    records = []
    for ordinal, detection in enumerate(detections[:2]):
        crop = detection.get("crop")
        if crop is None or crop.size == 0:
            continue
        path = paths[f"ear_{ordinal}"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not cv2.imwrite(path, crop):
            raise OSError(f"failed to write ear crop to {path}")
        records.append(
            _crop_record(
                image_id=image_id,
                individual_id=individual_id,
                crop_kind="ear",
                crop_ordinal=ordinal,
                crop_path=path,
                confidence=float(detection["score"]) if detection.get("score") is not None else None,
                box=[int(value) for value in detection["box"]]
                if detection.get("box") is not None
                else None,
                schema_version=schema_version,
                source_fingerprint=source_fingerprint,
                split_fingerprint=split_fingerprint,
            )
        )
    return sorted(records, key=lambda record: record["crop_ordinal"])


def _source_image_path(
    row: pd.Series,
    source_root: str | os.PathLike[str] | None = None,
) -> str:
    for column in ("source_path", "absolute_path", "image_path", "path"):
        value = row.get(column)
        if pd.notna(value) and str(value):
            return os.path.abspath(str(value))
    relative = row.get("source_relative_path")
    if pd.isna(relative) or not str(relative):
        raise ValueError(
            f"image_id={row.get('image_id')!r} has no source image path column"
        )
    relative = str(relative)
    if not os.path.isabs(relative) and source_root is None:
        raise ValueError(
            f"image_id={row.get('image_id')!r} uses source_relative_path but "
            "no source_root was provided"
        )
    return (
        relative
        if os.path.isabs(relative)
        else os.path.join(os.fspath(source_root), relative)
    )


def run_bteh_detection(
    image_manifest: pd.DataFrame,
    crops_dir: str,
    manifest_path: str,
    detector,
    ear_detector,
    crop_size: int = 512,
    schema_version: str = ARTIFACT_SCHEMA_VERSION,
    source_fingerprint: str | None = None,
    split_fingerprint: str | None = None,
    source_root: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    """Run resumable normalized body and multi-ear detection using stable IDs.

    Terminal slot statuses (will not be retried on resume):
      ``accepted``       – crop was found and saved successfully.
      ``none_detected``  – detector ran but found nothing; slot is genuinely empty.
      ``not_applicable`` – parent slot was none_detected so this slot cannot exist.
    Retryable status:
      ``failed``         – an unexpected error; slot will be re-attempted.

    An image is considered complete when every requested slot (body, ear_0, ear_1)
    has a terminal status, so images with legitimately 0 or 1 ears are never re-run.
    """
    # -----------------------------------------------------------------------
    # Blocker 4: fail before the loop when ear generation is requested but
    # the model is unavailable — do not write partial misleading records.
    # -----------------------------------------------------------------------
    if getattr(ear_detector, "_available", True) is False:
        raise RuntimeError(
            "EarDetector model is unavailable. Cannot run normalized detection with ear "
            "generation enabled. Either fix the model or disable ear generation."
        )

    if "include_status" not in image_manifest.columns:
        raise ValueError(
            "image manifest is missing required column 'include_status'"
        )
    included = image_manifest[
        image_manifest["include_status"].isin({"included", "duplicate_primary"})
    ]
    if "individual_id" in included.columns:
        included = included[
            included["individual_id"].notna()
            & included["individual_id"].astype(str).str.strip().ne("")
            & included["individual_id"].astype(str).ne("unresolved")
        ]
    if "_pilot_role" in included.columns:
        included = included[included["_pilot_role"] == "pilot"]
    if "image_id" not in included.columns:
        raise ValueError("image manifest is missing required column 'image_id'")
    missing_ids = included["image_id"].isna() | included["image_id"].astype(str).str.strip().eq("")
    if missing_ids.any():
        raise ValueError(
            f"included rows have missing image_id values at indices "
            f"{included.index[missing_ids].tolist()}"
        )

    # -----------------------------------------------------------------------
    # Blocker 2: build image_id → individual_id lookup from the image manifest.
    # -----------------------------------------------------------------------
    if "individual_id" not in included.columns:
        raise ValueError(
            "image manifest is missing required column 'individual_id'. "
            "Populate it from the canonical manifest before running detection."
        )
    id_to_individual: dict[str, str] = {
        str(row["image_id"]): str(row["individual_id"])
        for _, row in included.iterrows()
    }

    crop_df = load_or_init_crop_manifest(manifest_path)
    if not crop_df.empty:
        assert_crop_manifest_integrity(
            crop_df,
            image_manifest,
            schema_version=schema_version,
            expected_source_fingerprint=source_fingerprint,
            expected_split_fingerprint=split_fingerprint,
        )
    records = crop_df.to_dict("records")
    for record in records:
        if (
            record.get("detector_status") == "accepted"
            and cv2.imread(str(record.get("crop_path", ""))) is None
        ):
            logger.warning(
                "Accepted crop is missing or unreadable; scheduling retry: %s",
                record.get("crop_path"),
            )
            record["detector_status"] = "failed"

    def _terminal_ids() -> set[str]:
        """Crop IDs with a terminal detector status."""
        return {
            str(record["crop_id"])
            for record in records
            if record.get("detector_status") in TERMINAL_CROP_STATUSES
        }

    for processed, (_, row) in enumerate(included.iterrows(), start=1):
        image_id = str(row["image_id"])
        individual_id = id_to_individual[image_id]

        # -----------------------------------------------------------------------
        # Blocker 3: an image is complete when all three slots are terminal.
        # This ensures 0-ear and 1-ear images are not re-run forever.
        # -----------------------------------------------------------------------
        terminal = _terminal_ids()
        expected_ids = {
            make_crop_id(image_id, "body", 0),
            make_crop_id(image_id, "ear", 0),
            make_crop_id(image_id, "ear", 1),
        }
        if expected_ids.issubset(terminal):
            continue

        image_path = _source_image_path(row, source_root=source_root)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            logger.warning("Could not read source image %s", image_path)
            paths = make_bteh_crop_paths(image_id, crops_dir)
            for crop_kind, ordinal, path_key in (
                ("body", 0, "body"),
                ("ear", 0, "ear_0"),
                ("ear", 1, "ear_1"),
            ):
                crop_id = make_crop_id(image_id, crop_kind, ordinal)
                records = [
                    record
                    for record in records
                    if record.get("crop_id") != crop_id
                ]
                records.append(
                    _crop_record(
                        image_id=image_id,
                        individual_id=individual_id,
                        crop_kind=crop_kind,
                        crop_ordinal=ordinal,
                        crop_path=paths[path_key],
                        detector_status="failed",
                        schema_version=schema_version,
                        source_fingerprint=source_fingerprint,
                        split_fingerprint=split_fingerprint,
                    )
                )
            continue

        body_id = make_crop_id(image_id, "body", 0)
        body_record = next(
            (
                record
                for record in records
                if record.get("crop_id") == body_id
                and record.get("detector_status") in TERMINAL_CROP_STATUSES
            ),
            None,
        )
        ear_input = image_bgr
        if body_record is None:
            body_record = _run_body_detection(
                image_bgr,
                image_id,
                individual_id,
                crops_dir,
                detector,
                crop_size,
                schema_version,
                source_fingerprint,
                split_fingerprint,
            )
            if body_record is not None:
                records = [record for record in records if record.get("crop_id") != body_id]
                records.append(body_record)
            else:
                # Body was not detected — record the slot as none_detected so it
                # is not retried, then attempt ear detection on the full image.
                records = [record for record in records if record.get("crop_id") != body_id]
                records.append(
                    _crop_record(
                        image_id=image_id,
                        individual_id=individual_id,
                        crop_kind="body",
                        crop_ordinal=0,
                        crop_path=make_bteh_crop_paths(image_id, crops_dir)["body"],
                        detector_status="none_detected",
                        schema_version=schema_version,
                        source_fingerprint=source_fingerprint,
                        split_fingerprint=split_fingerprint,
                    )
                )

        if body_record is not None and body_record.get("detector_status") == "accepted":
            loaded_body = cv2.imread(str(body_record["crop_path"]))
            if loaded_body is not None:
                ear_input = loaded_body

        # Determine which ear slots are still missing a terminal status.
        terminal = _terminal_ids()
        ear0_id = make_crop_id(image_id, "ear", 0)
        ear1_id = make_crop_id(image_id, "ear", 1)
        ear_slots_complete = ear0_id in terminal and ear1_id in terminal
        if not ear_slots_complete:
            try:
                ear_records = _run_ear_detection(
                    ear_input,
                    image_id,
                    individual_id,
                    crops_dir,
                    ear_detector,
                    schema_version,
                    source_fingerprint,
                    split_fingerprint,
                )
            except RuntimeError as exc:
                # Unexpected runtime error — mark ear_0 as failed (retryable)
                # and leave ear_1 unset so the image can be retried.
                logger.error("Ear detection failed for image_id=%s: %s", image_id, exc)
                if ear0_id not in terminal:
                    records = [
                        record for record in records if record.get("crop_id") != ear0_id
                    ]
                    records.append(
                        _crop_record(
                            image_id=image_id,
                            individual_id=individual_id,
                            crop_kind="ear",
                            crop_ordinal=0,
                            crop_path=make_bteh_crop_paths(image_id, crops_dir)["ear_0"],
                            detector_status="failed",
                            schema_version=schema_version,
                            source_fingerprint=source_fingerprint,
                            split_fingerprint=split_fingerprint,
                        )
                    )
            else:
                # Replace any existing ear records for this image.
                found_ordinals = {r["crop_ordinal"] for r in ear_records}
                for rec in ear_records:
                    cid = rec["crop_id"]
                    records = [r for r in records if r.get("crop_id") != cid]
                    records.append(rec)

                # Write terminal placeholder records for slots that were not
                # detected so that resume skips them on the next run.
                for ordinal, slot_id, slot_key in (
                    (0, ear0_id, "ear_0"),
                    (1, ear1_id, "ear_1"),
                ):
                    if ordinal in found_ordinals:
                        continue  # accepted record already appended above
                    if slot_id in terminal:
                        continue  # already has a terminal status
                    # Determine status: if ear_0 was not found either, ear_1
                    # is "not_applicable"; otherwise it is "none_detected".
                    if ordinal == 1 and 0 not in found_ordinals:
                        status = "not_applicable"
                    else:
                        status = "none_detected"
                    records = [r for r in records if r.get("crop_id") != slot_id]
                    records.append(
                        _crop_record(
                            image_id=image_id,
                            individual_id=individual_id,
                            crop_kind="ear",
                            crop_ordinal=ordinal,
                            crop_path=make_bteh_crop_paths(image_id, crops_dir)[slot_key],
                            detector_status=status,
                            schema_version=schema_version,
                            source_fingerprint=source_fingerprint,
                            split_fingerprint=split_fingerprint,
                        )
                    )

        crop_df = pd.DataFrame(records, columns=CROP_MANIFEST_COLUMNS)
        if processed % 50 == 0:
            save_crop_manifest(crop_df, manifest_path)

    crop_df = pd.DataFrame(records, columns=CROP_MANIFEST_COLUMNS)
    if crop_df.empty:
        crop_df = _empty_crop_manifest()
    assert_crop_manifest_integrity(
        crop_df,
        image_manifest,
        schema_version=schema_version,
        expected_source_fingerprint=source_fingerprint,
        expected_split_fingerprint=split_fingerprint,
    )
    save_crop_manifest(crop_df, manifest_path)
    return load_or_init_crop_manifest(manifest_path)


run_normalized_detection = run_bteh_detection


def main(partition):

    profiler = cProfile.Profile()
    profiler.enable()

    root_dir, processed_img_dir = load_data_dirs()
    input_img_dir = root_dir

    log_file_std_output, log_file_err_output = log_to_file(root_dir, "elephant_detection")

    metadata_filepath = os.path.join(root_dir, partition + "_dir", "metadata_" + partition + ".csv")
    metadata_table = load_metadata_file(metadata_filepath)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = ElephantDetector(backend="megadetector", conf=0.5, device=device)
    ear_detector = EarDetector(device=device)

    # crop_size default mirrors the giraffe pipeline's cropped_img_size
    crop_size = 512

    metadata_table = run_detection(
        metadata_table, metadata_filepath, input_img_dir, processed_img_dir, detector, ear_detector, crop_size
    )

    metadata_table.to_csv(metadata_filepath, index=False)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats()

    print_memory_usage()

    restore_stdout(log_file_std_output, log_file_err_output)


def _normalized_main(*, use_bteh_defaults: bool = False):
    """Normalized detection route for a canonical elephant image manifest."""
    default_crops_dir = None
    default_source_root = None
    if use_bteh_defaults:
        from configs.config_bteh import (
            ARTIFACT_VERSION_ROOT,
            BTEH_SOURCE_ROOT,
            CROPS_SUBDIR,
        )
        default_crops_dir = str(ARTIFACT_VERSION_ROOT / CROPS_SUBDIR)
        default_source_root = str(BTEH_SOURCE_ROOT)

    parser = argparse.ArgumentParser(
        description=(
            "Normalized elephant detection: generate body and ear crops "
            "from a canonical image manifest."
        )
    )
    parser.add_argument(
        "--image-manifest",
        required=True,
        help="Path to a canonical elephant image manifest parquet.",
    )
    parser.add_argument(
        "--crop-manifest",
        required=True,
        help="Path to write (or resume) the normalized crop manifest parquet.",
    )
    parser.add_argument(
        "--crops-dir",
        default=default_crops_dir,
        required=not use_bteh_defaults,
        help="Directory to write generated crop images.",
    )
    parser.add_argument(
        "--source-root",
        default=default_source_root,
        required=not use_bteh_defaults,
        help="Root used to resolve source_relative_path values.",
    )
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--schema-version", default=ARTIFACT_SCHEMA_VERSION)
    parser.add_argument("--source-fingerprint", required=True)
    parser.add_argument("--split-fingerprint", required=True)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (cuda/cpu). Defaults to cuda when available.",
    )
    parser.add_argument(
        "--disable-cudnn",
        action="store_true",
        help="Disable cuDNN and use generic CUDA kernels for incompatible hosts.",
    )
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False

    image_manifest = pd.read_parquet(args.image_manifest)

    detector = ElephantDetector(backend="megadetector", conf=0.5, device=device)
    ear_detector = EarDetector(device=device)

    run_bteh_detection(
        image_manifest=image_manifest,
        crops_dir=args.crops_dir,
        manifest_path=args.crop_manifest,
        detector=detector,
        ear_detector=ear_detector,
        crop_size=args.crop_size,
        schema_version=args.schema_version,
        source_fingerprint=args.source_fingerprint,
        split_fingerprint=args.split_fingerprint,
        source_root=args.source_root,
    )


def _bteh_main():
    _normalized_main(use_bteh_defaults=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run elephant detector to crop images")
    parser.add_argument(
        "--partition",
        type=str,
        default=None,
        help="(Legacy giraffe mode) Partition to process: query or reference",
    )
    parser.add_argument(
        "--bteh",
        action="store_true",
        default=False,
        help=(
            "Run normalized BTEH detection route (reads image manifest, writes "
            "crop manifest). Pass --help after --bteh for full options."
        ),
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        default=False,
        help=(
            "Run the source-agnostic normalized elephant route. Pass --help "
            "after --normalized for full options."
        ),
    )
    parser.add_argument(
        "--legacy-giraffe",
        action="store_true",
        default=False,
        help="Run the legacy positional giraffe crop pipeline.",
    )
    # Parse only the mode flags first so --bteh can hand off to its own parser.
    mode_args, remaining = parser.parse_known_args()

    selected_modes = sum(
        (mode_args.bteh, mode_args.normalized, mode_args.legacy_giraffe)
    )
    if selected_modes > 1:
        parser.error(
            "--bteh, --normalized, and --legacy-giraffe are mutually exclusive"
        )
    if mode_args.bteh:
        sys.argv = [sys.argv[0]] + remaining
        _bteh_main()
    elif mode_args.normalized:
        sys.argv = [sys.argv[0]] + remaining
        _normalized_main()
    elif mode_args.legacy_giraffe:
        if not mode_args.partition:
            parser.error("--partition is required with --legacy-giraffe")
        if remaining:
            parser.error(f"unrecognized arguments: {' '.join(remaining)}")
        main(mode_args.partition)
    else:
        parser.error("specify --normalized, --bteh, or --legacy-giraffe")
