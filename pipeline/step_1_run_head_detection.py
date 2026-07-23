# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Experiment head-region detection pipeline (step 1 — head).

Generates head crop records from accepted body crops (preferred) or the
original source image (fallback).  Outputs are written to a **separate**
experiment namespace under ``artifacts/v1/experiments/full_local_ensemble/``
and never mutate production selected-v1 crop manifests.

Terminal slot statuses (will not be retried on resume):
  ``accepted``       – head crop found and saved.
  ``none_detected``  – detector ran but found no head; slot genuinely empty.
  ``not_applicable`` – parent body slot was none_detected / unavailable.
  ``failed``         – unexpected error; retryable.

Source tracking:
  ``source_used``    – ``"body_crop"`` when a v1 body crop was available,
                       ``"original"`` when falling back to the source image.

Detector provenance:
  ``detector_fingerprint`` – short SHA-256 of the detection hyperparameters.
"""

import argparse
import hashlib
import json
import logging
import os
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config_bteh import (
    ARTIFACT_SCHEMA_VERSION,
    BTEH_SOURCE_ROOT,
    EXPERIMENT_ROOT,
    HEAD_CROPS_SUBDIR,
    HEAD_MANIFEST_FILENAME,
)
from utils.artifact_schema import (
    HEAD_EXPERIMENT_MANIFEST_COLUMNS,
    HEAD_EXPERIMENT_MANIFEST_DTYPES,
    TERMINAL_CROP_STATUSES,
    assert_head_experiment_manifest_integrity,
    make_crop_id,
)

logger = logging.getLogger(__name__)

HEAD_MANIFEST_FILENAME_DEFAULT = HEAD_MANIFEST_FILENAME


# ---------------------------------------------------------------------------
# Detector fingerprint
# ---------------------------------------------------------------------------

def head_detector_fingerprint(
    conf_threshold: float,
    min_area_frac: float,
    max_area_frac: float,
    min_aspect: float,
    max_aspect: float,
    iou_threshold: float,
    pad_frac: float,
    prompt: str = "elephant head.",
) -> str:
    """Return a short (16-char) SHA-256 hex of the head detector hyperparameters."""
    payload = json.dumps(
        {
            "prompt": prompt,
            "conf_threshold": conf_threshold,
            "min_area_frac": min_area_frac,
            "max_area_frac": max_area_frac,
            "min_aspect": min_aspect,
            "max_aspect": max_aspect,
            "iou_threshold": iou_threshold,
            "pad_frac": pad_frac,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _empty_head_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {col: pd.Series(dtype=dtype) for col, dtype in HEAD_EXPERIMENT_MANIFEST_DTYPES.items()}
    )


def load_or_init_head_manifest(manifest_path: str) -> pd.DataFrame:
    """Load an existing head manifest or return an empty normalized manifest."""
    if not os.path.isfile(manifest_path):
        return _empty_head_manifest()
    df = pd.read_parquet(manifest_path)
    missing = [c for c in HEAD_EXPERIMENT_MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"head manifest {manifest_path!r} is missing columns: {missing}"
        )
    return df


def save_head_manifest(df: pd.DataFrame, manifest_path: str) -> None:
    """Write the normalized head experiment manifest parquet."""
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    normalized = df.copy()
    for col, dtype in HEAD_EXPERIMENT_MANIFEST_DTYPES.items():
        if col in normalized.columns:
            normalized[col] = normalized[col].astype(dtype)
    normalized.to_parquet(manifest_path, index=False)


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------

def _head_record(
    *,
    image_id: str,
    individual_id: str,
    crop_path: str,
    schema_version: str,
    source_fingerprint: str | None,
    split_fingerprint: str | None,
    detector_fingerprint: str | None,
    source_used: str,
    detector_status: str = "accepted",
    confidence: float | None = None,
    box: list[int] | None = None,
) -> dict:
    return {
        "crop_id": make_crop_id(image_id, "head", 0),
        "image_id": image_id,
        "individual_id": individual_id,
        "crop_kind": "head",
        "crop_ordinal": 0,
        "crop_path": os.path.abspath(crop_path),
        "detector_confidence": confidence,
        "detector_box": json.dumps(box, separators=(",", ":")) if box is not None else None,
        "detector_status": detector_status,
        "review_status": "pending",
        "schema_version": schema_version,
        "source_fingerprint": source_fingerprint,
        "split_fingerprint": split_fingerprint,
        "source_used": source_used,
        "detector_fingerprint": detector_fingerprint,
    }


def make_head_crop_path(image_id: str, crops_dir: str) -> str:
    """Return the canonical absolute head crop path for an image."""
    return os.path.join(
        os.path.abspath(crops_dir), f"{make_crop_id(image_id, 'head', 0)}.jpg"
    )


# ---------------------------------------------------------------------------
# Source image resolution
# ---------------------------------------------------------------------------

def _source_image_path(row: pd.Series) -> str:
    for col in ("source_path", "absolute_path", "image_path", "path"):
        value = row.get(col)
        if pd.notna(value) and str(value):
            return os.path.abspath(str(value))
    relative = row.get("source_relative_path")
    if pd.isna(relative) or not str(relative):
        raise ValueError(
            f"image_id={row.get('image_id')!r} has no source image path column"
        )
    relative = str(relative)
    return relative if os.path.isabs(relative) else str(BTEH_SOURCE_ROOT / relative)


# ---------------------------------------------------------------------------
# Core detection runner
# ---------------------------------------------------------------------------

def run_head_detection(
    image_manifest: pd.DataFrame,
    body_manifest: "pd.DataFrame | None",
    crops_dir: str,
    manifest_path: str,
    head_detector,
    crop_size: int = 512,
    schema_version: str = ARTIFACT_SCHEMA_VERSION,
    source_fingerprint: str | None = None,
    split_fingerprint: str | None = None,
    detector_fingerprint: str | None = None,
    conf_threshold: float = 0.30,
    min_area_frac: float = 0.02,
    max_area_frac: float = 0.70,
    min_aspect: float = 0.40,
    max_aspect: float = 2.50,
    iou_threshold: float = 0.50,
    pad_frac: float = 0.10,
) -> pd.DataFrame:
    """Run resumable experiment head detection.

    For each included image the pipeline attempts detection in this order:

    1. **Body-crop first**: if the v1 ``body_manifest`` contains an accepted
       body crop for the image, that crop is the detection input
       (``source_used="body_crop"``).
    2. **Original fallback**: if no accepted body crop exists, the full source
       image is used (``source_used="original"``).

    The head slot (``head_0``) gets a terminal status of ``accepted``,
    ``none_detected``, or ``not_applicable``. ``failed`` remains retryable.

    ``not_applicable`` is written when the body slot was ``none_detected`` in
    the v1 manifest AND no source image was readable, making head detection
    impossible.

    Outputs are written under ``crops_dir`` and ``manifest_path`` — both must
    be inside the experiment namespace, not the production v1 subtree.

    Production selected-v1 artifacts are **never** read for mutation; only
    accepted body-crop paths are read for pixel data.
    """
    # ------------------------------------------------------------------
    # Pre-loop guard: fail clearly before touching any I/O when the model
    # is unavailable and was explicitly required.
    # ------------------------------------------------------------------
    if getattr(head_detector, "_available", True) is False:
        raise RuntimeError(
            "HeadDetector model is unavailable. Cannot run head detection. "
            "Either fix the model or disable head generation."
        )

    # ------------------------------------------------------------------
    # Validate required manifest columns.
    # ------------------------------------------------------------------
    if "include_status" not in image_manifest.columns:
        raise ValueError("image manifest is missing required column 'include_status'")
    if "image_id" not in image_manifest.columns:
        raise ValueError("image manifest is missing required column 'image_id'")
    if "individual_id" not in image_manifest.columns:
        raise ValueError("image manifest is missing required column 'individual_id'")

    included = image_manifest[
        image_manifest["include_status"].isin({"included", "duplicate_primary"})
    ].copy()
    included = included[
        included["individual_id"].notna()
        & included["individual_id"].astype(str).str.strip().ne("")
        & included["individual_id"].astype(str).ne("unresolved")
    ]
    if "_pilot_role" in included.columns:
        included = included[included["_pilot_role"] == "pilot"]

    missing_ids = (
        included["image_id"].isna()
        | included["image_id"].astype(str).str.strip().eq("")
    )
    if missing_ids.any():
        raise ValueError(
            f"included rows have missing image_id values at indices "
            f"{included.index[missing_ids].tolist()}"
        )

    id_to_individual: dict[str, str] = {
        str(row["image_id"]): str(row["individual_id"])
        for _, row in included.iterrows()
    }

    # Build body-crop lookup: image_id → crop_path (accepted body crops only).
    body_crop_paths: dict[str, str] = {}
    if body_manifest is not None and not body_manifest.empty:
        accepted_bodies = body_manifest[
            (body_manifest["crop_kind"] == "body")
            & (body_manifest["detector_status"] == "accepted")
        ]
        for _, brow in accepted_bodies.iterrows():
            iid = str(brow["image_id"])
            body_crop_paths[iid] = str(brow["crop_path"])

    # Compute detector_fingerprint if not supplied.
    if detector_fingerprint is None:
        detector_fingerprint = head_detector_fingerprint(
            conf_threshold=conf_threshold,
            min_area_frac=min_area_frac,
            max_area_frac=max_area_frac,
            min_aspect=min_aspect,
            max_aspect=max_aspect,
            iou_threshold=iou_threshold,
            pad_frac=pad_frac,
            prompt=getattr(head_detector, "prompt", "elephant head."),
        )

    head_df = load_or_init_head_manifest(manifest_path)
    existing_fingerprints = {
        value
        for value in head_df["detector_fingerprint"].dropna().astype(str).unique()
        if value
    }
    if existing_fingerprints and existing_fingerprints != {detector_fingerprint}:
        raise ValueError(
            "Cannot resume head detection with different detector parameters: "
            f"existing={sorted(existing_fingerprints)}, current={detector_fingerprint}. "
            "Start a fresh experiment manifest instead."
        )
    records = head_df.to_dict("records")

    def _terminal_ids() -> set[str]:
        return {
            str(r["crop_id"])
            for r in records
            if r.get("detector_status") in TERMINAL_CROP_STATUSES
        }

    for processed, (_, row) in enumerate(included.iterrows(), start=1):
        image_id = str(row["image_id"])
        individual_id = id_to_individual[image_id]
        head_id = make_crop_id(image_id, "head", 0)
        crop_path = make_head_crop_path(image_id, crops_dir)

        # Skip images whose head slot already has a terminal status.
        if head_id in _terminal_ids():
            continue

        # ---- Resolve detection input: body-crop first, original fallback ----
        source_used: str
        image_bgr: np.ndarray | None = None

        if image_id in body_crop_paths:
            body_path = body_crop_paths[image_id]
            if os.path.isfile(body_path):
                image_bgr = cv2.imread(body_path)
                source_used = "body_crop"

        if image_bgr is None:
            # Fallback to original source image.
            try:
                src_path = _source_image_path(row)
            except ValueError:
                src_path = None

            if src_path and os.path.isfile(src_path):
                image_bgr = cv2.imread(src_path)
                source_used = "original"
            else:
                source_used = "original"

        if image_bgr is None:
            # Neither body crop nor source image readable → not_applicable.
            records = [r for r in records if r.get("crop_id") != head_id]
            records.append(
                _head_record(
                    image_id=image_id,
                    individual_id=individual_id,
                    crop_path=crop_path,
                    schema_version=schema_version,
                    source_fingerprint=source_fingerprint,
                    split_fingerprint=split_fingerprint,
                    detector_fingerprint=detector_fingerprint,
                    source_used=source_used,
                    detector_status="not_applicable",
                )
            )
            if processed % 50 == 0:
                save_head_manifest(
                    pd.DataFrame(records, columns=HEAD_EXPERIMENT_MANIFEST_COLUMNS),
                    manifest_path,
                )
            continue

        # ---- Run detection -----------------------------------------------
        try:
            detection = head_detector.detect_head(
                image_bgr,
                conf_threshold=conf_threshold,
                min_area_frac=min_area_frac,
                max_area_frac=max_area_frac,
                min_aspect=min_aspect,
                max_aspect=max_aspect,
                iou_threshold=iou_threshold,
                pad_frac=pad_frac,
            )
        except Exception as exc:
            logger.error("Head detection failed for image_id=%s: %s", image_id, exc)
            records = [r for r in records if r.get("crop_id") != head_id]
            records.append(
                _head_record(
                    image_id=image_id,
                    individual_id=individual_id,
                    crop_path=crop_path,
                    schema_version=schema_version,
                    source_fingerprint=source_fingerprint,
                    split_fingerprint=split_fingerprint,
                    detector_fingerprint=detector_fingerprint,
                    source_used=source_used,
                    detector_status="failed",
                )
            )
        else:
            if detection is not None:
                crop = detection.get("crop")
                if crop is not None and crop.size > 0:
                    resized = cv2.resize(crop, (crop_size, crop_size))
                    os.makedirs(os.path.dirname(crop_path), exist_ok=True)
                    if not cv2.imwrite(crop_path, resized):
                        raise OSError(f"failed to write head crop to {crop_path}")
                    records = [r for r in records if r.get("crop_id") != head_id]
                    records.append(
                        _head_record(
                            image_id=image_id,
                            individual_id=individual_id,
                            crop_path=crop_path,
                            schema_version=schema_version,
                            source_fingerprint=source_fingerprint,
                            split_fingerprint=split_fingerprint,
                            detector_fingerprint=detector_fingerprint,
                            source_used=source_used,
                            detector_status="accepted",
                            confidence=float(detection["score"]),
                            box=[int(v) for v in detection["box"]],
                        )
                    )
                else:
                    detection = None

            if detection is None:
                records = [r for r in records if r.get("crop_id") != head_id]
                records.append(
                    _head_record(
                        image_id=image_id,
                        individual_id=individual_id,
                        crop_path=crop_path,
                        schema_version=schema_version,
                        source_fingerprint=source_fingerprint,
                        split_fingerprint=split_fingerprint,
                        detector_fingerprint=detector_fingerprint,
                        source_used=source_used,
                        detector_status="none_detected",
                    )
                )

        if processed % 50 == 0:
            save_head_manifest(
                pd.DataFrame(records, columns=HEAD_EXPERIMENT_MANIFEST_COLUMNS),
                manifest_path,
            )

    head_df = pd.DataFrame(records, columns=HEAD_EXPERIMENT_MANIFEST_COLUMNS)
    if head_df.empty:
        head_df = _empty_head_manifest()

    assert_head_experiment_manifest_integrity(
        head_df,
        image_manifest,
        schema_version=schema_version,
        expected_detector_fingerprint=detector_fingerprint,
        expected_source_fingerprint=source_fingerprint,
        expected_split_fingerprint=split_fingerprint,
    )
    save_head_manifest(head_df, manifest_path)
    return load_or_init_head_manifest(manifest_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _head_main() -> None:
    """Experiment head-region generation CLI (step 1 --head)."""
    parser = argparse.ArgumentParser(
        description=(
            "Experiment head-region detection (step 1 --head): generate head crops "
            "from accepted v1 body crops (primary) or original source images (fallback). "
            "Outputs are written under experiments/full_local_ensemble/ and never "
            "mutate production selected-v1 artifacts."
        )
    )
    parser.add_argument(
        "--image-manifest",
        required=True,
        help="Path to the canonical BTEH image manifest parquet.",
    )
    parser.add_argument(
        "--body-manifest",
        default=None,
        help=(
            "Path to the production v1 crop manifest parquet (read-only). "
            "Accepted body crops are used as the primary head-detection input."
        ),
    )
    parser.add_argument(
        "--head-manifest",
        default=str(EXPERIMENT_ROOT / HEAD_MANIFEST_FILENAME),
        help="Path to write (or resume) the experiment head manifest parquet.",
    )
    parser.add_argument(
        "--crops-dir",
        default=str(EXPERIMENT_ROOT / HEAD_CROPS_SUBDIR),
        help="Directory to write generated head crop images.",
    )
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--schema-version", default=ARTIFACT_SCHEMA_VERSION)
    parser.add_argument(
        "--source-fingerprint",
        required=True,
        help="SHA-256 fingerprint of the source image set.",
    )
    parser.add_argument(
        "--split-fingerprint",
        required=True,
        help="SHA-256 fingerprint of the split assignment.",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.30)
    parser.add_argument("--min-area-frac", type=float, default=0.02)
    parser.add_argument("--max-area-frac", type=float, default=0.70)
    parser.add_argument("--min-aspect", type=float, default=0.40)
    parser.add_argument("--max-aspect", type=float, default=2.50)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--pad-frac", type=float, default=0.10)
    parser.add_argument("--prompt", default="elephant head.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (cuda/cpu). Defaults to cuda when available.",
    )
    parser.add_argument(
        "--disable-cudnn",
        action="store_true",
        help="Disable cuDNN for hosts with incompatible CUDA kernels.",
    )
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False

    from models.detector import HeadDetector, _GroundingDINOBackend

    backend = _GroundingDINOBackend(device=device)
    head_detector = HeadDetector(backend=backend, prompt=args.prompt)

    image_manifest = pd.read_parquet(args.image_manifest)
    body_manifest: pd.DataFrame | None = None
    if args.body_manifest:
        body_manifest = pd.read_parquet(args.body_manifest)

    run_head_detection(
        image_manifest=image_manifest,
        body_manifest=body_manifest,
        crops_dir=args.crops_dir,
        manifest_path=args.head_manifest,
        head_detector=head_detector,
        crop_size=args.crop_size,
        schema_version=args.schema_version,
        source_fingerprint=args.source_fingerprint,
        split_fingerprint=args.split_fingerprint,
        conf_threshold=args.conf_threshold,
        min_area_frac=args.min_area_frac,
        max_area_frac=args.max_area_frac,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
        iou_threshold=args.iou_threshold,
        pad_frac=args.pad_frac,
    )


if __name__ == "__main__":
    _head_main()
