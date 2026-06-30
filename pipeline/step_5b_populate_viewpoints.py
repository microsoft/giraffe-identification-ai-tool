# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
import pstats
import cProfile
import argparse

import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.detector import ElephantDetector
from configs.config_elephant import VIEWPOINT_COL, VIEWPOINT_VALUES, ID_COL, IMAGE_ID_COL
from utils.helpers_matching import load_data_dirs, load_metadata_file, print_memory_usage


def _zoomed_crop_path(path_relative_to_root, processed_img_dir):
    parts = path_relative_to_root.rsplit(".", 1)
    ext = parts[1] if len(parts) == 2 else "jpg"
    stem = os.path.basename(parts[0])
    return os.path.join(
        processed_img_dir, "zoomed_version", f"{stem}_cropped_torso_zoomed.{ext}"
    )


def _load_image(path_relative_to_root, root_dir, processed_img_dir):
    crop_path = _zoomed_crop_path(path_relative_to_root, processed_img_dir)
    if os.path.isfile(crop_path):
        img = cv2.imread(crop_path)
        if img is not None:
            return img

    raw_path = os.path.join(root_dir, path_relative_to_root)
    if os.path.isfile(raw_path):
        return cv2.imread(raw_path)

    return None


def run_viewpoint_tagging(metadata_df, metadata_filepath, root_dir, processed_img_dir, detector, force):
    updated = 0
    skipped = 0
    failed = 0

    for idx, row in metadata_df.iterrows():
        current_vp = str(row.get(VIEWPOINT_COL, "unknown")).strip().lower()

        if not force and current_vp != "unknown":
            skipped += 1
            continue

        path_rel = row.get("path_relative_to_root", "")
        if not path_rel:
            failed += 1
            continue

        image_bgr = _load_image(path_rel, root_dir, processed_img_dir)
        if image_bgr is None:
            print(f"  Could not load image for row {idx}: {path_rel}")
            failed += 1
            continue

        try:
            _, viewpoint_tag = detector.crop(image_bgr)
        except Exception as exc:
            print(f"  Detector error on row {idx}: {exc}")
            failed += 1
            continue

        metadata_df.loc[idx, VIEWPOINT_COL] = viewpoint_tag
        updated += 1

        # Save every 50 rows to guard against interruption
        if updated % 50 == 0:
            metadata_df.to_csv(metadata_filepath, index=False)
            print(f"  updated={updated} skipped={skipped} failed={failed} (checkpoint saved)")

    return metadata_df, updated, skipped, failed


def main(partition, force):
    profiler = cProfile.Profile()
    profiler.enable()

    root_dir, processed_img_dir = load_data_dirs()

    metadata_filepath = os.path.join(root_dir, partition + "_dir", "metadata_" + partition + ".csv")
    metadata_df = load_metadata_file(metadata_filepath)

    if VIEWPOINT_COL not in metadata_df.columns:
        metadata_df[VIEWPOINT_COL] = "unknown"

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = ElephantDetector(backend="megadetector", conf=0.5, device=device)

    metadata_df, updated, skipped, failed = run_viewpoint_tagging(
        metadata_df, metadata_filepath, root_dir, processed_img_dir, detector, force
    )

    metadata_df.to_csv(metadata_filepath, index=False)
    print(f"\nDone: updated={updated}, skipped={skipped}, failed={failed}")
    print(f"Metadata saved to: {metadata_filepath}")

    print("\nViewpoint distribution:")
    print(metadata_df[VIEWPOINT_COL].value_counts().to_string())

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats()

    print_memory_usage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retroactively assign viewpoint tags to catalog images (step 5b)"
    )
    parser.add_argument(
        "--partition", type=str, default="reference", help="Partition name (default: reference)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-tag all rows, not just those with viewpoint == 'unknown'",
    )
    args = parser.parse_args()
    main(args.partition, args.force)
