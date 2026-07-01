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
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers_matching import print_memory_usage, log_to_file, restore_stdout
from utils.helpers_matching import load_data_dirs, load_metadata_file
from models.detector import ElephantDetector
from configs.config_elephant import VIEWPOINT_COL, CROP_SUBDIR


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


def _check_if_processed_image_exists(path_relative_to_root: str, processed_img_dir: str) -> bool:
    return os.path.isfile(_crop_output_path(path_relative_to_root, processed_img_dir))


def run_detection(metadata_table, metadata_filepath, input_img_dir, processed_img_dir, detector, crop_size):

    zoomed_dir = os.path.join(processed_img_dir, "zoomed_version")
    os.makedirs(zoomed_dir, exist_ok=True)
    metadata_table["ai_found_torso"] = metadata_table["ai_found_torso"].astype(object)

    for idx, row in tqdm(metadata_table.iterrows(), total=metadata_table.shape[0]):

        if idx % 100 == 0:
            metadata_table.to_csv(metadata_filepath, index=False)

        path_rel = row["path_relative_to_root"]
        full_path = os.path.join(input_img_dir, path_rel)

        if not os.path.exists(full_path):
            metadata_table.loc[idx, "ai_found_torso"] = "file_not_found"
            continue

        if _check_if_processed_image_exists(path_rel, processed_img_dir):
            metadata_table.loc[idx, "ai_found_torso"] = "existing_item"
            continue

        image_bgr = cv2.imread(full_path)
        if image_bgr is None:
            metadata_table.loc[idx, "ai_found_torso"] = "file_not_found"
            continue

        crop, viewpoint = detector.crop(image_bgr)
        metadata_table.loc[idx, VIEWPOINT_COL] = viewpoint

        if crop is not None:
            # Resize to square crop_size × crop_size (same output contract as original step_1)
            crop_resized = cv2.resize(crop, (crop_size, crop_size))
            out_path = _crop_output_path(path_rel, processed_img_dir)
            cv2.imwrite(out_path, crop_resized)
            metadata_table.loc[idx, "ai_found_torso"] = "True"
        else:
            metadata_table.loc[idx, "ai_found_torso"] = "False"

    return metadata_table


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

    # crop_size default mirrors the giraffe pipeline's cropped_img_size
    crop_size = 512

    metadata_table = run_detection(
        metadata_table, metadata_filepath, input_img_dir, processed_img_dir, detector, crop_size
    )

    metadata_table.to_csv(metadata_filepath, index=False)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats()

    print_memory_usage()

    restore_stdout(log_file_std_output, log_file_err_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run elephant detector to crop images")
    parser.add_argument(
        "--partition",
        type=str,
        default="query",
        help="Partition to process: query or reference",
    )
    args = parser.parse_args()
    main(args.partition)
