# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

import os
import sys
import json
import pstats
import cProfile
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers_matching import (
    load_data_dirs,
    load_metadata_file,
    print_memory_usage,
    log_to_file,
    restore_stdout,
)
from utils.utils_embeddings import cosine_topk, load_index_parquet
from configs.config_elephant import (
    ID_COL,
    IMAGE_ID_COL,
    ACTIVE_DESCRIPTORS,
    EMBEDDINGS_SUBDIR,
    SHORTLIST_K,
    CALIBRATION_DIR,
    LOCAL_MATCHER_MIN_INLIERS,
    CROP_SUBDIR,
    INDEX_PARQUET_FILENAME,
)
from models.calibration import Calibrator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AUC (trapezoidal rule, no sklearn dependency for this metric)
# ---------------------------------------------------------------------------

def _roc_auc(scores, labels):
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order]
    tpr = np.cumsum(labels_sorted) / labels_sorted.sum()
    fpr_inc = (1 - labels_sorted) / (len(labels_sorted) - labels_sorted.sum())
    auc = float(np.dot(tpr, fpr_inc))
    return auc


# ---------------------------------------------------------------------------
# Embedding loader
# ---------------------------------------------------------------------------

def _load_ref_embeddings(root_dir, partition):
    emb_dir = os.path.join(root_dir, f"{partition}_dir", EMBEDDINGS_SUBDIR)
    matrices = {}
    for desc in ACTIVE_DESCRIPTORS:
        normalized_path = os.path.join(emb_dir, f"{desc}.npy")
        legacy_path = os.path.join(emb_dir, f"{partition}_{desc}.npy")
        npy_path = normalized_path if os.path.isfile(normalized_path) else legacy_path
        if not os.path.isfile(npy_path):
            logger.warning("Embedding file not found: %s", npy_path)
            matrices[desc] = None
        else:
            matrices[desc] = np.load(npy_path).astype(np.float32)
            logger.info("Loaded %s/%s embeddings %s", partition, desc, matrices[desc].shape)
    return matrices


def load_descriptor_mapping(desc, partition_dir):
    """Load one normalized descriptor mapping, with a warned legacy fallback."""
    embeddings_dir = os.path.join(partition_dir, EMBEDDINGS_SUBDIR)
    mapping_path = os.path.join(embeddings_dir, f"{desc}_mapping.parquet")
    if os.path.isfile(mapping_path):
        mapping = pd.read_parquet(mapping_path)
        required = {"image_id", "individual_id", "embedding_row", "crop_id"}
        missing = sorted(required - set(mapping.columns))
        if missing:
            raise ValueError(
                f"descriptor mapping {mapping_path!r} is missing columns: {missing}"
            )
        return mapping

    warning = (
        f"Normalized descriptor mapping not found for {desc!r}; "
        "falling back to the legacy wide index parquet"
    )
    logger.warning(warning)
    warnings.warn(warning, RuntimeWarning, stacklevel=2)
    partition_name = os.path.basename(os.path.normpath(partition_dir))
    if partition_name.endswith("_dir"):
        partition_name = partition_name[:-4]
    candidates = [
        os.path.join(
            embeddings_dir, f"{partition_name}_{INDEX_PARQUET_FILENAME}"
        ),
        os.path.join(embeddings_dir, INDEX_PARQUET_FILENAME),
    ]
    legacy_path = next((path for path in candidates if os.path.isfile(path)), None)
    if legacy_path is None:
        raise FileNotFoundError(
            f"neither {mapping_path!r} nor a legacy index parquet exists; tried {candidates}"
        )
    legacy = pd.read_parquet(legacy_path)
    row_column = f"{desc}_row"
    required_legacy = {IMAGE_ID_COL, ID_COL, row_column}
    missing = sorted(required_legacy - set(legacy.columns))
    if missing:
        raise ValueError(f"legacy index {legacy_path!r} is missing columns: {missing}")
    crop_ids = (
        legacy["crop_id"].astype(str)
        if "crop_id" in legacy.columns
        else legacy[IMAGE_ID_COL].astype(str) + "__legacy"
    )
    return pd.DataFrame(
        {
            "image_id": legacy[IMAGE_ID_COL].astype(str),
            "individual_id": legacy[ID_COL].astype(str),
            "embedding_row": legacy[row_column].astype(int),
            "crop_id": crop_ids,
        }
    )


# ---------------------------------------------------------------------------
# Crop loader
# ---------------------------------------------------------------------------

def _load_crop(crop_path):
    import cv2
    if not crop_path or not os.path.isfile(crop_path):
        return None
    img = cv2.imread(crop_path)
    return img


# ---------------------------------------------------------------------------
# All-pairs calibration scoring
# ---------------------------------------------------------------------------

def run_all_pairs(
    metadata_df,
    emb_matrices,
    index_df=None,
    descriptor_mappings=None,
):
    """
    Computes all pairwise cosine similarities within the reference partition.
    Labels: 1 if the pair shares the same individual_id, 0 otherwise.
    Also computes closed-set top-1 accuracy (excluding self-match).

    Returns desc_data and local_data in the format expected by main().
    """
    using_legacy_index = descriptor_mappings is None
    if descriptor_mappings is None:
        if index_df is None or index_df.empty:
            raise ValueError(
                "descriptor mappings and legacy index parquet are both unavailable"
            )
        descriptor_mappings = {}
        for desc in ACTIVE_DESCRIPTORS:
            row_column = f"{desc}_row"
            if row_column not in index_df.columns:
                raise ValueError(
                    f"legacy index parquet is missing required column {row_column!r}"
                )
            descriptor_mappings[desc] = pd.DataFrame(
                {
                    "image_id": index_df[IMAGE_ID_COL].astype(str),
                    "individual_id": index_df[ID_COL].astype(str),
                    "embedding_row": index_df[row_column].astype(int),
                    "crop_id": (
                        index_df["crop_id"].astype(str)
                        if "crop_id" in index_df.columns
                        else index_df[IMAGE_ID_COL].astype(str) + "__legacy"
                    ),
                }
            )

    desc_data = {desc: {"scores": [], "labels": [], "fold_top1": []} for desc in ACTIVE_DESCRIPTORS}
    local_data = {"scores": [], "labels": [], "fold_top1": []}

    if IMAGE_ID_COL not in metadata_df.columns or ID_COL not in metadata_df.columns:
        raise ValueError(
            f"metadata must contain {IMAGE_ID_COL!r} and {ID_COL!r} for calibration"
        )
    metadata_identities = metadata_df.set_index(
        metadata_df[IMAGE_ID_COL].astype(str)
    )[ID_COL].astype(str)

    for desc in ACTIVE_DESCRIPTORS:
        if emb_matrices.get(desc) is None:
            continue
        if desc not in descriptor_mappings:
            raise ValueError(f"descriptor mapping is missing for {desc!r}")
        mapping = descriptor_mappings[desc]
        required = {"image_id", "individual_id", "embedding_row", "crop_id"}
        missing_columns = sorted(required - set(mapping.columns))
        if missing_columns:
            raise ValueError(
                f"descriptor mapping for {desc!r} is missing columns: {missing_columns}"
            )
        if mapping.empty:
            continue

        if using_legacy_index:
            missing_mappings = sorted(
                set(metadata_df[IMAGE_ID_COL].astype(str))
                - set(mapping["image_id"].astype(str))
            )
            if missing_mappings:
                raise AssertionError(
                    f"legacy descriptor {desc!r} has metadata image_ids with no index join: "
                    f"{missing_mappings}"
                )

        unknown_images = sorted(
            set(mapping["image_id"].astype(str)) - set(metadata_identities.index)
        )
        if unknown_images:
            raise AssertionError(
                f"descriptor {desc!r} mapping image_ids are missing from metadata: "
                f"{unknown_images}"
            )
        expected_ids = mapping["image_id"].astype(str).map(metadata_identities)
        mismatches = mapping[
            expected_ids.to_numpy() != mapping["individual_id"].astype(str).to_numpy()
        ]
        if not mismatches.empty:
            values = mismatches[["image_id", "individual_id"]].to_dict("records")
            raise AssertionError(
                f"descriptor {desc!r} individual_id join mismatch: {values}"
            )

        rows = mapping["embedding_row"].astype(int).tolist()
        matrix = emb_matrices[desc]
        out_of_range = [row for row in rows if row < 0 or row >= len(matrix)]
        if out_of_range:
            raise AssertionError(
                f"descriptor {desc!r} embedding rows out of range for "
                f"matrix length {len(matrix)}: {out_of_range}"
            )
        emb = matrix[rows]
        ids = mapping["individual_id"].astype(str).to_numpy()
        n_valid = len(mapping)
        logger.info(
            "[%s] all-pairs scoring over %d crops (%d individuals).",
            desc,
            n_valid,
            len(np.unique(ids)),
        )

        # Full cosine similarity matrix (embeddings are L2-normalised)
        sim_matrix = emb @ emb.T  # (n_valid, n_valid)

        scores, labels, top1_correct = [], [], []
        for i in range(n_valid):
            for j in range(i + 1, n_valid):
                scores.append(float(sim_matrix[i, j]))
                labels.append(1 if ids[i] == ids[j] else 0)

            # Closed-set top-1 (exclude self)
            row_sims = sim_matrix[i].copy()
            row_sims[i] = -np.inf
            top1_correct.append(1 if ids[int(np.argmax(row_sims))] == ids[i] else 0)

        desc_data[desc]["scores"] = scores
        desc_data[desc]["labels"] = labels
        desc_data[desc]["fold_top1"] = top1_correct

        n_pos = int(sum(labels))
        n_neg = len(labels) - n_pos
        top1_acc = float(np.mean(top1_correct))
        logger.info("[%s] %d pos pairs, %d neg pairs | closed-set top-1: %.3f",
                    desc, n_pos, n_neg, top1_acc)

    return desc_data, local_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train WildFusion calibrators (step 4b)")
    parser.add_argument(
        "--partition",
        type=str,
        default="reference",
        help="Partition to train calibrators on (LOIO within it). Default: reference",
    )
    parser.add_argument(
        "--session-col",
        type=str,
        default=None,
        dest="session_col",
        help="Optional column name to guard same-session leakage in LOIO splits",
    )
    parser.add_argument(
        "--skip-local",
        action="store_true",
        default=False,
        help="Skip local matcher scoring (use when LocalMatcher deps are not available)",
    )
    args = parser.parse_args()

    profiler = cProfile.Profile()
    profiler.enable()

    root_dir, _ = load_data_dirs()

    log_file_std_output, log_file_err_output = log_to_file(
        root_dir, "train_calibration", subdir=f"{args.partition}_dir"
    )

    logging.basicConfig(level=logging.INFO)

    partition = args.partition
    partition_dir = os.path.join(root_dir, f"{partition}_dir")

    # Load metadata
    metadata_filepath = os.path.join(partition_dir, f"metadata_{partition}.csv")
    metadata_df = load_metadata_file(metadata_filepath)

    n_images = len(metadata_df)
    n_individuals = metadata_df[ID_COL].nunique() if ID_COL in metadata_df.columns else 0
    print(f"Loaded metadata: {n_images} images, {n_individuals} individuals")

    if n_individuals < 2:
        print(f"ERROR: need at least 2 individuals for LOIO calibration, found {n_individuals}.")
        restore_stdout(log_file_std_output, log_file_err_output)
        sys.exit(1)

    # Load embeddings
    emb_matrices = _load_ref_embeddings(root_dir, partition)

    # Ensure metadata has IMAGE_ID_COL
    if IMAGE_ID_COL not in metadata_df.columns:
        metadata_df[IMAGE_ID_COL] = metadata_df["path_relative_to_root"].apply(
            lambda p: os.path.splitext(os.path.basename(p))[0]
        )

    # Run all-pairs scoring for calibration
    descriptor_mappings = {
        desc: load_descriptor_mapping(desc, partition_dir)
        for desc in ACTIVE_DESCRIPTORS
    }
    desc_data, local_data = run_all_pairs(
        metadata_df,
        emb_matrices,
        descriptor_mappings=descriptor_mappings,
    )

    # Top-1 accuracy report (closed-set, per image)
    all_top1 = desc_data[ACTIVE_DESCRIPTORS[0]]["fold_top1"]
    if all_top1:
        top1_mean = float(np.mean(all_top1))
        top1_std = float(np.std(all_top1))
    else:
        top1_mean = float("nan")
        top1_std = float("nan")
    print(f"\nClosed-set top-1 accuracy: mean={top1_mean:.4f}  std={top1_std:.4f}  (over {len(all_top1)} images)")

    # Fit and save calibrators
    calib_dir = os.path.join(root_dir, CALIBRATION_DIR)
    os.makedirs(calib_dir, exist_ok=True)

    manifest_descriptors = {}

    channels = list(ACTIVE_DESCRIPTORS) + ([] if args.skip_local else ["local"])

    for channel in channels:
        if channel == "local":
            all_scores = np.array(local_data["scores"], dtype=np.float64)
            all_labels = np.array(local_data["labels"], dtype=np.float64)
        else:
            all_scores = np.array(desc_data[channel]["scores"], dtype=np.float64)
            all_labels = np.array(desc_data[channel]["labels"], dtype=np.float64)

        n_positive = int(all_labels.sum())
        n_negative = int((1 - all_labels).sum())
        total = len(all_labels)

        if total == 0:
            print(f"\n[{channel}] No pairs collected — skipping calibrator.")
            continue

        auc = _roc_auc(all_scores, all_labels)

        cal = Calibrator()
        try:
            cal.fit(all_scores, all_labels)
        except Exception as exc:
            print(f"\n[{channel}] Calibrator fit failed: {exc}. Skipping.")
            continue

        save_path = os.path.join(calib_dir, f"{channel}.pkl")
        cal.save(save_path)

        print(
            f"\n[{channel}] method={cal.method}  n_positive={n_positive}  "
            f"n_negative={n_negative}  LOIO AUC={auc:.4f}  saved→{save_path}"
        )

        manifest_descriptors[channel] = {
            "method": cal.method,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "auc": round(auc, 6) if not np.isnan(auc) else None,
        }

    # Save calibration manifest
    manifest = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "partition": partition,
        "n_individuals": n_individuals,
        "n_images": n_images,
        "descriptors": manifest_descriptors,
        "loio_top1_mean": round(top1_mean, 6) if not np.isnan(top1_mean) else None,
        "loio_top1_std": round(top1_std, 6) if not np.isnan(top1_std) else None,
    }
    manifest_path = os.path.join(calib_dir, "calibration_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nCalibration manifest saved to {manifest_path}")

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(20)

    print_memory_usage()

    restore_stdout(log_file_std_output, log_file_err_output)


if __name__ == "__main__":
    main()
