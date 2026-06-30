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
    make_loio_splits,
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
        npy_path = os.path.join(emb_dir, f"{partition}_{desc}.npy")
        if not os.path.isfile(npy_path):
            logger.warning("Embedding file not found: %s", npy_path)
            matrices[desc] = None
        else:
            matrices[desc] = np.load(npy_path).astype(np.float32)
            logger.info("Loaded %s/%s embeddings %s", partition, desc, matrices[desc].shape)
    return matrices


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
# LOIO calibration loop
# ---------------------------------------------------------------------------

def run_loio(metadata_df, emb_matrices, index_df, partition, root_dir, skip_local, session_col):
    individuals = metadata_df[ID_COL].dropna().unique()
    n_individuals = len(individuals)

    if n_individuals < 2:
        logger.error("Need at least 2 individuals for LOIO; found %d. Aborting.", n_individuals)
        sys.exit(1)

    # Build image_id → index-parquet row lookup
    if index_df.empty:
        logger.error("Index parquet is empty. Run step_2 for the reference partition first.")
        sys.exit(1)

    img_id_to_idx_row = {}
    for _, irow in index_df.iterrows():
        img_id_to_idx_row[str(irow[IMAGE_ID_COL])] = irow

    # Accumulators: {desc: {"scores": [], "labels": [], "fold_top1": []}}
    desc_data = {desc: {"scores": [], "labels": [], "fold_top1": []} for desc in ACTIVE_DESCRIPTORS}
    local_data = {"scores": [], "labels": [], "fold_top1": []}

    local_matcher = None
    if not skip_local:
        try:
            from models.local_matcher import LocalMatcher
            local_matcher = LocalMatcher()
        except Exception as exc:
            logger.warning("LocalMatcher unavailable (%s). Skipping local scoring.", exc)
            skip_local = True

    for fold_idx, (gallery_df, probe_df) in enumerate(
        make_loio_splits(metadata_df, ID_COL, session_col)
    ):
        held_out_id = probe_df[ID_COL].iloc[0]
        logger.info("LOIO fold %d/%d — held-out: %s (%d probe, %d gallery)",
                    fold_idx + 1, n_individuals, held_out_id, len(probe_df), len(gallery_df))

        if len(gallery_df) == 0:
            logger.warning("Empty gallery for fold %d. Skipping.", fold_idx + 1)
            continue

        gallery_ids = gallery_df[IMAGE_ID_COL].astype(str).tolist()
        gallery_individual_ids = gallery_df[ID_COL].tolist()

        # For each descriptor: build gallery embedding matrix for this fold
        gallery_emb = {}
        gallery_rows = []
        for img_id in gallery_ids:
            irow = img_id_to_idx_row.get(img_id)
            if irow is None:
                gallery_rows.append(None)
            else:
                gallery_rows.append(irow)

        valid_gallery_mask = [r is not None for r in gallery_rows]
        valid_gallery_individual_ids = [
            gallery_individual_ids[i] for i in range(len(gallery_rows)) if valid_gallery_mask[i]
        ]
        valid_gallery_img_ids = [
            gallery_ids[i] for i in range(len(gallery_rows)) if valid_gallery_mask[i]
        ]

        for desc in ACTIVE_DESCRIPTORS:
            if emb_matrices[desc] is None:
                gallery_emb[desc] = None
                continue
            rows_for_desc = []
            for i, irow in enumerate(gallery_rows):
                if irow is None:
                    continue
                row_col = f"{desc}_row"
                if row_col not in irow.index:
                    continue
                rows_for_desc.append(int(irow[row_col]))
            if rows_for_desc:
                gallery_emb[desc] = emb_matrices[desc][rows_for_desc]
            else:
                gallery_emb[desc] = None

        # Per-probe scoring
        fold_desc_scores = {desc: {"scores": [], "labels": []} for desc in ACTIVE_DESCRIPTORS}
        fold_local = {"scores": [], "labels": []}
        fold_top1_correct = []

        for _, probe_row in probe_df.iterrows():
            probe_img_id = str(probe_row[IMAGE_ID_COL]) if IMAGE_ID_COL in probe_row.index else ""
            probe_individual_id = probe_row[ID_COL]

            probe_idx_row = img_id_to_idx_row.get(probe_img_id)
            if probe_idx_row is None:
                logger.warning("No index parquet row for probe image_id='%s'. Skipping.", probe_img_id)
                continue

            # Gather cosine scores per descriptor
            per_desc_top_scores = {}
            per_desc_top_indices = {}

            for desc in ACTIVE_DESCRIPTORS:
                if emb_matrices[desc] is None or gallery_emb[desc] is None:
                    continue
                row_col = f"{desc}_row"
                if row_col not in probe_idx_row.index:
                    logger.warning("Column %s missing in index parquet for probe %s.", row_col, probe_img_id)
                    continue
                probe_emb_row = int(probe_idx_row[row_col])
                probe_vec = emb_matrices[desc][probe_emb_row]

                k = min(SHORTLIST_K, len(valid_gallery_individual_ids))
                sims, top_idxs = cosine_topk(probe_vec, gallery_emb[desc], k)

                per_desc_top_scores[desc] = sims
                per_desc_top_indices[desc] = top_idxs

                top_individual_ids = [valid_gallery_individual_ids[i] for i in top_idxs]
                is_same = np.array([gid == probe_individual_id for gid in top_individual_ids], dtype=np.float32)

                fold_desc_scores[desc]["scores"].extend(sims.tolist())
                fold_desc_scores[desc]["labels"].extend(is_same.tolist())

            # Top-1 accuracy (use first available descriptor as reference ranking)
            ref_desc = next((d for d in ACTIVE_DESCRIPTORS if d in per_desc_top_indices), None)
            if ref_desc is not None:
                top1_individual = valid_gallery_individual_ids[per_desc_top_indices[ref_desc][0]]
                fold_top1_correct.append(int(top1_individual == probe_individual_id))

            # Local matching on shortlisted candidates (first descriptor's ranking)
            if not skip_local and local_matcher is not None and ref_desc is not None:
                probe_crop_path = (
                    str(probe_idx_row["crop_path"])
                    if "crop_path" in probe_idx_row.index and pd.notna(probe_idx_row["crop_path"])
                    else ""
                )
                probe_crop = _load_crop(probe_crop_path)
                if probe_crop is None:
                    logger.debug("Probe crop not found for '%s'. Skipping local.", probe_img_id)
                else:
                    top_idxs_ref = per_desc_top_indices[ref_desc]
                    for rank_pos, gal_i in enumerate(top_idxs_ref):
                        gal_img_id = valid_gallery_img_ids[gal_i]
                        gal_idx_row = img_id_to_idx_row.get(gal_img_id)
                        if gal_idx_row is None:
                            continue
                        gal_crop_path = (
                            str(gal_idx_row["crop_path"])
                            if "crop_path" in gal_idx_row.index and pd.notna(gal_idx_row["crop_path"])
                            else ""
                        )
                        gal_crop = _load_crop(gal_crop_path)
                        if gal_crop is None:
                            continue
                        try:
                            n_inliers, _ = local_matcher.score(probe_crop, gal_crop)
                        except Exception as exc:
                            logger.debug("Local matcher error: %s", exc)
                            n_inliers = 0
                        gal_individual_id = valid_gallery_individual_ids[gal_i]
                        is_same_local = int(gal_individual_id == probe_individual_id)
                        fold_local["scores"].append(float(n_inliers))
                        fold_local["labels"].append(is_same_local)

        # Accumulate fold data
        for desc in ACTIVE_DESCRIPTORS:
            desc_data[desc]["scores"].extend(fold_desc_scores[desc]["scores"])
            desc_data[desc]["labels"].extend(fold_desc_scores[desc]["labels"])
        if fold_local["scores"]:
            local_data["scores"].extend(fold_local["scores"])
            local_data["labels"].extend(fold_local["labels"])
        if fold_top1_correct:
            desc_data[ACTIVE_DESCRIPTORS[0]]["fold_top1"].append(np.mean(fold_top1_correct))

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

    # Load index parquet
    index_df = load_index_parquet(root_dir, partition)
    if index_df.empty:
        # Fall back to the naming convention used in step_2 ({partition}_{INDEX_PARQUET_FILENAME})
        from configs.config_elephant import INDEX_PARQUET_FILENAME
        alt_path = os.path.join(partition_dir, EMBEDDINGS_SUBDIR, f"{partition}_{INDEX_PARQUET_FILENAME}")
        if os.path.isfile(alt_path):
            index_df = pd.read_parquet(alt_path)
            print(f"Loaded index parquet from alternate path: {alt_path}")
        else:
            print("ERROR: index parquet not found. Run step_2 for the reference partition first.")
            restore_stdout(log_file_std_output, log_file_err_output)
            sys.exit(1)

    # Ensure IMAGE_ID_COL is present in index_df
    if IMAGE_ID_COL not in index_df.columns and "image_id" in index_df.columns:
        index_df = index_df.rename(columns={"image_id": IMAGE_ID_COL})

    # Ensure metadata has IMAGE_ID_COL
    if IMAGE_ID_COL not in metadata_df.columns:
        metadata_df[IMAGE_ID_COL] = metadata_df["path_relative_to_root"].apply(
            lambda p: os.path.splitext(os.path.basename(p))[0]
        )

    # Run LOIO loop
    desc_data, local_data = run_loio(
        metadata_df, emb_matrices, index_df, partition, root_dir,
        args.skip_local, args.session_col,
    )

    # Top-1 accuracy report across folds
    all_top1 = desc_data[ACTIVE_DESCRIPTORS[0]]["fold_top1"]
    if all_top1:
        top1_mean = float(np.mean(all_top1))
        top1_std = float(np.std(all_top1))
    else:
        top1_mean = float("nan")
        top1_std = float("nan")
    print(f"\nLOIO top-1 accuracy: mean={top1_mean:.4f}  std={top1_std:.4f}  (over {len(all_top1)} folds)")

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
