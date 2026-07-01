# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

import os
import sys
import pickle
import pstats
import cProfile
import logging
import numpy as np
import pandas as pd
import faiss
import cv2
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers_matching import load_data_dirs, load_metadata_file
from utils.helpers_matching import print_memory_usage, log_to_file, restore_stdout
from configs.config_elephant import (
    ACTIVE_DESCRIPTORS,
    EMBEDDINGS_SUBDIR,
    FAISS_SUBDIR,
    INDEX_PARQUET_FILENAME,
    CALIBRATION_DIR,
    LOCAL_MATCHER_BACKEND,
    LOCAL_MATCHER_KEYPOINTS,
    LOCAL_MATCHER_MIN_INLIERS,
    NUM_RECOMMENDED_IDS,
    MATCH_ACCEPT_THRESHOLD,
)
from models.embedder import GlobalEmbedder
from models.local_matcher import LocalMatcher
from models.calibration import Calibrator
from models.fusion import WildFusionMatcher, Recommendation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WildFusionMatcher factory
# ---------------------------------------------------------------------------

def build_wildfusion(root_dir: str) -> WildFusionMatcher:
    reference_dir = os.path.join(root_dir, "reference_dir")
    faiss_dir     = os.path.join(reference_dir, FAISS_SUBDIR)
    calib_dir     = os.path.join(root_dir, CALIBRATION_DIR)

    # Embedders — not needed when using pre-computed query embeddings (step 2 already ran).
    # Skip loading large models to avoid long initialization.
    embedders = {}

    # FAISS indexes and ref_meta
    faiss_indexes = {}
    ref_meta      = {}
    for desc in ACTIVE_DESCRIPTORS:
        index_path = os.path.join(faiss_dir, f"{desc}.index")
        if not os.path.isfile(index_path):
            logger.error("FAISS index not found: %s  — run step_2 for reference first.", index_path)
            sys.exit(1)
        faiss_indexes[desc] = faiss.read_index(index_path)

        meta_path = os.path.join(faiss_dir, f"reference_{desc}_meta.pkl")
        if not os.path.isfile(meta_path):
            logger.error("FAISS meta not found: %s", meta_path)
            sys.exit(1)
        with open(meta_path, "rb") as fh:
            ref_meta[desc] = pickle.load(fh)

    # Local matcher
    local_matcher = LocalMatcher(
        backend=LOCAL_MATCHER_BACKEND,
        max_keypoints=LOCAL_MATCHER_KEYPOINTS,
        min_inliers=LOCAL_MATCHER_MIN_INLIERS,
    )

    # Calibrators (optional — absent pre-Phase-4)
    calibrators = {}
    candidate_names = list(ACTIVE_DESCRIPTORS) + ["local"]
    for name in candidate_names:
        cal_path = os.path.join(calib_dir, f"{name}.pkl")
        if os.path.isfile(cal_path):
            cal = Calibrator().load(cal_path)
            calibrators[name] = cal
            logger.info("Loaded calibrator '%s' (method=%s).", name, cal.method)
        else:
            logger.info("Calibrator for '%s' not found at %s — skipping.", name, cal_path)

    return WildFusionMatcher(
        embedders=embedders,
        faiss_indexes=faiss_indexes,
        ref_meta=ref_meta,
        local_matcher=local_matcher,
        calibrators=calibrators,
    )


# ---------------------------------------------------------------------------
# Metadata column management
# ---------------------------------------------------------------------------

def add_columns_for_matching_results(query_metadata: pd.DataFrame) -> pd.DataFrame:
    cols = ["matching_attempt", "matching_status"]
    for i in range(1, NUM_RECOMMENDED_IDS + 1):
        cols += [
            f"match_individual_{i}",
            f"match_image_{i}",
            f"match_viewpoint_{i}",
            f"match_global_sim_{i}",
            f"match_local_count_{i}",
            f"match_fused_sim_{i}",
        ]
    str_cols = {"matching_attempt", "matching_status"}
    for i in range(1, NUM_RECOMMENDED_IDS + 1):
        str_cols |= {f"match_individual_{i}", f"match_image_{i}", f"match_viewpoint_{i}"}

    for col in cols:
        if col not in query_metadata.columns:
            query_metadata[col] = np.nan
        if col in str_cols:
            query_metadata[col] = query_metadata[col].astype(object)
    return query_metadata


def fill_matching_results(
    query_metadata: pd.DataFrame,
    query_image_path: str,
    recommendations: list,
) -> pd.DataFrame:
    matching_index = query_metadata[query_metadata["path_relative_to_root"] == query_image_path].index

    if matching_index.empty:
        return query_metadata

    status = "not_matched"
    if recommendations and recommendations[0].fused_sim >= MATCH_ACCEPT_THRESHOLD:
        status = "matched"

    query_metadata.loc[matching_index, "matching_status"] = status

    for i, rec in enumerate(recommendations[:NUM_RECOMMENDED_IDS], start=1):
        query_metadata.loc[matching_index, f"match_individual_{i}"] = rec.individual_id
        query_metadata.loc[matching_index, f"match_image_{i}"]      = rec.image_id
        query_metadata.loc[matching_index, f"match_viewpoint_{i}"]  = rec.viewpoint
        # Store mean of global sims as the representative global similarity for UI display
        global_sim_val = float(np.mean(list(rec.global_sims.values()))) if rec.global_sims else 0.0
        query_metadata.loc[matching_index, f"match_global_sim_{i}"] = global_sim_val
        query_metadata.loc[matching_index, f"match_local_count_{i}"] = rec.local_inliers
        query_metadata.loc[matching_index, f"match_fused_sim_{i}"]  = rec.fused_sim

    return query_metadata


# ---------------------------------------------------------------------------
# Query sweep
# ---------------------------------------------------------------------------

def sweep_over_query_images(
    metadata_filepath: str,
    query_metadata: pd.DataFrame,
    embeddings_per_desc: dict,
    query_index_df: pd.DataFrame,
    wildfusion: WildFusionMatcher,
) -> pd.DataFrame:
    query_metadata = add_columns_for_matching_results(query_metadata)

    # Build a quick lookup: image_id → row-index in each embedding matrix
    img_id_to_row: dict[str, int] = {}
    for _, idx_row in query_index_df.iterrows():
        img_id_to_row[str(idx_row["image_id"])] = int(idx_row.get("megadescriptor_row", idx_row.name))

    for idx, row in tqdm(query_metadata.iterrows(), total=len(query_metadata), desc="Matching"):

        if idx % 100 == 0:
            query_metadata.to_csv(metadata_filepath, index=False)

        query_image_path = row["path_relative_to_root"]
        query_metadata.loc[idx, "matching_attempt"] = "failed"

        # Skip rows that have already been processed
        if "matching_status" in row and row["matching_status"] in {"not_matched", "matched"}:
            query_metadata.loc[idx, "matching_attempt"] = "existing"
            continue

        # Resolve image_id for this row
        img_id = str(row.get("image_id", "")) if "image_id" in row.index else ""
        if not img_id:
            img_id = os.path.splitext(os.path.basename(query_image_path))[0]

        # Look up precomputed embedding row index
        emb_row = img_id_to_row.get(img_id)
        if emb_row is None:
            logger.warning("No embedding row found for image_id='%s'. Skipping.", img_id)
            continue

        # Build per-descriptor query embedding dict from pre-computed matrices
        query_embedding_per_desc = {}
        for desc in ACTIVE_DESCRIPTORS:
            if desc in embeddings_per_desc:
                query_embedding_per_desc[desc] = embeddings_per_desc[desc][emb_row]

        # Resolve crop path for local matcher
        crop_path = ""
        if "crop_path" in row.index and pd.notna(row["crop_path"]):
            crop_path = str(row["crop_path"])
        else:
            # Match the crop path reconstruction used in step_2
            orig_path = query_image_path
            parts = orig_path.rsplit(".", 1)
            img_filename = os.path.basename(orig_path)
            ext = parts[1] if len(parts) == 2 else "jpg"
            stem = img_filename.rsplit(".", 1)[0]
            root_dir, _ = load_data_dirs()
            from configs.config_elephant import CROP_SUBDIR
            crop_path = os.path.join(root_dir, CROP_SUBDIR, "zoomed_version", f"{stem}_cropped_torso_zoomed.{ext}")

        query_crop_bgr = cv2.imread(crop_path) if os.path.isfile(crop_path) else None

        if query_crop_bgr is None:
            logger.debug("Query crop not found for '%s'. Local matching will be skipped.", query_image_path)

        query_metadata.loc[idx, "matching_attempt"] = "success"

        recommendations = wildfusion.identify(query_embedding_per_desc, query_crop_bgr)
        query_metadata = fill_matching_results(query_metadata, query_image_path, recommendations)

    return query_metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    profiler = cProfile.Profile()
    profiler.enable()

    root_dir, _ = load_data_dirs()

    log_file_std_output, log_file_err_output = log_to_file(root_dir, "matching_algorithm")

    # Load query metadata
    metadata_query_filepath = os.path.join(root_dir, "query_dir", "metadata_query.csv")
    query_metadata = load_metadata_file(metadata_query_filepath)

    # Load pre-computed query embeddings and index parquet
    query_embeddings_dir = os.path.join(root_dir, "query_dir", EMBEDDINGS_SUBDIR)
    embeddings_per_desc: dict[str, np.ndarray] = {}
    for desc in ACTIVE_DESCRIPTORS:
        npy_path = os.path.join(query_embeddings_dir, f"query_{desc}.npy")
        if not os.path.isfile(npy_path):
            logger.error("Query embeddings not found: %s  — run step_2 --partition query first.", npy_path)
            sys.exit(1)
        embeddings_per_desc[desc] = np.load(npy_path)
        logger.info("Loaded query embeddings '%s' shape=%s", desc, embeddings_per_desc[desc].shape)

    query_index_parquet = os.path.join(query_embeddings_dir, f"query_{INDEX_PARQUET_FILENAME}")
    if not os.path.isfile(query_index_parquet):
        logger.error("Query index parquet not found: %s", query_index_parquet)
        sys.exit(1)
    query_index_df = pd.read_parquet(query_index_parquet)

    # Build WildFusionMatcher (loads FAISS indexes, calibrators, local matcher)
    wildfusion = build_wildfusion(root_dir)

    # Run matching
    query_metadata = sweep_over_query_images(
        metadata_query_filepath,
        query_metadata,
        embeddings_per_desc,
        query_index_df,
        wildfusion,
    )
    query_metadata.to_csv(metadata_query_filepath, index=False)
    logger.info("Matching complete. Results saved to %s", metadata_query_filepath)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats()

    print_memory_usage()

    restore_stdout(log_file_std_output, log_file_err_output)


if __name__ == "__main__":
    main()
