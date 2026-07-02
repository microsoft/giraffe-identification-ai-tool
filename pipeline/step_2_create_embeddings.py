# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

import os
import sys
import pickle
import pstats
import cProfile
import argparse
import logging
import time
import cv2
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers_matching import load_data_dirs, load_metadata_file
from utils.helpers_matching import print_memory_usage, log_to_file, restore_stdout
from configs.config_elephant import (
    ACTIVE_DESCRIPTORS,
    EMBEDDINGS_SUBDIR,
    FAISS_SUBDIR,
    INDEX_PARQUET_FILENAME,
    CROP_SUBDIR,
    EAR_CROP_SUBDIR,
    EAR_DESCRIPTORS,
    ID_COL,
    IMAGE_ID_COL,
    VIEWPOINT_COL,
)
from models.embedder import GlobalEmbedder

logger = logging.getLogger(__name__)

_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Crop path helpers
# ---------------------------------------------------------------------------

def _resolve_crop_path(row: pd.Series, root_dir: str, desc: str = "") -> str:
    """
    Returns the on-disk crop path for a metadata row.
    For ear descriptors, returns the GroundingDINO ear crop path.
    For body descriptors, uses 'crop_path' column if present, otherwise
    reconstructs from the original path using the same naming convention as step_1.
    """
    orig_path = row["path_relative_to_root"]
    img_filename = os.path.basename(orig_path)
    parts = orig_path.rsplit(".", 1)
    ext = parts[1] if len(parts) == 2 else "jpg"
    stem = img_filename.rsplit(".", 1)[0]

    if desc in EAR_DESCRIPTORS:
        return os.path.join(root_dir, EAR_CROP_SUBDIR, f"{stem}_ear_cropped.{ext}")

    if "crop_path" in row.index and pd.notna(row["crop_path"]) and str(row["crop_path"]).strip():
        return str(row["crop_path"])

    crop_filename = f"{stem}_cropped_torso_zoomed.{ext}"
    return os.path.join(root_dir, CROP_SUBDIR, "zoomed_version", crop_filename)


def _image_id_for_row(row: pd.Series) -> str:
    if IMAGE_ID_COL in row.index and pd.notna(row[IMAGE_ID_COL]) and str(row[IMAGE_ID_COL]).strip():
        return str(row[IMAGE_ID_COL])
    return os.path.splitext(os.path.basename(row["path_relative_to_root"]))[0]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _apply_clahe(bgr: np.ndarray) -> np.ndarray:
    """Enhance local contrast via CLAHE on the L channel (LAB space)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab_enhanced = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def _normalize_viewpoint(bgr: np.ndarray, viewpoint: str) -> np.ndarray:
    """Flip right-facing body crops to left-facing (canonical orientation)."""
    if viewpoint == "right":
        return cv2.flip(bgr, 1)
    return bgr


# Per-descriptor embedding
# ---------------------------------------------------------------------------

def embed_partition(
    metadata_table: pd.DataFrame,
    embedder: GlobalEmbedder,
    root_dir: str,
    desc: str = "",
) -> tuple:
    """
    Loads crops, embeds them with the given embedder.
    Returns (embeddings np.ndarray (n, D), valid_indices list-of-int)
    where valid_indices corresponds to rows in metadata_table that had a readable crop.
    """
    images = []
    valid_indices = []
    is_ear = desc in EAR_DESCRIPTORS

    for idx, row in tqdm(metadata_table.iterrows(), total=len(metadata_table), desc=f"loading crops for {desc or embedder.backend}"):
        crop_path = _resolve_crop_path(row, root_dir, desc=desc)
        if os.path.isfile(crop_path):
            img = cv2.imread(crop_path)
            if img is None:
                logger.warning("cv2 could not read crop: %s", crop_path)
            else:
                if is_ear:
                    img = _apply_clahe(img)
                else:
                    viewpoint = str(row.get(VIEWPOINT_COL, "")) if VIEWPOINT_COL in row.index else ""
                    img = _normalize_viewpoint(img, viewpoint)
            images.append(img)
        elif is_ear:
            # No ear crop available — leave as zero embedding (no fallback for ears)
            logger.debug("Ear crop not found for row %s; embedding will be zero.", idx)
            images.append(None)
        else:
            # Fall back to full original image when body crop hasn't been created yet
            orig_path = os.path.normpath(os.path.join(root_dir, row["path_relative_to_root"]))
            if os.path.isfile(orig_path):
                img = cv2.imread(orig_path)
                if img is None:
                    logger.warning("cv2 could not read original: %s", orig_path)
                else:
                    viewpoint = str(row.get(VIEWPOINT_COL, "")) if VIEWPOINT_COL in row.index else ""
                    img = _normalize_viewpoint(img, viewpoint)
                    logger.debug("Using full image (no crop): %s", orig_path)
                images.append(img)
            else:
                logger.warning("Neither crop nor original found for row %s", idx)
                images.append(None)
        valid_indices.append(idx)

    # Filter out None entries for batch embedding; track positions
    valid_images = [(i, img) for i, img in enumerate(images) if img is not None]
    positions    = [t[0] for t in valid_images]
    bgr_list     = [t[1] for t in valid_images]

    n_total = len(metadata_table)
    dim     = embedder.dim
    embeddings = np.zeros((n_total, dim), dtype=np.float32)

    if bgr_list:
        logger.info("Embedding %d crops with %s...", len(bgr_list), embedder.backend)
        batch_embs = embedder.embed_batch(bgr_list, batch_size=_BATCH_SIZE)  # (n_valid, D)
        for batch_pos, meta_pos in enumerate(positions):
            embeddings[meta_pos] = batch_embs[batch_pos]

    return embeddings, valid_indices


# ---------------------------------------------------------------------------
# FAISS index builder
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(partition: str):

    profiler = cProfile.Profile()
    profiler.enable()

    root_dir, _ = load_data_dirs()

    log_file_std_output, log_file_err_output = log_to_file(root_dir, "create_embeddings")

    partition_dir = os.path.join(root_dir, f"{partition}_dir")
    embeddings_dir = os.path.join(partition_dir, EMBEDDINGS_SUBDIR)
    os.makedirs(embeddings_dir, exist_ok=True)

    metadata_filepath = os.path.join(partition_dir, f"metadata_{partition}.csv")
    metadata_table = load_metadata_file(metadata_filepath)

    # Build image_id list and body crop_path list up-front for parquet
    image_ids  = [_image_id_for_row(row) for _, row in metadata_table.iterrows()]
    crop_paths = [_resolve_crop_path(row, root_dir, desc="") for _, row in metadata_table.iterrows()]

    # Track which faiss row each metadata row maps to (per descriptor)
    row_counters: dict[str, np.ndarray] = {}

    for desc in ACTIVE_DESCRIPTORS:
        logger.info("=== Descriptor: %s ===", desc)
        t0 = time.time()

        embedder = GlobalEmbedder(backend=desc)
        embeddings, _ = embed_partition(metadata_table, embedder, root_dir, desc=desc)

        npy_path = os.path.join(embeddings_dir, f"{partition}_{desc}.npy")
        np.save(npy_path, embeddings)
        logger.info("Saved embeddings to %s  (%.1f s)", npy_path, time.time() - t0)

        # faiss row index is just the position in the matrix
        row_counters[desc] = np.arange(len(metadata_table))

        if partition == "reference":
            faiss_dir = os.path.join(partition_dir, FAISS_SUBDIR)
            os.makedirs(faiss_dir, exist_ok=True)

            index = build_faiss_index(embeddings)
            faiss_path = os.path.join(faiss_dir, f"{desc}.index")
            faiss.write_index(index, faiss_path)
            logger.info("Saved FAISS index to %s", faiss_path)

            # meta list: position i → (individual_id, image_id, crop_path, viewpoint)
            meta_list = []
            for i, (_, row) in enumerate(metadata_table.iterrows()):
                ind_id   = str(row.get(ID_COL, "")) if ID_COL in row.index else ""
                img_id   = image_ids[i]
                crop_p   = crop_paths[i]
                viewpt   = str(row.get(VIEWPOINT_COL, "unknown")) if VIEWPOINT_COL in row.index else "unknown"
                meta_list.append((ind_id, img_id, crop_p, viewpt))

            meta_pkl_path = os.path.join(faiss_dir, f"reference_{desc}_meta.pkl")
            with open(meta_pkl_path, "wb") as fh:
                pickle.dump(meta_list, fh)
            logger.info("Saved FAISS meta to %s", meta_pkl_path)

        del embedder

    # -----------------------------------------------------------------------
    # Write index parquet
    # -----------------------------------------------------------------------
    index_records = []
    for i, (_, row) in enumerate(metadata_table.iterrows()):
        rec = {
            "image_id":             image_ids[i],
            "path_relative_to_root": row["path_relative_to_root"],
            "individual_id":        str(row.get(ID_COL, "")) if ID_COL in row.index else "",
            "viewpoint":            str(row.get(VIEWPOINT_COL, "unknown")) if VIEWPOINT_COL in row.index else "unknown",
            "crop_path":            crop_paths[i],
            "partition":            partition,
        }
        for desc in ACTIVE_DESCRIPTORS:
            rec[f"{desc}_row"] = int(row_counters[desc][i])
        index_records.append(rec)

    index_df = pd.DataFrame(index_records)
    parquet_path = os.path.join(embeddings_dir, f"{partition}_{INDEX_PARQUET_FILENAME}")
    index_df.to_parquet(parquet_path, index=False)
    logger.info("Saved index parquet to %s", parquet_path)

    # Also persist metadata csv (may have been enriched with crop_path column)
    metadata_table.to_csv(metadata_filepath, index=False)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats()

    print_memory_usage()

    restore_stdout(log_file_std_output, log_file_err_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create global deep embeddings for elephant re-ID (step 2)")
    parser.add_argument(
        "--partition",
        type=str,
        required=True,
        choices=["query", "reference"],
        help="Partition to embed: query or reference",
    )
    args = parser.parse_args()
    main(args.partition)
