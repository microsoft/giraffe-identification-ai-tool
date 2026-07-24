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
from configs.config_artifacts import ARTIFACT_SCHEMA_VERSION
from utils.artifact_schema import (
    DESCRIPTOR_MAPPING_COLUMNS,
    DESCRIPTOR_MAPPING_DTYPES,
    assert_descriptor_mapping_integrity,
)

logger = logging.getLogger(__name__)

_BATCH_SIZE = 32
BODY_VIEWPOINT_PREPROCESS_FINGERPRINT = "viewpoint-right-to-left-v1"


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


def _empty_descriptor_mapping() -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: pd.Series(dtype=dtype)
            for column, dtype in DESCRIPTOR_MAPPING_DTYPES.items()
        }
    )


def embed_from_crop_manifest(
    crop_manifest: pd.DataFrame,
    embedder,
    descriptor_name: str,
    *,
    is_ear: bool = False,
    apply_clahe_to_ear: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Embed readable accepted crops without creating placeholder vectors."""
    required = {
        "crop_id",
        "image_id",
        "crop_kind",
        "crop_ordinal",
        "crop_path",
        "detector_status",
        "schema_version",
        "source_fingerprint",
        "split_fingerprint",
    }
    missing = sorted(required - set(crop_manifest.columns))
    if missing:
        raise ValueError(f"crop manifest is missing columns required for embedding: {missing}")

    crop_kind = "ear" if is_ear else "body"
    selected = crop_manifest[
        (crop_manifest["detector_status"] == "accepted")
        & (crop_manifest["crop_kind"] == crop_kind)
    ].copy()
    selected = selected.sort_values(
        ["image_id", "crop_ordinal", "crop_id"], kind="stable"
    )

    # -----------------------------------------------------------------------
    # Blocker 2: require non-empty individual_id for every accepted crop that
    # enters an embedding matrix.  An empty string here would silently produce
    # a descriptor mapping that cannot be used for identity-aware evaluation.
    # -----------------------------------------------------------------------
    if "individual_id" not in selected.columns:
        raise ValueError(
            "crop manifest is missing 'individual_id'; populate it from the "
            "canonical image manifest before embedding"
        )
    empty_individual = (
        selected["individual_id"].isna()
        | selected["individual_id"].astype(str).str.strip().eq("")
    )
    if empty_individual.any():
        bad_crops = selected.loc[empty_individual, "crop_id"].tolist()
        raise ValueError(
            f"accepted crops have empty individual_id; cannot embed: {bad_crops}"
        )

    images: list[np.ndarray] = []
    rows: list[pd.Series] = []
    for _, crop_row in selected.iterrows():
        image = cv2.imread(str(crop_row["crop_path"]))
        if image is None:
            raise OSError(
                f"accepted crop is unreadable: {crop_row['crop_path']}"
            )
        if is_ear and apply_clahe_to_ear:
            image = _apply_clahe(image)
        elif not is_ear and "viewpoint" in selected.columns:
            image = _normalize_viewpoint(
                image,
                str(crop_row.get("viewpoint", "")),
            )
        images.append(image)
        rows.append(crop_row)

    if not images:
        return _empty_descriptor_mapping(), np.empty(
            (0, int(embedder.dim)), dtype=np.float32
        )

    matrix = np.asarray(
        embedder.embed_batch(images, batch_size=_BATCH_SIZE), dtype=np.float32
    )
    if matrix.ndim != 2 or matrix.shape != (len(images), int(embedder.dim)):
        raise AssertionError(
            f"embedder returned shape {matrix.shape}; expected {(len(images), int(embedder.dim))}"
        )
    if not np.isfinite(matrix).all():
        raise AssertionError("embedder returned non-finite vectors")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        bad_rows = np.where(norms[:, 0] == 0)[0].tolist()
        raise AssertionError(f"embedder returned zero-norm vectors at rows {bad_rows}")
    matrix = matrix / norms

    mapping_records = []
    for embedding_row, crop_row in enumerate(rows):
        mapping_records.append(
            {
                "descriptor_name": descriptor_name,
                "embedding_row": embedding_row,
                "faiss_row": pd.NA,
                "crop_id": str(crop_row["crop_id"]),
                "image_id": str(crop_row["image_id"]),
                "individual_id": str(crop_row.get("individual_id", "")),
                "crop_kind": str(crop_row["crop_kind"]),
                "crop_ordinal": int(crop_row["crop_ordinal"]),
                "crop_path": os.path.abspath(str(crop_row["crop_path"])),
                "schema_version": str(crop_row["schema_version"]),
                "source_fingerprint": crop_row.get("source_fingerprint"),
                "split_fingerprint": crop_row.get("split_fingerprint"),
                "model_preprocess_fingerprint": None,
            }
        )
    mapping_df = pd.DataFrame(mapping_records, columns=DESCRIPTOR_MAPPING_COLUMNS)
    for column, dtype in DESCRIPTOR_MAPPING_DTYPES.items():
        mapping_df[column] = mapping_df[column].astype(dtype)
    return mapping_df, matrix.astype(np.float32, copy=False)


def build_bteh_descriptor_artifacts(
    crop_manifest: pd.DataFrame,
    embedder_factory,
    descriptor_name: str,
    artifact_dir: str,
    *,
    is_reference: bool,
    is_ear: bool = False,
    schema_version: str = ARTIFACT_SCHEMA_VERSION,
    source_fingerprint: str | None = None,
    split_fingerprint: str | None = None,
    model_fingerprint: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build and validate one descriptor's normalized elephant artifacts."""
    def _resolve_provenance(column: str, supplied: str | None) -> str | None:
        values = {
            str(value)
            for value in crop_manifest[column].dropna().unique()
            if str(value)
        }
        if len(values) > 1:
            raise ValueError(
                f"crop manifest has multiple {column} values: {sorted(values)}"
            )
        inherited = next(iter(values), None)
        if supplied is not None and inherited != supplied:
            raise ValueError(
                f"{column} mismatch: crop manifest has {inherited!r}, "
                f"caller supplied {supplied!r}"
            )
        return inherited

    resolved_source_fingerprint = _resolve_provenance(
        "source_fingerprint",
        source_fingerprint,
    )
    resolved_split_fingerprint = _resolve_provenance(
        "split_fingerprint",
        split_fingerprint,
    )
    embedder = embedder_factory(descriptor_name)
    mapping_df, matrix = embed_from_crop_manifest(
        crop_manifest,
        embedder,
        descriptor_name,
        is_ear=is_ear,
    )
    mapping_df["schema_version"] = schema_version
    mapping_df["source_fingerprint"] = resolved_source_fingerprint
    mapping_df["split_fingerprint"] = resolved_split_fingerprint
    effective_model_fingerprint = model_fingerprint
    if not is_ear and "viewpoint" in crop_manifest.columns:
        effective_model_fingerprint = (
            f"{model_fingerprint}+{BODY_VIEWPOINT_PREPROCESS_FINGERPRINT}"
            if model_fingerprint
            else BODY_VIEWPOINT_PREPROCESS_FINGERPRINT
        )
    mapping_df["model_preprocess_fingerprint"] = effective_model_fingerprint
    if is_reference:
        mapping_df["faiss_row"] = pd.Series(
            range(len(mapping_df)), dtype=DESCRIPTOR_MAPPING_DTYPES["faiss_row"]
        )
        index = build_faiss_index(matrix)
    else:
        mapping_df["faiss_row"] = pd.Series(
            [pd.NA] * len(mapping_df), dtype=DESCRIPTOR_MAPPING_DTYPES["faiss_row"]
        )
        index = None

    for column, dtype in DESCRIPTOR_MAPPING_DTYPES.items():
        mapping_df[column] = mapping_df[column].astype(dtype)

    assert_descriptor_mapping_integrity(
        mapping_df,
        matrix,
        index,
        is_reference=is_reference,
        schema_version=schema_version,
        expected_source_fingerprint=resolved_source_fingerprint,
        expected_split_fingerprint=resolved_split_fingerprint,
        expected_model_fingerprint=effective_model_fingerprint,
    )

    os.makedirs(artifact_dir, exist_ok=True)
    mapping_df.to_parquet(
        os.path.join(artifact_dir, f"{descriptor_name}_mapping.parquet"),
        index=False,
    )
    np.save(os.path.join(artifact_dir, f"{descriptor_name}.npy"), matrix)
    if is_reference:
        faiss.write_index(index, os.path.join(artifact_dir, f"{descriptor_name}.index"))
    return mapping_df, matrix


build_normalized_descriptor_artifacts = build_bteh_descriptor_artifacts


def _normalized_main(*, use_bteh_defaults: bool = False):
    """Normalized embedding route for canonical elephant crop manifests."""
    default_artifact_dir = None
    if use_bteh_defaults:
        from configs.config_bteh import (
            ARTIFACT_VERSION_ROOT,
            EMBEDDINGS_SUBDIR_BTEH,
        )
        default_artifact_dir = str(
            ARTIFACT_VERSION_ROOT / EMBEDDINGS_SUBDIR_BTEH
        )
    from configs.config_elephant import EAR_DESCRIPTORS

    parser = argparse.ArgumentParser(
        description=(
            "Normalized elephant embedding: build descriptor mapping, "
            "embedding matrix, and reference FAISS index from a crop manifest."
        )
    )
    parser.add_argument(
        "--crop-manifest",
        required=True,
        help="Path to the normalized crop manifest parquet.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=default_artifact_dir,
        required=not use_bteh_defaults,
        help="Directory to write mapping/matrix/index artifacts.",
    )
    parser.add_argument(
        "--partition",
        choices=["query", "reference"],
        required=True,
        help="'reference' builds a FAISS index; 'query' skips it.",
    )
    parser.add_argument(
        "--descriptors",
        nargs="+",
        default=list(ACTIVE_DESCRIPTORS),
        help="Descriptor names to build (default: all ACTIVE_DESCRIPTORS).",
    )
    parser.add_argument("--schema-version", default=ARTIFACT_SCHEMA_VERSION)
    parser.add_argument("--source-fingerprint", required=True)
    parser.add_argument("--split-fingerprint", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument(
        "--disable-cudnn",
        action="store_true",
        help="Disable cuDNN and use generic CUDA kernels for incompatible hosts.",
    )
    args = parser.parse_args()

    if args.disable_cudnn:
        import torch
        torch.backends.cudnn.enabled = False

    crop_manifest = pd.read_parquet(args.crop_manifest)
    is_reference = args.partition == "reference"

    for desc in args.descriptors:
        is_ear = desc in EAR_DESCRIPTORS

        def _factory(d=desc):
            return GlobalEmbedder(backend=d)

        logger.info(
            "=== normalized descriptor: %s (ear=%s) ===",
            desc,
            is_ear,
        )
        build_bteh_descriptor_artifacts(
            crop_manifest=crop_manifest,
            embedder_factory=_factory,
            descriptor_name=desc,
            artifact_dir=args.artifact_dir,
            is_reference=is_reference,
            is_ear=is_ear,
            schema_version=args.schema_version,
            source_fingerprint=args.source_fingerprint,
            split_fingerprint=args.split_fingerprint,
            model_fingerprint=args.model_fingerprint,
        )
        logger.info("Artifacts written to %s", args.artifact_dir)


def _bteh_main():
    _normalized_main(use_bteh_defaults=True)


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
    normalized_flags = {
        flag
        for flag in ("--bteh", "--normalized")
        if flag in sys.argv[1:]
    }
    if len(normalized_flags) > 1:
        raise SystemExit("--bteh and --normalized are mutually exclusive")
    if normalized_flags:
        flag = normalized_flags.pop()
        sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[1:] if arg != flag]
        if flag == "--normalized":
            _normalized_main()
        else:
            _bteh_main()
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Create global deep embeddings for elephant re-ID (step 2)")
    parser.add_argument(
        "--partition",
        type=str,
        default=None,
        choices=["query", "reference"],
        help="(Legacy giraffe mode) Partition to embed: query or reference",
    )
    mode_args = parser.parse_args()
    if mode_args.partition is None:
        parser.error(
            "--partition is required for legacy giraffe mode "
            "(or pass --normalized/--bteh)"
        )
    main(mode_args.partition)
