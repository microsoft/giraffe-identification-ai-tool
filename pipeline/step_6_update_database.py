# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
import pstats
import cProfile
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers_matching import load_data_dirs, load_metadata_file, load_embeddings
from utils.helpers_matching import print_memory_usage, log_to_file, restore_stdout
from utils.utils_embeddings import (
    load_embeddings_matrix,
    save_embeddings,
    load_index_parquet,
    save_index_parquet,
)
from utils.utils_matching import build_and_save_global_index
from configs.config_elephant import (
    ID_COL,
    IMAGE_ID_COL,
    ACTIVE_DESCRIPTORS,
    EMBEDDINGS_SUBDIR,
    FAISS_SUBDIR,
)


def update_original_ref_db(query_matching_results_df, root_dir, metadata_ref_original):

    print("\n============ Database Update Summary ============")

    ref_dir = os.path.join(root_dir, "reference_dir")
    query_dir = os.path.join(root_dir, "query_dir")

    # Load reference index parquet
    ref_index_df = load_index_parquet(root_dir, "reference")
    new_ref_rows = []

    for desc_name in ACTIVE_DESCRIPTORS:

        query_emb_path = os.path.join(query_dir, EMBEDDINGS_SUBDIR, f"query_{desc_name}.npy")
        ref_emb_path = os.path.join(ref_dir, EMBEDDINGS_SUBDIR, f"reference_{desc_name}.npy")

        if not os.path.isfile(query_emb_path):
            print(f"Warning: query embedding file not found for '{desc_name}': {query_emb_path}")
            continue

        query_matrix = np.load(query_emb_path).astype(np.float32)

        if os.path.isfile(ref_emb_path):
            ref_matrix = np.load(ref_emb_path).astype(np.float32)
        else:
            print(f"Reference embedding file not found for '{desc_name}'; starting fresh.")
            ref_matrix = np.empty((0, query_matrix.shape[1]), dtype=np.float32)

        row_col = f"{desc_name}_row"
        rows_to_append = []

        for _, qrow in tqdm(query_matching_results_df.iterrows(), desc=f"Appending {desc_name}"):
            if row_col not in qrow or pd.isna(qrow[row_col]):
                continue
            emb_idx = int(qrow[row_col])
            if emb_idx < 0 or emb_idx >= len(query_matrix):
                print(f"Warning: embedding row {emb_idx} out of range for query matrix ({len(query_matrix)} rows).")
                continue
            rows_to_append.append(emb_idx)

        if not rows_to_append:
            print(f"No valid embedding rows to append for '{desc_name}'.")
            continue

        new_vecs = query_matrix[rows_to_append]
        updated_ref_matrix = np.vstack([ref_matrix, new_vecs]) if ref_matrix.shape[0] > 0 else new_vecs

        # Save updated reference embeddings
        os.makedirs(os.path.dirname(ref_emb_path), exist_ok=True)
        np.save(ref_emb_path, updated_ref_matrix)
        print(f"[{desc_name}] Reference matrix updated: {ref_matrix.shape[0]} → {updated_ref_matrix.shape[0]} rows.")

        # Rebuild FAISS index for this descriptor
        faiss_index_dir = os.path.join(ref_dir, FAISS_SUBDIR)
        build_and_save_global_index(updated_ref_matrix, desc_name, faiss_index_dir)
        print(f"[{desc_name}] FAISS index rebuilt with {updated_ref_matrix.shape[0]} vectors.")

        # Collect new index-parquet rows only on first descriptor pass to avoid duplicates
        if desc_name == ACTIVE_DESCRIPTORS[0]:
            for list_pos, qrow_idx in enumerate(rows_to_append):
                src_row = query_matching_results_df.iloc[
                    query_matching_results_df[row_col].tolist().index(qrow_idx)
                    if row_col in query_matching_results_df.columns else 0
                ]
                new_ref_rows.append(
                    {
                        IMAGE_ID_COL: src_row.get(IMAGE_ID_COL, ""),
                        "path_relative_to_root": src_row.get("path_relative_to_root", ""),
                        ID_COL: src_row.get("assigned_individual_id", ""),
                        "viewpoint": src_row.get("viewpoint", "unknown"),
                        "crop_path": src_row.get("crop_path", ""),
                        "partition": "reference",
                        f"{ACTIVE_DESCRIPTORS[0]}_row": ref_matrix.shape[0] + list_pos,
                    }
                )

    # Update reference index parquet
    if new_ref_rows:
        new_rows_df = pd.DataFrame(new_ref_rows)
        updated_ref_index = pd.concat([ref_index_df, new_rows_df], ignore_index=True)
        save_index_parquet(updated_ref_index, root_dir, "reference")
        print(f"Reference index parquet updated: {len(ref_index_df)} → {len(updated_ref_index)} rows.")

    # Update metadata
    print("\nMetadata size BEFORE update:", metadata_ref_original.shape)
    append_cols = ["path_relative_to_root", "assigned_individual_id", IMAGE_ID_COL]
    append_cols_present = [c for c in append_cols if c in query_matching_results_df.columns]
    df_to_append = query_matching_results_df[append_cols_present].copy()
    df_to_append = df_to_append.rename(columns={"assigned_individual_id": ID_COL})
    metadata_ref_updated = pd.concat([metadata_ref_original, df_to_append], axis=0).reset_index(drop=True)
    print("Metadata size AFTER update:", metadata_ref_updated.shape)

    print("\n=================================================")
    return metadata_ref_updated


def save_updated_db(root_dir, metadata_ref_updated):
    out_path = os.path.join(root_dir, "reference_dir", "metadata_reference_updated.csv")
    metadata_ref_updated.to_csv(out_path, index=False)
    print("\nSaved updated reference metadata file to:", out_path)


def main():

    profiler = cProfile.Profile()
    profiler.enable()

    root_dir, _ = load_data_dirs()

    log_file_std_output, log_file_err_output = log_to_file(root_dir, "update_ref_database")

    metadata = {}
    for partition in ["query", "reference"]:
        metadata_filepath = os.path.join(root_dir, partition + "_dir", "metadata_" + partition + ".csv")
        metadata[partition] = load_metadata_file(metadata_filepath)
    query_metadata = metadata["query"]

    required_non_nan_columns = ["path_relative_to_root", "assigned_individual_id"]
    columns_needed = ["path_relative_to_root", "assigned_individual_id", "database_update_status"]

    missing_columns = [col for col in columns_needed if col not in query_metadata.columns]
    if missing_columns:
        print(f"Error: The following required columns are missing from query_metadata: {missing_columns}")
    else:
        query_matching_results_df = query_metadata[query_metadata["database_update_status"] == "processed"].copy()
        print(f"Filter processed dataframe shape based on query_metadata: {query_matching_results_df.shape}")

        query_matching_results_df = query_matching_results_df.dropna(subset=required_non_nan_columns)

        if query_matching_results_df.empty:
            print("Error: No rows remain after filtering. Exiting.")
        else:
            metadata_ref_updated = update_original_ref_db(
                query_matching_results_df, root_dir, metadata["reference"]
            )
            save_updated_db(root_dir, metadata_ref_updated)

    query_metadata.loc[query_matching_results_df.index, "final_update_status"] = "completed"
    query_metadata.to_csv(os.path.join(root_dir, "query_dir", "metadata_query.csv"), index=False)

    print_memory_usage()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats()

    restore_stdout(log_file_std_output, log_file_err_output)


if __name__ == "__main__":
    main()
