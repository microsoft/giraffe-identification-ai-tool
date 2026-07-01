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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils_matching import run_union_find, replace_negatives_with_unique_values
from utils.helpers_matching import (
    calculate_partitioning_accuracy,
    save_partitioning_results,
    mint_new_individual_id,
    load_data_dirs,
    load_metadata_file,
    load_embeddings,
    print_memory_usage,
    log_to_file,
    restore_stdout,
)
from configs.config_elephant import (
    ID_COL,
    IMAGE_ID_COL,
    MATCH_ACCEPT_THRESHOLD,
    EMBEDDINGS_SUBDIR,
)
from configs.config_matching import auto_accept_model_matching_results, formatted_string_for_setup


def get_update_status(row, reference_basenames):
    if row["filename"] in reference_basenames:
        return "existing"
    elif row["human_input"] in ["AcceptId", "AssignNewId"]:
        return "processed"
    else:
        return "skipped"


def get_serial_to_aid_dict(df_assign_new_id):
    serial_to_aid_dict = {}
    if ID_COL in df_assign_new_id.columns:
        valid = df_assign_new_id[
            (df_assign_new_id[ID_COL] != -1) & df_assign_new_id[ID_COL].notna()
        ]
        serial_to_aid_dict = valid.set_index(IMAGE_ID_COL)[ID_COL].to_dict()
    print("\n length of serial_to_aid_dict {}".format(len(serial_to_aid_dict)))
    return serial_to_aid_dict


def filter_data_with_human_inputs(metadata):

    query_metadata = metadata["query"]
    reference_metadata = metadata["reference"]

    print("Columns before drop:", query_metadata.columns.tolist())
    query_metadata.drop(columns=["assigned_individual_id"], inplace=True, errors="ignore")
    print("Columns after drop:", query_metadata.columns.tolist())

    query_metadata["filename"] = query_metadata["path_relative_to_root"].apply(os.path.basename)
    reference_metadata["filename"] = reference_metadata["path_relative_to_root"].apply(os.path.basename)
    reference_basenames = set(reference_metadata["filename"])

    query_metadata["database_update_status"] = query_metadata.apply(
        lambda x: get_update_status(x, reference_basenames), axis=1
    )
    query_metadata.drop(columns=["filename"], inplace=True)
    reference_metadata.drop(columns=["filename"], inplace=True)

    return query_metadata


def inference_per_query_for_partitioning(query_row, query_metadata):
    """
    Use the fused similarity score already computed by step_3.

    If match_fused_sim_1 >= MATCH_ACCEPT_THRESHOLD the query image is linked
    to match_image_1; otherwise it is assigned -1 (new individual).
    """
    image_id = query_row[IMAGE_ID_COL]
    path = query_row["path_relative_to_root"]

    sim = query_row.get("match_fused_sim_1", np.nan)
    matched_image = query_row.get("match_image_1", None)

    if pd.notna(sim) and float(sim) >= MATCH_ACCEPT_THRESHOLD and pd.notna(matched_image):
        top_image_id = matched_image
        fused_sim = float(sim)
    else:
        top_image_id = -1
        fused_sim = np.nan

    return (
        [top_image_id],
        [image_id],
        [1 if top_image_id != -1 else np.nan],
        [fused_sim],
        [path],
    )


def run_partitioning_algorithm(df_assign_new_id):

    serial_to_aid_dict = get_serial_to_aid_dict(df_assign_new_id)

    pred_matches_ls = []
    query_serials_ls = []
    serials_count_ls = []
    distances_ls = []
    image_filepath_ls = []

    for _, query_row in df_assign_new_id.iterrows():
        top_images, query_ids, counts, sims, filepaths = inference_per_query_for_partitioning(
            query_row, df_assign_new_id
        )
        pred_matches_ls.extend(top_images)
        query_serials_ls.extend(query_ids)
        serials_count_ls.extend(counts)
        distances_ls.extend(sims)
        image_filepath_ls.extend(filepaths)

    print("\n query_serials_ls", len(query_serials_ls), "\n pred_matches_ls", len(pred_matches_ls), sep="\n")

    actual_ground_truth_ls = [
        serial_to_aid_dict.get(sid, np.nan) for sid in query_serials_ls
    ]

    # Replace -1 sentinels with unique placeholder integers for union-find
    pred_matches_arr = np.array(
        [hash(m) % (2**31) if isinstance(m, str) else int(m) for m in pred_matches_ls],
        dtype=np.int64,
    )
    query_serials_arr = np.array(
        [hash(s) % (2**31) if isinstance(s, str) else int(s) for s in query_serials_ls],
        dtype=np.int64,
    )

    pred_matches_arr_revised = replace_negatives_with_unique_values(pred_matches_arr)
    value_to_new_id = run_union_find(query_serials_arr, pred_matches_arr_revised)

    # Cluster IDs → mint real string individual IDs
    cluster_ids = [value_to_new_id.get(v, -1) for v in pred_matches_arr_revised]
    unique_clusters = sorted(set(cluster_ids))
    cluster_to_individual = {c: mint_new_individual_id() for c in unique_clusters}
    new_ids_ls = [cluster_to_individual[c] for c in cluster_ids]

    results_df = save_partitioning_results(
        image_filepath_ls,
        query_serials_ls,
        pred_matches_ls,
        pred_matches_arr_revised.tolist(),
        new_ids_ls,
        serials_count_ls,
        distances_ls,
        actual_ground_truth_ls,
    )
    return results_df


def record_ids_for_accepted_items_by_expert(query_metadata):
    if query_metadata["match_individual_1"].isna().any():
        print("Warning: There are NaN values in the 'match_individual_1' column!")

    query_metadata.loc[query_metadata["human_input"] == "AcceptId", "assigned_individual_id"] = (
        query_metadata.loc[query_metadata["human_input"] == "AcceptId", "match_individual_1"]
    )
    return query_metadata


def merge_new_ids_results_and_metadata_query(query_metadata, results_df):
    unique_results_df = results_df[["path_relative_to_root", "assigned_individual_id"]].drop_duplicates()

    duplicates = unique_results_df[unique_results_df.duplicated("path_relative_to_root", keep=False)]
    if not duplicates.empty:
        print(
            f"Warning: {duplicates['path_relative_to_root'].nunique()} rows with duplicate "
            f"'path_relative_to_root' values in 'results_df'. These rows will not be processed correctly."
        )
        print(duplicates)
        return query_metadata

    return query_metadata.merge(unique_results_df, on="path_relative_to_root", how="left")


def main():

    profiler = cProfile.Profile()
    profiler.enable()

    root_dir, _ = load_data_dirs()

    log_file_std_output, log_file_err_output = log_to_file(root_dir, "partition_new_items")

    metadata = {}
    for partition in ["query", "reference"]:
        metadata_filepath = os.path.join(root_dir, partition + "_dir", "metadata_" + partition + ".csv")
        metadata[partition] = load_metadata_file(metadata_filepath)

    if auto_accept_model_matching_results:
        print("Warning: Accepting all model results from the initial round of matching for validation purpose only.")
        metadata["query"].loc[metadata["query"]["matching_status"] == "matched", "human_input"] = "AcceptId"
        metadata["query"].loc[metadata["query"]["matching_status"] == "not_matched", "human_input"] = "AssignNewId"

    columns_needed_for_processing = ["path_relative_to_root", "match_individual_1", "human_input"]
    missing_columns = [col for col in columns_needed_for_processing if col not in metadata["query"].columns]

    if missing_columns:
        print(f"Error: The following required columns are missing from query_metadata: {missing_columns}")
    else:
        query_metadata = filter_data_with_human_inputs(metadata)

        df_assign_new_id = query_metadata[query_metadata["human_input"] == "AssignNewId"].reset_index().copy()

        if df_assign_new_id.empty:
            print("The filtered dataFrame based on human input to analyze for partitioning is empty. Exiting...")
        else:
            print("\n df_assign_new_id shape {}".format(df_assign_new_id.shape))

            results_df = run_partitioning_algorithm(df_assign_new_id)

            file_path = os.path.join(root_dir, "query_dir", "query_to_ids_" + formatted_string_for_setup + ".csv")
            print(formatted_string_for_setup)
            print(file_path)
            _, _ = calculate_partitioning_accuracy(file_path)

            query_metadata = merge_new_ids_results_and_metadata_query(query_metadata, results_df)

        query_metadata = record_ids_for_accepted_items_by_expert(query_metadata)
        query_metadata.to_csv(os.path.join(root_dir, "query_dir", "metadata_query.csv"), index=False)

    print_memory_usage()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats()

    restore_stdout(log_file_std_output, log_file_err_output)


if __name__ == "__main__":
    main()
