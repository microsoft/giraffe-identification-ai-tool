# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


import os
import sys
import csv
import time
import json
import uuid
import psutil
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from sklearn.metrics.cluster import adjusted_rand_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_matching import formatted_string_for_setup
from configs.config_elephant import (
    NEW_ID_PREFIX,
    EMBEDDINGS_SUBDIR,
    ACTIVE_DESCRIPTORS,
    ID_COL,
    IMAGE_ID_COL,
)

load_dotenv()
container_name = os.getenv("container_name")
data_root_abs_path = os.getenv("data_root_abs_path")


# ---------------------------------------------------------------------------
# ID minting
# ---------------------------------------------------------------------------

def mint_new_individual_id() -> str:
    """Return a fresh, unique string ID for an unknown individual."""
    return NEW_ID_PREFIX + uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def get_img_paths_from_a_folder(root_dir, relative_to_root_image_dir):
    keyname_col = "path_relative_to_root"
    file_path_dict = {keyname_col: []}
    file_path_dict[keyname_col] = [
        os.path.join(relative_to_root_image_dir, img)
        for img in os.listdir(os.path.join(root_dir, relative_to_root_image_dir))
        if img.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    ]

    csv_dir = os.path.join(root_dir, "sample_query_metadata_files")
    saved_file_path = os.path.join(
        csv_dir,
        "metadata_query__" + relative_to_root_image_dir.replace("/", "__") + ".csv",
    )
    pd.DataFrame(file_path_dict).to_csv(saved_file_path, index=False)
    return saved_file_path


def get_img_paths_from_several_folders(root_dir, image_subdir, textfile_path):
    with open(textfile_path, "r") as file:
        folder_paths = [line.strip() for line in file.readlines()]

    keyname_col = "path_relative_to_root"
    file_path_dict = {keyname_col: []}

    for a_path in folder_paths:
        image_dir = os.path.join(root_dir, image_subdir, a_path)
        if not os.path.exists(image_dir):
            print(f"Directory does not exist: {image_dir}")
            continue
        image_files = [
            os.path.join(image_subdir, a_path, img)
            for img in os.listdir(image_dir)
            if img.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        ]
        file_path_dict[keyname_col].extend(image_files)

    csv_dir = os.path.join(root_dir, "sample_query_metadata_files/query_based_on_backlog_May2020_Feb2023/")
    os.makedirs(csv_dir, exist_ok=True)
    csv_file_path = os.path.join(csv_dir, "metadata_query_backlog.csv")
    pd.DataFrame(file_path_dict).to_csv(csv_file_path, index=False)
    print(f"CSV file saved at: {csv_file_path}")


def load_metadata_file(metadata_filepath):
    keyname_col = "path_relative_to_root"

    if not os.path.exists(metadata_filepath):
        print("Error: File '{}' does not exist.".format(metadata_filepath))
        sys.exit(1)

    try:
        df = pd.read_csv(metadata_filepath)
    except Exception as e:
        print("Failed to read '{}': {}".format(metadata_filepath, e))
        sys.exit(1)

    if keyname_col not in df.columns:
        print("Error: " + keyname_col + " column not found in '{}'.".format(metadata_filepath))
        sys.exit(1)

    filenames = df[keyname_col].apply(os.path.basename)
    duplicate_filenames = filenames[filenames.duplicated()]

    if not duplicate_filenames.empty:
        print(
            "Error: Duplicate image filenames found in "
            + keyname_col
            + " column in '{}':".format(metadata_filepath)
        )
        print(duplicate_filenames.to_string(index=False))
        sys.exit(1)

    print("File '{}' exists and contains ".format(metadata_filepath) + keyname_col + " column.")
    return df


def load_embeddings(root_dir, partitions=("query", "reference"), descriptors=None):
    """Load embedding matrices and index parquets for given partitions.

    Returns:
        embeddings_dict : {partition: {desc_name: np.ndarray(n, D)}}
        index_df_dict   : {partition: pd.DataFrame}
    """
    if descriptors is None:
        descriptors = ACTIVE_DESCRIPTORS

    embeddings_dict = {}
    index_df_dict = {}

    for partition in partitions:
        part_dir = os.path.join(root_dir, partition + "_dir")
        emb_dir = os.path.join(part_dir, EMBEDDINGS_SUBDIR)

        embeddings_dict[partition] = {}
        for desc in descriptors:
            npy_path = os.path.join(emb_dir, f"{partition}_{desc}.npy")
            if not os.path.isfile(npy_path):
                print(f"Warning: embedding file not found: {npy_path}")
                embeddings_dict[partition][desc] = np.empty((0,), dtype=np.float32)
                continue
            start = time.time()
            embeddings_dict[partition][desc] = np.load(npy_path).astype(np.float32)
            print(
                f"Loaded {partition}/{desc} embeddings {embeddings_dict[partition][desc].shape} "
                f"in {time.time()-start:.3f}s"
            )

        parquet_path = os.path.join(emb_dir, "index.parquet")
        if os.path.isfile(parquet_path):
            index_df_dict[partition] = pd.read_parquet(parquet_path)
        else:
            print(f"Warning: index parquet not found: {parquet_path}")
            index_df_dict[partition] = pd.DataFrame()

    return embeddings_dict, index_df_dict


def load_data_dirs():
    root_dir = os.path.join(data_root_abs_path, container_name)
    processed_img_dir = os.path.join(root_dir, "processed_images")
    os.makedirs(processed_img_dir, exist_ok=True)
    return root_dir, processed_img_dir


# ---------------------------------------------------------------------------
# Logging / profiling
# ---------------------------------------------------------------------------

def log_to_file(root_dir, keyword, subdir="query_dir"):
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_dir = os.path.join(root_dir, subdir, "logs")
    os.makedirs(log_file_dir, exist_ok=True)

    log_filename_std_output = os.path.join(
        log_file_dir, "log__std_output__" + keyword + "__" + current_datetime + ".log"
    )
    log_filename_err_output = os.path.join(
        log_file_dir, "log__err_output__" + keyword + "__" + current_datetime + ".log"
    )

    log_file_std_output = open(log_filename_std_output, "w")
    log_file_err_output = open(log_filename_err_output, "w")

    sys.stdout = log_file_std_output
    sys.stderr = log_file_err_output

    return log_file_std_output, log_file_err_output


def restore_stdout(log_file_std_output, log_file_err_output):
    log_file_std_output.close()
    log_file_err_output.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


def print_memory_usage():
    print("\nMemory Usage Details:")
    process = psutil.Process(os.getpid())
    print("\nMemory Usage Summary: {:.2f} MB".format(process.memory_info().rss / 1024**2))


# ---------------------------------------------------------------------------
# Accuracy / results helpers
# ---------------------------------------------------------------------------

def save_accuracy_to_csv_with_timestamp(accuracy, output_dir, run_name="run"):
    csv_file_name = f"{run_name}_accuracy.csv"
    csv_file_path = os.path.join(output_dir, csv_file_name)

    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "accuracy"])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, accuracy])

    print(f"Accuracy saved to {csv_file_path}")


def save_all_accuracy_to_csv(accuracy, accuracy_matched, accuracy_not_matched, save_dir, run_name="run"):
    data = {
        "Metric": ["accuracy", "accuracy_matched", "accuracy_not_matched"],
        "Value": [accuracy, accuracy_matched, accuracy_not_matched],
    }
    df = pd.DataFrame(data)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{run_name}_accuracy.csv")
    df.to_csv(save_path, index=False)
    print(f"Accuracy data saved to {save_path}")


def save_partitioning_results(
    image_filepath_ls,
    query_serials_ls,
    pred_matches_ls,
    pred_matches_ls_revised,
    new_ids_ls,
    serials_count_ls,
    distances_ls,
    actual_ground_truth_ls,
):
    df = pd.DataFrame(
        {
            "path_relative_to_root": image_filepath_ls,
            IMAGE_ID_COL: query_serials_ls,
            "actual_ground_truth": actual_ground_truth_ls,
            "pred_match": pred_matches_ls,
            "pred_match_revised": pred_matches_ls_revised,
            "assigned_individual_id": new_ids_ls,
            "serial_count": serials_count_ls,
            "distance": distances_ls,
        }
    )

    query_to_ids = {}
    for image_id, new_id in zip(query_serials_ls, new_ids_ls):
        key = str(image_id)
        if key not in query_to_ids:
            query_to_ids[key] = []
        query_to_ids[key].append(str(new_id))

    root_dir, _ = load_data_dirs()
    output_dir = os.path.join(root_dir, "query_dir", "query_to_ids_" + formatted_string_for_setup + ".json")
    with open(output_dir, "w") as json_file:
        json.dump(query_to_ids, json_file, indent=4)
    df.to_csv(
        os.path.join(root_dir, "query_dir", "query_to_ids_" + formatted_string_for_setup + ".csv"),
        index=False,
    )

    print("DataFrame:")
    print(df.head(5))
    print("\nMapping Dictionary:")
    print(query_to_ids)

    return df


def calculate_partitioning_accuracy(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping accuracy calculation.")
        return None, None

    df = pd.read_csv(file_path)
    required_cols = {IMAGE_ID_COL, "assigned_individual_id", "actual_ground_truth"}

    if not required_cols.issubset(df.columns):
        print(
            f"Missing required columns {required_cols - set(df.columns)} in {file_path}. "
            "Skipping accuracy calculation."
        )
        return None, None

    duplicate_counts = df.groupby(IMAGE_ID_COL)["assigned_individual_id"].nunique()
    multiple_ids = duplicate_counts[duplicate_counts > 1]
    if not multiple_ids.empty:
        print(
            f"Warning: {len(multiple_ids)} paths have multiple `assigned_individual_id` values. "
            "This may affect accuracy calculations."
        )
        print(multiple_ids)

    df = df.drop_duplicates(subset=[IMAGE_ID_COL, "assigned_individual_id"])
    valid_rows = df.dropna(subset=["assigned_individual_id", "actual_ground_truth"])

    assigned_ids = valid_rows["assigned_individual_id"].values
    ground_truth_ids = valid_rows["actual_ground_truth"].values

    if len(assigned_ids) > 0 and len(ground_truth_ids) > 0:
        ari_score = adjusted_rand_score(assigned_ids, ground_truth_ids)
        print("Adjusted Rand Index:", ari_score)
    else:
        print("No valid data to compute Adjusted Rand Index.")

    return assigned_ids, ground_truth_ids


def save_merged_accuracy_results(accuracy_results, root_dir, reset=False):
    csv_file = os.path.join(root_dir, "query_dir", "accuracy_results.csv")

    if os.path.exists(csv_file) and not reset:
        existing_df = pd.read_csv(csv_file)
        value_columns_existing = [col for col in existing_df.columns if col.startswith("Value")]
        new_value_column_name = f"Value_{len(value_columns_existing) + 1}"
        accuracy_results.rename(columns={"Value": new_value_column_name}, inplace=True)
        merged_df = pd.merge(existing_df, accuracy_results, on="Metric", how="outer")
        merged_df.to_csv(csv_file, index=False)
        print(f"Merged and saved to CSV with new column {new_value_column_name}.")
    else:
        accuracy_results.to_csv(csv_file, index=False)
        print("Saved new accuracy results to CSV.")


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def make_loio_splits(metadata_df: pd.DataFrame, id_col: str, session_col: str = None):
    """Leave-one-individual-out cross-validation splits.

    Yields (train_df, probe_df) for each unique individual. When session_col
    is provided, any training image that shares a session with a probe image
    is removed to prevent same-session leakage.
    """
    for held_out_id in metadata_df[id_col].dropna().unique():
        probe = metadata_df[metadata_df[id_col] == held_out_id]
        train = metadata_df[metadata_df[id_col] != held_out_id]
        if session_col and session_col in metadata_df.columns:
            probe_sessions = set(probe[session_col].dropna())
            train = train[~train[session_col].isin(probe_sessions)]
        yield train, probe
