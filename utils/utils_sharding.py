# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd
from collections import OrderedDict
from configs.config_matching import faiss_mode_cutoff_re_id

def create_freq_table_from_catalog(ref_data):
    """
    Create a frequency table based on the ground truth annotation column, 'AID2021', 
    and '#Serial' for each image.
    
    :param ref_data: catalog reference dataframe
    :return: frequency dictionary showing number of images for each unique ID
    """

    freq_table = ref_data.groupby(['AID2021']).count().reset_index()
    freq_table.rename(columns={'#Serial':'count'}, inplace=True)
    freq_table.sort_values('count', ascending = False, inplace=True)
    # print(freq_table)

    freq_dict = OrderedDict()
    for _, rows in freq_table.iterrows():
        freq_dict[int(rows['AID2021'])] = int(rows['count'])
    # print(freq_dict)
    
    return freq_dict

def split_ids_by_freq(freq_table, min_dataset_size, max_dataset_size, json_output):
    """
    Splits IDs into bins based on frequency table while ensuring each bin has at least
    min_dataset_size and at most max_dataset_size. Any leftover data is redistributed.
    
    :param freq_table: dict where keys are IDs and values are their frequencies
    :param min_dataset_size: Minimum number of IDs per bin
    :param max_dataset_size: Maximum number of IDs per bin
    :return: List of bins (each bin is a list of IDs)
    :return: List of bins sizes
    """
    # Sort IDs by frequency (descending)
    sorted_ids = sorted(freq_table.items(), key=lambda x: x[1], reverse=True)
    
    total_size = sum(freq_table.values())
    print(f'catalog total_size: {total_size}')
    
    num_bins = max(1, round(total_size / min_dataset_size)-1)
    print(f'num_bins (number of shards): {num_bins}')
    
    bins = [[] for _ in range(num_bins)]
    bin_sizes = [0] * num_bins
    
    # Distribute IDs into bins
    for id_, freq in sorted_ids:
        # Find the bin with the least current size
        min_bin_idx = bin_sizes.index(min(bin_sizes))
        
        # Assign ID to that bin
        bins[min_bin_idx].append(id_)
        bin_sizes[min_bin_idx] += freq
    
    print(f'bin_sizes (shards sizes): {bin_sizes}')
    
    # Merge bins if any are below min_dataset_size
    merged_bins = []
    merged_bin_sizes = []
    current_bin = []
    current_size = 0
    
    for b in bins:
        bin_size = sum(freq_table[id_] for id_ in b)
        if current_size + bin_size <= max_dataset_size:
            current_bin.extend(b)
            current_size += bin_size
        else:
            if current_bin:
                merged_bins.append(current_bin)
                merged_bin_sizes.append(current_size)
            current_bin = b
            current_size = bin_size
    
    if current_bin:
        merged_bins.append(current_bin)
        merged_bin_sizes.append(current_size)
    
    # If the last bin is smaller than min_dataset_size, merge it with the smallest existing bin
    if len(merged_bins) > 1 and merged_bin_sizes[-1] < min_dataset_size:
        smallest_bin_idx = min(range(len(merged_bins) - 1), key=lambda i: merged_bin_sizes[i])
        merged_bins[smallest_bin_idx].extend(merged_bins.pop())
        merged_bin_sizes[smallest_bin_idx] += merged_bin_sizes.pop()

    merged_bins_dict = {str(i): item for i, item in enumerate(merged_bins)}
    with open(os.path.join(json_output, 'shards_mapping.json'), 'w') as json_file:
        json.dump(merged_bins_dict, json_file, indent=4)
    
    with open(os.path.join(json_output, "shards_sizes.txt"), "w") as f:
        for item in merged_bin_sizes:
            f.write(str(item) + "\n")
    
    return merged_bins_dict, merged_bin_sizes

def merge_query_shards(folder_path, output_filename):
    """
    Loads all CSV files ending with '_shard.csv' from the folder, selects the top 3 items per row
    across all files based on the highest matching_mode_x value, and saves the combined results.

    :param folder_path: Path to the folder containing the CSV files.
    :param output_filename: Name of the output CSV file.
    """
    shard_files = [f for f in os.listdir(folder_path) if "_shard_" in f and f.endswith(".csv")]
    combined_data = []
    meta_dataframes = []
    
    for file in shard_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        
        match_cols = [
            (f"matched_img_serial_{i}", f"matched_label_{i}", 
             f"matching_mean_dist_{i}", f"matching_mode_{i}")
            for i in range(1, 4)
        ]
        
        all_match_data = df[[col for cols in match_cols for col in cols]].values.reshape(len(df), 3, 4)
        combined_data.append(all_match_data)
        meta_dataframes.append(df)
    
    combined_data = np.concatenate(combined_data, axis=1)
    sorted_indices = combined_data[:, :, 3].argsort(axis=1)[:, ::-1]
    sorted_data = combined_data[np.arange(len(combined_data))[:, None], sorted_indices]
    
    final_df = meta_dataframes[0].copy()
    for i in range(3):
        final_df[f"matched_img_serial_{i+1}"], final_df[f"matched_label_{i+1}"], \
        final_df[f"matching_mean_dist_{i+1}"], final_df[f"matching_mode_{i+1}"] = \
            sorted_data[:, i, 0], sorted_data[:, i, 1], sorted_data[:, i, 2], sorted_data[:, i, 3]
    
    final_df["matching_status"] = np.where(
        (final_df["matching_mode_1"].notna()) & (final_df["matching_mode_1"] > faiss_mode_cutoff_re_id),
        "matched",
        "not_matched"
        )

    final_df.to_csv(os.path.join(folder_path, output_filename), index=False)