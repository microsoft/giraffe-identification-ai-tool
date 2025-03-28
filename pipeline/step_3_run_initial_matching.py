# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
import shutil
import pstats
import cProfile
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils_matching import normalize, train_faiss, reshape_reference_data_for_faiss
from utils.helpers_matching import load_data_dirs, load_metadata_file, load_pkl_files
from utils.helpers_matching import print_memory_usage, log_to_file, restore_stdout
from configs.config_matching import num_recommended_ids, faiss_distance_cutoff_re_id, faiss_mode_cutoff_re_id
from configs.config_matching import min_shard_size, max_shard_size
from utils.utils_sharding import create_freq_table_from_catalog, split_ids_by_freq
from utils.utils_sharding import merge_query_shards

def build_shards_from_catalog(ref_data, sharding_results_dir):
    ref_freq_dict = create_freq_table_from_catalog(ref_data)
    merged_bins_dict, _ = split_ids_by_freq(ref_freq_dict, min_shard_size, max_shard_size, sharding_results_dir)
    return merged_bins_dict

def inference_per_query_for_re_identification(query_descriptors_sift, query_metadata, query_image_path, faiss_index,  all_labels_train, all_serials_train):
    
    # Initialize a dict
    faiss_result_dict = {
        'path_relative_to_root': [query_image_path],
        'matching_status': ['not_matched']
    }
    
    for i in range(1, num_recommended_ids + 1):
        faiss_result_dict[f'matched_img_serial_{i}'] = [np.nan]
        faiss_result_dict[f'matched_label_{i}'] = [np.nan]
        faiss_result_dict[f'matching_mean_dist_{i}'] = [np.nan]
        faiss_result_dict[f'matching_mode_{i}'] = [np.nan]
        
        
    # Do FAISS search
    k = 1  # Number of neighbors to search for
    query_descriptors_sift_normalized = normalize(query_descriptors_sift)
    distances, indices = faiss_index.search(query_descriptors_sift_normalized, k=k) # distances and indices will have the shape (n_queries, k)
    
    # print(distances.shape)
    # print(indices.shape)
    
    # Convert to numpy arrays for filtering
    distances = np.array(distances.flatten())
    indices = np.array(indices.flatten())

    # pred_labels is a 1D list with predicted labels
    pred_labels = all_labels_train[indices]

    # Sort serials is a 1D list with predicted serials
    pred_serials = all_serials_train[indices]

    # Process each list of distances and labels: corresponds to a query results
    serial_to_label_dict = dict(zip(pred_serials, pred_labels))
    
    # Apply the distance threshold to filter distances and labels
    mask = distances < faiss_distance_cutoff_re_id
    filtered_distances = distances[mask]
    filtered_serials = pred_serials[mask]
    
    # Initialize a dictionary to store aggregated scores and counts for each label
    serial_scores_mapping = {}

    # Aggregate scores and counts for each predicted label
    for serial, score in zip(filtered_serials, filtered_distances):
        if serial in serial_scores_mapping:
            serial_scores_mapping[serial][0] = serial_to_label_dict[serial] # collect the ground truth label for specific image/query/serial
            serial_scores_mapping[serial][1] += 1     # Increment the count
            serial_scores_mapping[serial][2] += score  # Increment the score
            serial_scores_mapping[serial][3] = serial_scores_mapping[serial][2]/serial_scores_mapping[serial][1]  # calc the avg score
        else:
            serial_scores_mapping[serial] = [serial_to_label_dict[serial], 1, score, score]  # Initialize with score and count = 1

    # Sort serial by aggregated scores in descending order
    sorted_recommendations = sorted(serial_scores_mapping.items(), key=lambda x: x[1][1], reverse=True)
    
   # Iterate and populate the dictionary
    for idx in range(num_recommended_ids):
        if idx < len(sorted_recommendations):
            serial, value = sorted_recommendations[idx]
            faiss_result_dict[f'matched_label_{idx+1}'] = value[0]
            faiss_result_dict[f'matched_img_serial_{idx+1}'] = serial
            faiss_result_dict[f'matching_mode_{idx+1}'] = value[1]
            faiss_result_dict[f'matching_mean_dist_{idx+1}'] = value[3]
        
    # Accept or reject a match
    most_similar_value = faiss_result_dict['matching_mode_1']
    if not np.isnan(most_similar_value) and most_similar_value > faiss_mode_cutoff_re_id:
        faiss_result_dict['matching_status'] = ['matched']
    
    # Call a function to process the filled faiss_result_dict (if applicable)
    query_metadata = fill_matching_results(query_metadata, faiss_result_dict)
    
    return query_metadata

def fill_matching_results(query_metadata, faiss_result_dict):
    
    query_path = faiss_result_dict['path_relative_to_root'][0]
    matching_index = query_metadata[query_metadata['path_relative_to_root'] == query_path].index
    
    # If there is a matching row, fill in the values directly
    if not matching_index.empty:
        for column, value in faiss_result_dict.items():
            if column != 'path_relative_to_root':  # Skip 'path_relative_to_root' itself
                query_metadata.loc[matching_index, column] = value
    
    return query_metadata

def add_columns_for_matching_results(query_metadata):
    
    cols = ['matching_attempt', 'matching_status']
    
    for i in range(1, num_recommended_ids + 1):
        cols += [
            f'matched_img_serial_{i}',
            f'matched_label_{i}',
            f'matching_mean_dist_{i}',
            f'matching_mode_{i}'
            ]
        
    # Add each column if it doesn't already exist
    for col in cols:
        if col not in query_metadata.columns:
            query_metadata[col] = np.nan
    
    return query_metadata

def sweep_over_query_images_for_inference(metadata_filepath, query_descriptor_dict, query_metadata, faiss_index, all_labels_train, all_serials_train):
    query_metadata = add_columns_for_matching_results(query_metadata)
    
    for idx, row in tqdm(query_metadata.iterrows(), total=query_metadata.shape[0]):
        
        if idx%100 == 0:
            query_metadata.to_csv(metadata_filepath, index=False)
    
        query_image_path = row['path_relative_to_root']
        image_filename = os.path.basename(query_image_path)
        query_metadata.loc[idx,'matching_attempt'] = 'failed'
        
        if 'matching_status' in row and row['matching_status'] in {'not_matched', 'matched'}:
            query_metadata.loc[idx,'matching_attempt'] = 'existing'
            print("Query has been processed before.")
        else:
            if image_filename in query_descriptor_dict:
                query_descriptors_sift = query_descriptor_dict[image_filename]
                if query_descriptors_sift is not None:
                    query_metadata.loc[idx,'matching_attempt'] = 'success'
                    query_metadata = inference_per_query_for_re_identification(query_descriptors_sift, query_metadata, query_image_path, faiss_index,  all_labels_train, all_serials_train)
    
    return query_metadata

def remove_and_recreate_temp_shards_dir(temp_shards_dir):
    # Check if directory exists before attempting to remove
    if os.path.exists(temp_shards_dir):
        shutil.rmtree(temp_shards_dir)  # Deletes the directory and all files inside it
        print(f"Removed directory: {temp_shards_dir}")
    else:
        print(f"Directory {temp_shards_dir} does not exist.")
    os.makedirs(temp_shards_dir, exist_ok=True)

def main():
    
    # Call the profiling function
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Load data dirs 
    root_dir, _ = load_data_dirs()
    
    # Set up logging files
    log_file_std_output, log_file_err_output = log_to_file(root_dir, 'matching_algorithm')
    
    # Load pkl files for descritors
    descriptors_data = load_pkl_files(root_dir)
    
    # Load reference metadata csv data
    metadata_reference_filepath = os.path.join(root_dir, 'reference_dir', 'metadata_reference.csv')
    reference_metadata = load_metadata_file(metadata_reference_filepath)
    
    # Build shards from catalog
    sharding_results_dir = os.path.join(root_dir, 'reference_dir')
    shards_dict = build_shards_from_catalog(reference_metadata, sharding_results_dir)
    
    # Load query metadata csv data
    metadata_query_filepath = os.path.join(root_dir, 'query_dir', 'metadata_query.csv')
    query_metadata = load_metadata_file(metadata_query_filepath)
    
    # Create a temp directory to store query metadata shards
    temp_shards_dir = os.path.join(root_dir, 'query_dir', 'temp')
    remove_and_recreate_temp_shards_dir(temp_shards_dir)

    # Iterate over shards
    for shard_name, shard_data in shards_dict.items():
        
        print(f"\nProcessing shard: {shard_name}")
        print(f"Number of identities in the shard: {len(shard_data)}")
        
        # Create a copy of the query metadata
        query_metadata_shard = query_metadata.copy()
        shard_metadata_query_filepath = os.path.join(temp_shards_dir, f"metadata_query_shard_{shard_name}.csv")
        
        # Train a faiss index
        all_descriptors_train, all_labels_train, all_serials_train = reshape_reference_data_for_faiss(descriptors_data['reference'], keys=shard_data)
        faiss_index = train_faiss(all_descriptors_train)
    
        # Call matching function
        query_metadata_shard = sweep_over_query_images_for_inference(shard_metadata_query_filepath, descriptors_data['query'], query_metadata_shard, faiss_index,  all_labels_train, all_serials_train)
        query_metadata_shard.to_csv(shard_metadata_query_filepath, index=False)
    
    # Merge shard files
    output_merged_query_filename = metadata_query_filepath
    merge_query_shards(temp_shards_dir, output_merged_query_filename)
    
    # Disabling the profiling function
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    
    # Print memory usage
    print_memory_usage()
    
    # Restore stdout 
    restore_stdout(log_file_std_output, log_file_err_output)

if __name__ == "__main__":
    main()