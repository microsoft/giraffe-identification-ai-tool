# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
import pickle
import pstats
import cProfile
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers_matching import load_data_dirs, load_metadata_file, load_pkl_files
from utils.helpers_matching import print_memory_usage, log_to_file, restore_stdout
from configs.config_matching import gt_keyname_col

def update_original_ref_db(query_matching_results_df, descriptors_data, metadata_ref_original, columns_needed_for_merging):
    
    print("\n============ Database Update Summary ============")
    
    # Load initial descriptor files
    query_descriptors_data = descriptors_data['query']
    ref_descriptors_data = descriptors_data['reference']
    print("\nInitial number of giraffe photos in ref_descriptors_data:", len(ref_descriptors_data))

    
    # Concatenate metadata files
    df_for_merging = query_matching_results_df[columns_needed_for_merging].copy()
    df_for_merging = df_for_merging.rename(columns={'new_id_aligned_with_ref': gt_keyname_col})
    df_for_merging['FileName'] = df_for_merging['path_relative_to_root'].map(lambda x: os.path.basename(x))
    df_for_merging['#Serial'] = df_for_merging['#Serial'].astype(int)
    df_for_merging[gt_keyname_col] = df_for_merging[gt_keyname_col].astype(int)
    
    
    print("\nMetadata size BEFORE updating operation:", metadata_ref_original.shape)
    metadata_ref_updated = pd.concat([metadata_ref_original, df_for_merging], axis=0).reset_index(drop=True)
    print("\nMetadata size AFTER updating operation:", metadata_ref_updated.shape)
    
    # Counters for new entries and updates
    new_entries_descriptors = 0
    updates_descriptors = 0
    counter = 0
    
    # Loop over query items
    for _, query_row in tqdm(query_matching_results_df.iterrows()):
        
        # Set some variables
        counter += 1
        new_giraffe_id = int(query_row['new_id_aligned_with_ref'])
        new_universal_id = int(query_row['#Serial'])
        query_descriptors_sift = query_descriptors_data[os.path.basename(query_row['path_relative_to_root'])]
        # print(query_descriptors_sift.shape)
    
        # Update ref_descriptors_data dict
        if new_giraffe_id in ref_descriptors_data:
            ref_descriptors_data[new_giraffe_id][0].append(query_descriptors_sift)
            ref_descriptors_data[new_giraffe_id][1].append(new_universal_id)
            updates_descriptors += 1  # Count as an update
        else:
            ref_descriptors_data[new_giraffe_id] = ([query_descriptors_sift], [new_universal_id])
            new_entries_descriptors += 1  # Count as a new entry

        if counter == 5:
            break
        
    # Print final stats after update
    print("\nFinal number of giraffe photos in ref_descriptors_data:", len(ref_descriptors_data))

    # Summary of changes
    print(f"\nNew entries in ref_descriptors_data: {new_entries_descriptors}, Updates in ref_descriptors_data: {updates_descriptors}")
    print("\n=================================================")

    return ref_descriptors_data, metadata_ref_updated

def save_updated_db(root_dir, ref_descriptors_data, metadata_ref_updated):
    
    # Save ref metadata file
    metadata_ref_updated.to_csv(os.path.join(root_dir, 'reference_dir', 'metadata_reference_updated.csv'), index=False)
    print('\nSaved new metadata file.')
    
    # Save ref_descriptors_data pkl data
    with open(os.path.join(root_dir, 'reference_dir', 'giraffes_reference_descriptors_updated.pkl'), 'wb') as f:
        pickle.dump(ref_descriptors_data, f)
        print('Saved new ref_descriptors_data file.')

def main():
    
    # Call the profiling function
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Load blob data dirs 
    root_dir ,_ = load_data_dirs()
    
    # Set up logging files
    log_file_std_output, log_file_err_output = log_to_file(root_dir, 'update_ref_database')
    
    # Load metadata
    metadata = {}
    for partition in ['query', 'reference']:
        metadata_filepath = os.path.join(root_dir, partition + '_dir', 'metadata_' + partition + '.csv')
        metadata[partition] = load_metadata_file(metadata_filepath)
    query_metadata = metadata['query']

    # Load pkl files for descritors
    descriptors_data = load_pkl_files(root_dir)

    # Define required columns in processed query table
    columns_needed_for_merging = [
        '#Serial', 'path_relative_to_root', 
        'giraffes_count', 'detection_coverage', 'segmentation_coverage', 'combined_coverage', 'descriptors_size', 
        'database_update_status', 'new_id_aligned_with_ref'
    ]
    # Define required columns are not NaN in processed query table
    required_non_nan_columns = ['#Serial', 'path_relative_to_root', 'new_id_aligned_with_ref']
    
    # Check if all required columns exist
    missing_columns = [col for col in columns_needed_for_merging if col not in query_metadata.columns]
    if missing_columns:
        print(f"Error: The following required columns are missing from query_metadata: {missing_columns}")
    else:
        # Filter processed rows for id assignment
        query_matching_results_df = query_metadata[query_metadata['database_update_status'] == 'processed'].copy()
        print(f"Filter processed dataframe shape based on query_metadata: {query_matching_results_df.shape}")
        
        # Ensure required columns are not NaN
        query_matching_results_df = query_matching_results_df.dropna(subset=required_non_nan_columns)

        # Check if any rows remain after filtering
        if query_matching_results_df.empty:
            print("Error: No rows remain after filtering. Exiting.")
        else:
            # Update the original reference database
            ref_descriptors_data, metadata_ref_updated = update_original_ref_db(query_matching_results_df, descriptors_data, metadata['reference'], columns_needed_for_merging)
            save_updated_db(root_dir, ref_descriptors_data, metadata_ref_updated)
    
    # Update query metadata
    query_metadata.loc[query_matching_results_df.index, 'final_update_status'] = 'completed'
    query_metadata.to_csv(os.path.join(root_dir, 'query_dir', 'metadata_query.csv'), index=False)
    
    # Print memory usage
    print_memory_usage()
    
    # Disabling the profiling function
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    
    # Restore stdout 
    restore_stdout(log_file_std_output, log_file_err_output)

if __name__ == "__main__":
    main()