# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
import pstats
import cProfile
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils_matching import normalize, reshape_reference_data_for_faiss
from utils.utils_matching import train_faiss, load_trained_faiss_ref, train_faiss_partial_data
from utils.utils_matching import run_union_find, replace_negatives_with_unique_values
from utils.helpers_matching import calculate_partitioning_accuracy, save_partitioning_results
from utils.helpers_matching import get_new_label_Id, get_new_serial_Id
from utils.helpers_matching import load_data_dirs, load_metadata_file, load_pkl_files
from utils.helpers_matching import print_memory_usage, log_to_file, restore_stdout
from configs.config_matching import faiss_distance_cutoff, faiss_mode_cutoff, formatted_string_for_setup
from configs.config_matching import n_features, faiss_index_dir, partitioning_initialization, inference_mode, gt_keyname_col
from configs.config_matching import auto_accept_model_matching_results


def get_update_status(row, reference_basenames):
    if row['filename'] in reference_basenames:
        return 'existing'
    elif row['human_input'] in ['AcceptId', 'AssignNewId']:
        return 'processed'
    else:
        return 'skipped'

def get_serial_to_aid_dict(df_assign_new_id):
    serial_to_aid_dict = []
    if gt_keyname_col in  df_assign_new_id.columns:
        df_assign_new_id[(df_assign_new_id[gt_keyname_col] != -1) & df_assign_new_id[gt_keyname_col].notna()]
        serial_to_aid_dict = df_assign_new_id.set_index('#Serial')[gt_keyname_col].to_dict()
    print('\n length of serial_to_aid_dict {}'.format(len(serial_to_aid_dict)))
    return serial_to_aid_dict
    
def re_assign_ids_to_align_with_ref_db(new_ids_ls, first_new_giraffe_id):
    if inference_mode == 'off':
        first_new_giraffe_id += 20000000
    unique_ids = sorted(set(new_ids_ls))  # Ensure consistent ordering
    list_of_new_ids = range(first_new_giraffe_id, first_new_giraffe_id + len(unique_ids))
    
    mapping = dict(zip(unique_ids, list_of_new_ids))  # Create a mapping of old IDs to new IDs
    new_ids_ls_revised = [mapping[item] for item in new_ids_ls]  # Apply mapping
    
    return np.array(new_ids_ls_revised)

def filter_data_with_human_inputs(metadata, first_new_serial):
    
    """ This function helps select the qualifed rows in query metadata for updating the reference dataset. """
    
    query_metadata = metadata['query']
    reference_metadata = metadata['reference']
    
    # If there are some existing 'new_id_aligned_with_ref' column drop it
    print("Columns before drop:", query_metadata.columns.tolist())
    query_metadata.drop(columns=['new_id_aligned_with_ref'], inplace=True, errors='ignore')
    print("Columns after drop:", query_metadata.columns.tolist())

    # Get basename for files in both dataframes
    query_metadata['filename'] = query_metadata['path_relative_to_root'].apply(os.path.basename)
    reference_metadata['filename'] = reference_metadata['path_relative_to_root'].apply(os.path.basename)
    reference_basenames = set(reference_metadata['filename'])
    
    # Add database_update_status column to focus on items that are "not already in ref" and "approved by human experts"
    query_metadata['database_update_status'] = query_metadata.apply(lambda x: get_update_status(x, reference_basenames), axis=1)
    query_metadata.drop(columns = ['filename'], inplace=True)
    reference_metadata.drop(columns = ['filename'], inplace=True)    

    # Update the '#Serial' column directly in query_metadata for rows qualified to be processed
    filtered_df_to_be_added_to_db = query_metadata[query_metadata['database_update_status'] == 'processed']
    query_metadata.loc[query_metadata['database_update_status'] == 'processed', '#Serial'] = range(first_new_serial, first_new_serial + len(filtered_df_to_be_added_to_db))

    # Reorder columns, keeping '#Serial' as the first column
    query_metadata = query_metadata[['#Serial'] + [col for col in query_metadata.columns if col != '#Serial']]
    
    return query_metadata

def build_faiss_index(df_assign_new_id, query_descriptors_data):
    
    # Reshape data to train an index
    ref_for_query_batch_data_descriptors = {}
    for _, query_row in df_assign_new_id.iterrows():
        image_filename_key = os.path.basename(query_row['path_relative_to_root'])
        query_descriptors_sift = query_descriptors_data[image_filename_key]
        ref_for_query_batch_data_descriptors[query_row['#Serial']] = ([np.float32(query_descriptors_sift)], [query_row['#Serial']]) # to keep format for next line
    all_descriptors_train, all_labels_train, all_serials_train = reshape_reference_data_for_faiss(ref_for_query_batch_data_descriptors)
    all_descriptors_train_normalized = normalize(all_descriptors_train.astype(np.float32))
    
    if partitioning_initialization == 0:
        # Build a dedicated index for all query items in new batch
        all_serials_train_ref = []
        faiss_index = train_faiss(all_descriptors_train)
    
    elif partitioning_initialization == 1:
        # Create an index based on some amount of data and use as initial index
        root_dir ,_ = load_data_dirs()
        faiss_index, all_descriptors_train_ref, all_labels_train_ref, all_serials_train_ref = train_faiss_partial_data(root_dir)
        faiss_index.add(all_descriptors_train_normalized.astype(np.float32))
        all_descriptors_train = np.concatenate([all_descriptors_train_ref, all_descriptors_train])
        all_labels_train = np.concatenate([all_labels_train_ref, all_labels_train])
        all_serials_train = np.concatenate([all_serials_train_ref, all_serials_train])
    else:
        # Load the full reference index and use as initial index
        faiss_index, all_descriptors_train_ref, all_labels_train_ref, all_serials_train_ref = load_trained_faiss_ref(faiss_index_dir)
        faiss_index.add(all_descriptors_train_normalized.astype(np.float32))
        all_descriptors_train = np.concatenate([all_descriptors_train_ref, all_descriptors_train])
        all_labels_train = np.concatenate([all_labels_train_ref, all_labels_train])
        all_serials_train = np.concatenate([all_serials_train_ref, all_serials_train])

    return faiss_index, all_descriptors_train, all_labels_train, all_serials_train, all_serials_train_ref

def record_ids_for_accepted_items_by_expert(query_metadata):
        
        # Check if 'matched_label_1' has NaN values
        if query_metadata['matched_label_1'].isna().any():
            print("Warning: There are NaN values in the 'matched_label_1' column!")
        
        # Complete new_id_aligned_with_ref for accepted ids for matched items
        query_metadata.loc[query_metadata['human_input'] == 'AcceptId', 'new_id_aligned_with_ref'] = query_metadata.loc[query_metadata['human_input'] == 'AcceptId', 'matched_label_1']
        return query_metadata

def run_partitioning_algorithm(df_assign_new_id, query_descriptors_data, first_new_giraffe_id):
    
    # Get mapping between serial number and ground truth labels
    serial_to_aid_dict = get_serial_to_aid_dict(df_assign_new_id)

    # Build faiss
    faiss_index, _, _, all_serials_train, all_serials_train_ref = build_faiss_index(df_assign_new_id, query_descriptors_data)
    
    # Initialize some lists
    
    pred_matches_ls = []
    query_serials_ls = []
    serials_count_ls = []
    distances_ls = []
    image_filepath_ls = []
    
    print('\n size of serial list associated with initial faiss index is {}'.format(len(all_serials_train)))
    print(f"Number of vectors in the initial index after adding new items: {faiss_index.ntotal} \n")

    # Iterate through each query row
    for _, query_row in df_assign_new_id.iterrows():
        
        image_filename_key = os.path.basename(query_row['path_relative_to_root'])
        query_descriptors_sift = query_descriptors_data[image_filename_key]

        # Perform FAISS inference to find closest query
        print('\n size of serial list associated with faiss index is {}'.format(len(all_serials_train)))
        print(f"Number of vectors in the updated index after adding new items: {faiss_index.ntotal} \n")
        
        
        exclusion_set = set(all_serials_train_ref)
        print('\n exclusion_set size: {}'.format(len(exclusion_set)))
        top_serials, query_serial, top_serial_counts, top_distance_avgs, image_filepath= inference_per_query_for_partitioning(faiss_index, query_descriptors_sift, query_row, all_serials_train, exclusion_set)
        
        pred_matches_ls.extend(top_serials)
        query_serials_ls.extend(query_serial)
        serials_count_ls.extend(top_serial_counts)
        distances_ls.extend(top_distance_avgs)
        image_filepath_ls.extend(image_filepath)
        
        print('\n top_serials', len(top_serials))
        print('\n query_serial', len(query_serial))      
        print('\n top_serial_counts', len(top_serial_counts))
        print('\n top_distance_avgs', len(top_distance_avgs))       
        print('\n image_filepath', len(image_filepath))
    
    print('\n query_serials_ls', len(query_serials_ls), '\n pred_matches_ls', len(pred_matches_ls), sep='\n')
    print('\n distances_ls', len(distances_ls), '\n serials_count_ls', len(serials_count_ls), sep='\n')
    
    # Create list of actual ground truth
    actual_ground_truth_ls = [serial_to_aid_dict[serial] if serial in serial_to_aid_dict else np.nan for serial in query_serials_ls]
    
    # Assign new ids to items not matched and indicated by -1
    pred_matches_ls_revised = replace_negatives_with_unique_values(np.array(pred_matches_ls, dtype=np.int64))
    
    # Run the union-find algorithm to get the new IDs
    value_to_new_id = run_union_find(np.array(query_serials_ls), pred_matches_ls_revised)

    # Mapping to assign new IDs to df
    new_ids_ls = np.array([value_to_new_id[val] if val in value_to_new_id else -1 for val in pred_matches_ls_revised])
    
    # Rename labels from partitioning of new giraffes step to align with reference db
    new_ids_ls_revised = re_assign_ids_to_align_with_ref_db(new_ids_ls, first_new_giraffe_id)
    
    # Save results
    results_df = save_partitioning_results(image_filepath_ls, query_serials_ls, pred_matches_ls, pred_matches_ls_revised, new_ids_ls_revised, serials_count_ls, distances_ls, actual_ground_truth_ls)
    
    return results_df

def inference_per_query_for_partitioning(faiss_index, query_descriptors_sift, query_row, all_serials_train, exclusion_set):
    
    # Add serial number of the query to the exclusion set for searching index
    query_serial = query_row['#Serial']
    exclusion_set.add(int(query_serial)) 
    print('\n exclusion set size')
    print(len(exclusion_set))

   # Normalize query descriptors
    query_descriptors_sift_normalized = normalize(query_descriptors_sift)

    # Initialize arrays to track matches
    n_keypoints = query_descriptors_sift_normalized.shape[0]
    filtered_distances = np.full(n_keypoints, np.inf)  # Initialize with inf distances
    filtered_indices = np.full(n_keypoints, -1)       # Initialize with invalid indices

    # Set initial unmatched query indices
    unmatched_indices = np.arange(n_keypoints)

    # Start with a small k and dynamically adjust
    k = 2
    n_attempts = 3 * (n_features + 1)
    
    while unmatched_indices.size > 0 and n_attempts > 0:
        
        # Track no of attempts to find a neighbor from index that is not equal to itself
        n_attempts -= 1
        
        # Perform faiss search for unmatched keypoints
        distances, indices = faiss_index.search(query_descriptors_sift_normalized[unmatched_indices].astype(np.float32), k=k)

        # Iterate through results for unmatched keypoints
        still_unmatched = []
        for i, query_idx in enumerate(unmatched_indices):
            valid_match_found = False

            # Check all neighbors for a valid match
            for dist, idx in zip(distances[i], indices[i]): # i denotes each keypoint for query
                if dist < faiss_distance_cutoff:
                    if all_serials_train[idx] not in exclusion_set:  # Ensure match is valid
                        filtered_distances[query_idx] = dist
                        filtered_indices[query_idx] = idx
                        valid_match_found = True
                        break  # Stop after finding the first valid match
                else:
                    filtered_distances[query_idx] = np.inf
                    filtered_indices[query_idx] = -1
                    valid_match_found = True
                    break  # Stop after finding that no valid match will be found

            if not valid_match_found:
                still_unmatched.append(query_idx)

        # Update unmatched indices for the next iteration
        unmatched_indices = np.array(still_unmatched)

        # Increase k for the next round of search if needed
        k += 1
    print('\n number of attempts to find a new item is {}'.format(3 * (n_features + 1) - n_attempts))

    # Generate predicted labels
    pred_serials = [all_serials_train[idx] if idx != -1 else -1 for idx in filtered_indices]
    
    print(len(filtered_distances))
    print(len(filtered_indices))
    print(len(pred_serials))
    
    # Apply the distance threshold to filter distances and labels
    mask = filtered_distances < faiss_distance_cutoff
    filtered_distances = np.array(filtered_distances)[mask]
    filtered_serials = np.array(pred_serials)[mask]
    
    print(len(filtered_distances))
    print(len(filtered_indices))
    print(len(filtered_serials))

    # Initialize a dictionary to store aggregated scores and counts for each prediction
    serial_scores_mapping = {}

    # Aggregate scores and counts for each predicted label
    for serial, score in zip(filtered_serials, filtered_distances):
        if serial != -1:
            if serial in serial_scores_mapping:
                serial_scores_mapping[serial][0] = query_row['#Serial'] # collect the ground truth label for specific image/query/serial
                serial_scores_mapping[serial][1] += 1     # Increment the count
                serial_scores_mapping[serial][2] += score  # Increment the score
                serial_scores_mapping[serial][3] = serial_scores_mapping[serial][2]/serial_scores_mapping[serial][1]  # calc the avg score
                serial_scores_mapping[serial][4] = query_row['path_relative_to_root']
                
            else:
                serial_scores_mapping[serial] = [query_row['#Serial'], 1, score, score, query_row['path_relative_to_root']]  # Initialize with score and count = 1

    # Sort labels by aggregated scores in descending order
    sorted_recommendations = sorted(serial_scores_mapping.items(), key=lambda x: x[1][1], reverse=True)
    
    print(f'\nserial:', query_row['#Serial'])
        
    filtered_recommendations = [
    (serial, value[0], value[1], value[3], value[4]) 
    for serial, value in sorted_recommendations 
    if value[1] > faiss_mode_cutoff
    ]

    # If there are filtered results, proceed with zip
    if filtered_recommendations:
        top_serials, query_serial, top_serial_counts, top_distance_avgs, image_filepath = zip(*filtered_recommendations)
    else:
        # If no valid recommendations after filtering, assign empty lists
        top_serials = [-1]
        query_serial = [query_row['#Serial']]
        image_filepath = [query_row['path_relative_to_root']]
        top_serial_counts = [np.nan]
        top_distance_avgs = [np.nan]

    
    return top_serials, query_serial, top_serial_counts, top_distance_avgs, image_filepath

def merge_new_ids_results_and_metadata_query(query_metadata, results_df):
    
    # Results df unique items
    unique_results_df = results_df.loc[:, ['path_relative_to_root', 'new_id_aligned_with_ref']].drop_duplicates()
    
    # Check for duplicate entries in 'unique_results_df' based on 'path_relative_to_root'
    duplicates = unique_results_df[unique_results_df.duplicated('path_relative_to_root', keep=False)]

    # If duplicates exist, warn the user
    if not duplicates.empty:
    
        print(f"Warning: There are {duplicates['path_relative_to_root'].nunique()} rows with duplicate 'path_relative_to_root' values in 'results_df'. These rows will not be processed correctly.")
        print(duplicates)
        return query_metadata
    
    # Else perform the merge
    else:
    
        merged_df = query_metadata.merge(
            unique_results_df, 
            on='path_relative_to_root', 
            how='left')
        return merged_df

def main():
    
    # Call the profiling function
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Load blob data dirs 
    root_dir ,_ = load_data_dirs()
    
    # Set up logging files
    log_file_std_output, log_file_err_output = log_to_file(root_dir, 'partition_new_items')

    # Load metadata
    metadata = {}
    for partition in ['query', 'reference']:
        metadata_filepath = os.path.join(root_dir, partition + '_dir', 'metadata_' + partition + '.csv')
        metadata[partition] = load_metadata_file(metadata_filepath)
        
    # Get starting IDs for updating database
    first_new_giraffe_id = get_new_label_Id(metadata['reference'])
    first_new_universal_id = get_new_serial_Id(metadata['reference'])
    
    # Load pkl files for descritors
    descriptors_data = load_pkl_files(root_dir, ['query'])
    
    # Add human input column in case of validation mode only
    if auto_accept_model_matching_results:
        # Print a warning indicating that we are accepting all model results
        print("Warning: Accepting all model results from the initial round of matching for validation purpose only.")

        # Assign 'human_input' based on 'matching_status'
        metadata['query'].loc[metadata['query']['matching_status'] == 'matched', 'human_input'] = 'AcceptId'
        metadata['query'].loc[metadata['query']['matching_status'] == 'not_matched', 'human_input'] = 'AssignNewId'
        
    # Check if all required columns exist
    columns_needed_for_processing = ['path_relative_to_root', 'matched_label_1', 'human_input']
    missing_columns = [col for col in columns_needed_for_processing if col not in metadata['query'].columns]
    
    if missing_columns:
        print(f"Error: The following required columns are missing from query_metadata: {missing_columns}")
    else:
        # Filter data for matching results with human inputs
        query_metadata = filter_data_with_human_inputs(metadata, first_new_universal_id)
        
        # Filter data for rows where 'human_input' is 'AssignNewId'
        df_assign_new_id = query_metadata[query_metadata['human_input'] == 'AssignNewId'].reset_index().copy()  # Work on a copy
        
        if df_assign_new_id.empty:
            print("The filtered dataFrame based on human input to analyze for partitioning is empty. Exiting...")
        else:
            print('\n df_assign_new_id shape {}'.format(df_assign_new_id.shape))

            # Partition and do new id assignments which require internal matching process within the same query batch
            results_df = run_partitioning_algorithm(df_assign_new_id, descriptors_data['query'], first_new_giraffe_id)
            
            # Calculate accuracy of partitioning
            file_path = os.path.join(root_dir, 'query_dir', 'query_to_ids_' + formatted_string_for_setup +'.csv')
            print(formatted_string_for_setup)
            print(file_path)
            _, _ = calculate_partitioning_accuracy(file_path)
                        
            # Join results with query metadata
            query_metadata = merge_new_ids_results_and_metadata_query(query_metadata, results_df)
        
        # Record re-identified items for which matched ids are accepted by expert
        query_metadata = record_ids_for_accepted_items_by_expert(query_metadata) 

        # Save updated results
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