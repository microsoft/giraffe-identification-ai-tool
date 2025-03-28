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
from sklearn.metrics.cluster import adjusted_rand_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_matching import gt_keyname_col
from configs.config_matching import faiss_distance_cutoff, faiss_mode_cutoff, faiss_distance_cutoff_re_id, faiss_mode_cutoff_re_id
from utils.helpers_matching import print_memory_usage, log_to_file, restore_stdout
from utils.helpers_matching import load_data_dirs, load_metadata_file
from utils.helpers_matching import save_merged_accuracy_results


def find_query_out_of_sample_records(metadata_table_ref, metadata_table_query):
    reference_ids = set(metadata_table_ref[gt_keyname_col])

    for idx, row in metadata_table_query.iterrows():
        if row[gt_keyname_col] in reference_ids:
            metadata_table_query.loc[idx, 'out_of_sample'] = 'False'
        else:
            metadata_table_query.loc[idx, 'out_of_sample'] = 'True'
    
    FP_table = metadata_table_query[(metadata_table_query['out_of_sample']=='True') & (metadata_table_query['matching_status']=='matched')]
    TN_table  = metadata_table_query[(metadata_table_query['out_of_sample']=='True') & (metadata_table_query['matching_status']=='not_matched')]
    TP_table  = metadata_table_query[(metadata_table_query['out_of_sample']=='False') & (metadata_table_query['matching_status']=='matched')]
    FN_table  = metadata_table_query[(metadata_table_query['out_of_sample']=='False') & (metadata_table_query['matching_status']=='not_matched')]
    P_table = metadata_table_query[metadata_table_query['matching_status']=='matched']
    N_table = metadata_table_query[metadata_table_query['matching_status']=='not_matched']
    
    return metadata_table_query, P_table, N_table, FP_table, TN_table, TP_table, FN_table

def evaluate_accuracy_high_level(FP_table, TN_table, TP_table, FN_table):
    
    # Compute confusion matrix
    FP = len(FP_table)
    TN = len(TN_table)
    TP = len(TP_table)
    FN = len(FN_table)
    
    # Compute metrics for positive class
    precision_pos = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    recall_pos = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    f1_pos = (2 * precision_pos * recall_pos) / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else np.nan
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else np.nan

    # Compute metrics for negative class
    precision_neg = TN / (TN + FN) if (TN + FN) > 0 else np.nan
    recall_neg = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    f1_neg = (2 * precision_neg * recall_neg) / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else np.nan

    metrics = {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'precision_positive': precision_pos,
        'recall_positive': recall_pos,
        'f1_score_positive': f1_pos,
        'precision_negative': precision_neg,
        'recall_negative': recall_neg,
        'f1_score_negative': f1_neg,
        'overall_accuracy': accuracy
    }

    return metrics

def evaluate_accuracy_for_re_identified_items(TP_table):

    # Compute accuracy on matched items only
    TP_table['correctness'] = (TP_table[gt_keyname_col] == TP_table['new_id_aligned_with_ref'])
    accuracy_matched = TP_table['correctness'].sum() / TP_table.shape[0] if TP_table.shape[0] > 0 else np.nan

    return accuracy_matched

def evaluate_accuracy_partitioning_new_items(N_table):
    
    a, b = N_table['new_id_aligned_with_ref'].astype(int), N_table[gt_keyname_col].astype(int)

    print("\n--- Info about Assigned IDs and Ground Truth Data ---")
    print(f"Total assigned IDs: {len(a)}")
    print(f"Total ground truth entries: {len(b)}")

    print("\nAssigned IDs:", a.values if len(a) <= 10 else f"{a.values[:10]} ... (truncated)")
    print("Ground Truth IDs:", b.values if len(b) <= 10 else f"{b.values[:10]} ... (truncated)")


    # Compute ARI if data exists
    if not a.empty and not b.empty:
        ari_score = adjusted_rand_score(a, b)
        print("Adjusted Rand Index:", ari_score)
        return ari_score

    print("No data to compute Adjusted Rand Index.")
    
    return None

def collect_accuracy_info(N_table, FP_table, TN_table, TP_table, FN_table, ref_data_size):
    # Initialize accuracy results
    accuracy_results = {
        'thresholds_dist_re_id': faiss_distance_cutoff_re_id,
        'thresholds_counts_re_id': faiss_mode_cutoff_re_id,
        'thresholds_dist_partitioning': faiss_distance_cutoff,
        'thresholds_counts_partitioning': faiss_mode_cutoff,
        'total_items_in_ref': ref_data_size,
        'total_number_of_queries' : len(FP_table) + len(TN_table) +  len(TP_table) + len(FN_table),
        'total_number_of_pos_queries' : len(FN_table) + len(TP_table),
        'total_number_of_neg_queries' : len(TN_table) +  len(FP_table)
    }

    # Compute different accuracy metrics
    accuracy_results.update(evaluate_accuracy_high_level(FP_table, TN_table, TP_table, FN_table))
    accuracy_results['accuracy_re_identified_items'] = evaluate_accuracy_for_re_identified_items(TP_table)
    accuracy_results['adjusted_rand_index_partitioning'] = evaluate_accuracy_partitioning_new_items(N_table)
    accuracy_results = pd.DataFrame(list(accuracy_results.items()), columns=['Metric', 'Value'])
    accuracy_results['Value'] = accuracy_results['Value'].apply(lambda x: round(x, 2) if isinstance(x, (float)) else x)
    print(accuracy_results)
    return accuracy_results

def main():
    """ IMPORTANT: This code makes sense if we have accepted all results of the model in initial matching round. """        
    # Call the profiling function
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Load data dirs 
    root_dir, _= load_data_dirs()
    
    # Set up logging files
    log_file_std_output, log_file_err_output = log_to_file(root_dir, 'compute_accuracy')
    
    # Load metadata csv data
    metadata_filepath_query = os.path.join(root_dir, 'query_dir', 'metadata_query.csv')
    query_metadata = load_metadata_file(metadata_filepath_query)
    query_metadata.drop(columns=['out_of_sample'], inplace=True, errors='ignore')

    
    metadata_filepath_ref = os.path.join(root_dir, 'reference_dir', 'metadata_reference.csv')
    ref_metadata = load_metadata_file(metadata_filepath_ref)
    ref_data_size = int(ref_metadata.shape[0])

    # Measure accuracy
    if gt_keyname_col in ref_metadata.columns: 
        if gt_keyname_col in query_metadata.columns and 'new_id_aligned_with_ref' in query_metadata.columns:
            # -1 in gt_keyname_col indicate the user is sure item does not exist in ref but does not actual label for it 
            filtered_data_with_valid_gt = query_metadata[(query_metadata['new_id_aligned_with_ref'].notna()) & (query_metadata[gt_keyname_col].notna()) & (query_metadata[gt_keyname_col] != -1)].copy()
            
            if len(filtered_data_with_valid_gt) != 0:  # Check if there are any valid rows
                
                # Make sure types are int
                filtered_data_with_valid_gt['new_id_aligned_with_ref'] = filtered_data_with_valid_gt['new_id_aligned_with_ref'].astype(int)
                filtered_data_with_valid_gt[gt_keyname_col] = filtered_data_with_valid_gt[gt_keyname_col].astype(int)
                
                # Find in/out of sample information about query items
                metadata_table_query, _, N_table, FP_table, TN_table, TP_table, FN_table = find_query_out_of_sample_records(ref_metadata, filtered_data_with_valid_gt)
                metadata_table_query.to_csv(os.path.join(root_dir, 'query_dir', 'metadata_query.csv'), index=False)
                
                # Evaluate and save matching results 
                accuracy_results = collect_accuracy_info(N_table, FP_table, TN_table, TP_table, FN_table, ref_data_size)
                save_merged_accuracy_results(accuracy_results, root_dir, True)
                
        else:
            print("Warning: {} and/or new_id_aligned_with_ref columns not found in {}.".format(gt_keyname_col, metadata_filepath_query))
    else:
        print("Warning: " + gt_keyname_col + " column not found in '{}'.".format(metadata_filepath_ref))

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