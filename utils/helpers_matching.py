# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


import os
import sys
import csv
import time
import json
import pickle
import psutil
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from sklearn.metrics.cluster import adjusted_rand_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_matching import gt_keyname_col, formatted_string_for_setup

load_dotenv()
container_name = os.getenv("container_name")
data_root_abs_path = os.getenv("data_root_abs_path")

def get_new_label_Id(reference_metadata):
    first_new_giraffe_id = int(reference_metadata[gt_keyname_col].max()) + 1
    return first_new_giraffe_id

def get_new_serial_Id(reference_metadata):
    first_new_universal_id = int(reference_metadata['#Serial'].max()) + 1
    return first_new_universal_id

def get_img_paths_from_a_folder(root_dir, relative_to_root_image_dir):
    
    keyname_col = 'path_relative_to_root'
    
    file_path_dict = {keyname_col:[]}
    file_path_dict[keyname_col] = [os.path.join(relative_to_root_image_dir, img) for img in os.listdir(os.path.join(root_dir, relative_to_root_image_dir)) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    csv_dir = os.path.join(root_dir, 'sample_query_metadata_files')
    saved_file_path = os.path.join(csv_dir, 'metadata_query__' +relative_to_root_image_dir.replace('/', '__')+ '.csv')
    pd.DataFrame(file_path_dict).to_csv(saved_file_path, index=False)
    
    return saved_file_path

def get_img_paths_from_several_folders(root_dir, image_subdir, textfile_path):
    # Read all lines from the text file
    with open(textfile_path, 'r') as file:
        folder_paths = [line.strip() for line in file.readlines()]

    # Dictionary to store image paths
    keyname_col = 'path_relative_to_root'
    file_path_dict = {keyname_col: []}

    for a_path in folder_paths:
        image_dir = os.path.join(root_dir, image_subdir, a_path)

        # Check if the directory exists
        if not os.path.exists(image_dir):
            print(f"Directory does not exist: {image_dir}")
            continue

        # Get all image file paths in the directory
        image_files = [
            os.path.join(image_subdir, a_path, img)
            for img in os.listdir(image_dir)
            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]
        file_path_dict[keyname_col].extend(image_files)

    # Define the output CSV directory and file
    csv_dir = os.path.join(root_dir, 'sample_query_metadata_files/query_based_on_backlog_May2020_Feb2023/')
    os.makedirs(csv_dir, exist_ok=True)
    csv_file_path = os.path.join(csv_dir, 'metadata_query_backlog.csv')

    # Save the paths to the CSV file
    pd.DataFrame(file_path_dict).to_csv(csv_file_path, index=False)
    print(f"CSV file saved at: {csv_file_path}")
    
def load_metadata_file(metadata_filepath):
    keyname_col = 'path_relative_to_root'
    
    # Check if the metadata file exists
    if not os.path.exists(metadata_filepath):
        print("Error: File '{}' does not exist.".format(metadata_filepath))
        sys.exit(1) 
    
    # Load the metadata file
    try:
        df = pd.read_csv(metadata_filepath)
    except Exception as e:
        print("Failed to read '{}': {}".format(metadata_filepath, e))
        sys.exit(1)  

    # Check if keyname_col column exists in the DataFrame
    if 'path_relative_to_root' not in df.columns:
        print("Error: " + keyname_col + " column not found in '{}'.".format(metadata_filepath))
        sys.exit(1)

    # Extract filenames temporarily and check for duplicates
    filenames = df[keyname_col].apply(os.path.basename)
    duplicate_filenames = filenames[filenames.duplicated()]
    
    if not duplicate_filenames.empty:
        print("Error: Duplicate image filenames found in " + keyname_col + " column in '{}':".format(metadata_filepath))
        print(duplicate_filenames.to_string(index=False))  
        sys.exit(1)
        
    # Proceed with returning the df
    print("File '{}' exists and contains ".format(metadata_filepath) + keyname_col + " column.")
    
    return df

def load_pkl_files(root_dir, partitions=['query', 'reference']):    

    # Load descriptors pkl data
    descriptors_data = {}
    for partition in partitions:
    
        data_dir = os.path.join(root_dir, partition + '_dir')
        if not os.path.isdir(data_dir):
            print(("Directory '{}' not found.".format(data_dir)))
            sys.exit(1)
    
        descriptors_filename = 'giraffes_' + partition + '_descriptors.pkl'
        descriptors_filepath = os.path.join(data_dir, descriptors_filename)
        if not os.path.isfile(descriptors_filepath):
            print("File '{}' not found.".format(descriptors_filepath))
            sys.exit(1)
        
        # Load data
        print('\nLoading ' + partition + ' descriptors file: {}'.format(descriptors_filename))
        start_time = time.time()
        
        with open(descriptors_filepath, 'rb') as f:
            descriptors_data[partition] = pickle.load(f)
            if descriptors_data[partition] is None:
                print("Image descriptors .pkl files does not have valid data: '{}'".format(descriptors_filepath))
                sys.exit(1)
        
        print('\nLoading time for ' + partition + ' descriptors loading {:.6f} seconds'.format(time.time() - start_time))
    
    return descriptors_data

def load_data_dirs():
    root_dir = os.path.join(data_root_abs_path, container_name)
    processed_img_dir = os.path.join(root_dir, 'processed_images')
    os.makedirs(processed_img_dir, exist_ok=True)
    
    return root_dir, processed_img_dir

def save_accuracy_to_csv_with_timestamp(accuracy, output_dir, run_name='run'):

    # Define the file path for saving accuracy
    csv_file_name = f'{run_name}_accuracy.csv'
    csv_file_path = os.path.join(output_dir, csv_file_name)
    
    # Open the file in write mode to overwrite existing content
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header first
        writer.writerow(['timestamp', 'accuracy'])
        
        # Write the current accuracy with a timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, accuracy])
    
    print(f"Accuracy saved to {csv_file_path}")

def save_all_accuracy_to_csv(accuracy, accuracy_matched, accuracy_not_matched, save_dir, run_name='run'):
    # Create a dictionary of all variables
    data = {
        'Metric': ['accuracy', 'accuracy_matched', 'accuracy_not_matched'],
        'Value': [accuracy, accuracy_matched, accuracy_not_matched]
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as CSV file
    save_path = os.path.join(save_dir, f'{run_name}_accuracy.csv')
    df.to_csv(save_path, index=False)
    print(f"Accuracy data saved to {save_path}")

def save_partitioning_results(image_filepath_ls, query_serials_ls, pred_matches_ls, pred_matches_ls_revised, new_ids_ls, serials_count_ls, distances_ls, actual_ground_truth_ls):
    
    # Step 1: Create a DataFrame
    df = pd.DataFrame({
        'path_relative_to_root': image_filepath_ls,
        'query_serial': query_serials_ls,
        'actual_ground_truth': actual_ground_truth_ls,
        'pred_match': pred_matches_ls,
        'pred_match_revised': pred_matches_ls_revised,
        'new_id_aligned_with_ref': new_ids_ls,
        'serial_count': serials_count_ls,
        'distance': distances_ls
    })
    
    # Step 2: Create a dictionary mapping query_serial to new_ids
    query_to_ids = {}
    for query_serial_int, new_id in zip(query_serials_ls, new_ids_ls):
        query_serial = str(query_serial_int)
        if query_serial not in query_to_ids:
            query_to_ids[query_serial] = []
        query_to_ids[query_serial].append(int(new_id))

    # Step 3: Save the mapping dictionary to a JSON and CSV files
    root_dir ,_ = load_data_dirs()
    output_dir = os.path.join(root_dir, 'query_dir', 'query_to_ids_' + formatted_string_for_setup + '.json')
    with open(output_dir, 'w') as json_file:
        json.dump(query_to_ids, json_file, indent=4)
    df.to_csv(os.path.join(root_dir, 'query_dir', 'query_to_ids_' + formatted_string_for_setup + '.csv'), index=False)

    # Display the DataFrame and dictionary
    print("DataFrame:")
    print(df.head(5))
    print("\nMapping Dictionary:")
    print(query_to_ids)
    
    return df

def log_to_file(root_dir, keyword, subdir='query_dir'):
    
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_dir = os.path.join(root_dir, subdir, 'logs')
    os.makedirs(log_file_dir, exist_ok=True)
    
    log_filename_std_output = os.path.join(log_file_dir, 'log__std_output__' + keyword + '__' + current_datetime + '.log')
    log_filename_err_output = os.path.join(log_file_dir, 'log__err_output__' + keyword + '__' + current_datetime + '.log')

    log_file_std_output = open(log_filename_std_output, 'w')
    log_file_err_output = open(log_filename_err_output, 'w')
    
    sys.stdout = log_file_std_output
    sys.stderr = log_file_err_output
    
    return log_file_std_output, log_file_err_output

def restore_stdout(log_file_std_output, log_file_err_output):
    
    # Restore stdout to its original state and close the log file.
    log_file_std_output.close()
    log_file_err_output.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

def print_memory_usage():
    print("\nMemory Usage Details:")
    process = psutil.Process(os.getpid())
    print("\nMemory Usage Summary: {:.2f} MB".format(process.memory_info().rss / 1024 ** 2))

def calculate_partitioning_accuracy(file_path):
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping accuracy calculation.")
        return

    # Load data
    df = pd.read_csv(file_path)

    # Required columns
    required_cols = {'query_serial', 'new_id_aligned_with_ref', 'actual_ground_truth'}
    
    # Check if all required columns exist
    if not required_cols.issubset(df.columns):
        print(f"Missing required columns {required_cols - set(df.columns)} in {file_path}. Skipping accuracy calculation.")
        return

    # Check if there are multiple `new_id_aligned_with_ref` values per `query_serial`
    duplicate_counts = df.groupby('query_serial')['new_id_aligned_with_ref'].nunique()
    multiple_ids = duplicate_counts[duplicate_counts > 1]

    if not multiple_ids.empty:
        print(f"Warning: {len(multiple_ids)} paths have multiple `new_id_aligned_with_ref` values. This may affect accuracy calculations.")
        print(multiple_ids)

    # Remove duplicates
    df = df.drop_duplicates(subset=['query_serial', 'new_id_aligned_with_ref'])

    # Filter rows with non-null values in relevant columns
    valid_rows = df.dropna(subset=['new_id_aligned_with_ref', 'actual_ground_truth'])

    # Extract values
    assigned_ids, ground_truth_ids = valid_rows['new_id_aligned_with_ref'].values, valid_rows['actual_ground_truth'].values

    # Compute ARI if valid data exists
    if len(assigned_ids) > 0 and len(ground_truth_ids) > 0:
        ari_score = adjusted_rand_score(assigned_ids, ground_truth_ids)
        print("Adjusted Rand Index:", ari_score)
    else:
        print("No valid data to compute Adjusted Rand Index.")
    
    return assigned_ids, ground_truth_ids

def save_merged_accuracy_results(accuracy_results, root_dir, reset=False):
    # Path to the CSV file
    csv_file = os.path.join(root_dir, 'query_dir', 'accuracy_results.csv')

    # Check if the CSV file already exists
    if os.path.exists(csv_file) and not reset:
        # Read the existing CSV into a dataframe
        existing_df = pd.read_csv(csv_file)
        
        # Rename the 'Value' column in accuracy_results to a unique name to avoid conflicts
        value_columns_existing = [col for col in existing_df.columns if col.startswith('Value')]
        new_value_column_name = f'Value_{len(value_columns_existing) + 1}'
        accuracy_results.rename(columns={'Value': new_value_column_name}, inplace=True)

        # Merge the existing dataframe with the new accuracy_results on the 'Metric' column
        merged_df = pd.merge(existing_df, accuracy_results, on='Metric', how='outer')
        
        # Save the updated dataframe to the same CSV
        merged_df.to_csv(csv_file, index=False)
        print(f"Merged and saved to CSV with new column {new_value_column_name}.")
    else:
        # If the CSV does not exist, simply save the new accuracy_results as a new CSV
        accuracy_results.to_csv(csv_file, index=False)
        print("Saved new accuracy results to CSV.")
