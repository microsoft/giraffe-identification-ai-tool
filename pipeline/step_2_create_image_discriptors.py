# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import cv2
import sys
import random
import pickle
import pstats
import cProfile
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers_matching import print_memory_usage, log_to_file, restore_stdout
from utils.helpers_matching  import load_data_dirs, load_metadata_file
from configs.config_matching import cropped_img_size, n_features


def update_descriptor_dict_labels_known(descriptor_dict, new_giraffe_id, new_universal_id, descriptors):
    if new_giraffe_id in descriptor_dict:
        descriptor_dict[new_giraffe_id][0].append(descriptors)
        descriptor_dict[new_giraffe_id][1].append(new_universal_id)
    else:
        descriptor_dict[new_giraffe_id] = ([descriptors], [new_universal_id])
    return descriptor_dict

def update_descriptor_dict_labels_NOT_known(descriptor_dict, img_filename, descriptors):
    descriptor_dict[img_filename] = descriptors
    return descriptor_dict

def get_sift_discriptor_based_on_saved_images_labels_known(metadata_table, metadata_filepath, ouptut_processed_img_dir):
    
    sift = cv2.SIFT_create(nfeatures=n_features, edgeThreshold=5)
    reference_data_descriptors = {}
    
    for idx, row in tqdm(metadata_table.iterrows(), total=metadata_table.shape[0]):
        
        print(idx)
        
        if idx%100 == 0:
            metadata_table.to_csv(metadata_filepath, index=False)
        
        a_giraffe_photo_path = row['path_relative_to_root']
        a_label = row['AID2021']
        a_serial = row['#Serial']
        
        parts = a_giraffe_photo_path.rsplit(".", 1)
        img_filename = os.path.basename(a_giraffe_photo_path)
        cropped_image_path = os.path.join(ouptut_processed_img_dir, "zoomed_version", img_filename).replace("." + parts[1], "_cropped_torso_zoomed." + parts[1])
        
        if os.path.isfile(cropped_image_path):
            
            cropped_image = np.array(ImageOps.exif_transpose(Image.open(cropped_image_path)))
            cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
            cropped_image_resized = cv2.resize(cropped_image, (cropped_img_size, cropped_img_size))
            
            ref_keypoints, ref_descriptors = sift.detectAndCompute(cropped_image_resized, None)
            
            if ref_descriptors is not None:
                update_descriptor_dict_labels_known(reference_data_descriptors, a_label, a_serial, np.float32(ref_descriptors))
                metadata_table.loc[idx, 'descriptors_size'] = str(ref_descriptors.shape)
        else:
            print("Processed Image Not Found: '{}'".format(cropped_image_path))
    
    return metadata_table, reference_data_descriptors 

def get_sift_discriptor_based_on_saved_images_labels_NOT_known(metadata_table, metadata_filepath, ouptut_processed_img_dir):
    
    sift = cv2.SIFT_create(nfeatures=n_features, edgeThreshold=5)
    query_data_descriptors = {}
    
    for idx, row in tqdm(metadata_table.iterrows(), total=metadata_table.shape[0]):
        
        if idx%100 == 0:
            metadata_table.to_csv(metadata_filepath, index=False)
        
        # Load vision processed images
        a_giraffe_photo_path = row['path_relative_to_root']
        
        parts = a_giraffe_photo_path.rsplit(".", 1)
        img_filename = os.path.basename(a_giraffe_photo_path)
        cropped_image_path = os.path.join(ouptut_processed_img_dir, "zoomed_version", img_filename).replace("." + parts[1], "_cropped_torso_zoomed." + parts[1])
        
        if os.path.isfile(cropped_image_path):
        
            cropped_image = np.array(ImageOps.exif_transpose(Image.open(cropped_image_path)))
            cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
            cropped_image_resized = cv2.resize(cropped_image_gray, (cropped_img_size, cropped_img_size))

            query_keypoints, query_descriptors = sift.detectAndCompute(cropped_image_resized, None)
        
            if query_descriptors is not None:
                update_descriptor_dict_labels_NOT_known(query_data_descriptors, img_filename, np.float32(query_descriptors))
                metadata_table.loc[idx, 'descriptors_size'] = str(query_descriptors.shape)
        else:
            print("Processed Image Not Found: '{}'".format(cropped_image_path))
    
    return metadata_table, query_data_descriptors 

def run_a_check_on_reference_data(filename):
    # Load data
    with open(os.path.join(filename), 'rb') as f:
        ref_data_descriptor = pickle.load(f)
    
    # Check the number of items in the dictionary
    num_items = len(ref_data_descriptor)
    print(f"Total items in dictionary: {num_items}")
    
    for ref_label, (ref_descriptors_list, universal_ids_list) in tqdm(ref_data_descriptor.items()):
        print(f'ref_label {ref_label}: no of images {len(ref_descriptors_list)}, no of items in serial_number {len(universal_ids_list)} ')
        print(ref_label)
        print(type(ref_label))
        print(len(ref_descriptors_list))
        print(len(universal_ids_list))
        print(ref_descriptors_list[0].dtype)
        break

def run_a_check_on_query_data(filename):
    # Load data
    with open(os.path.join(filename), 'rb') as f:
        query_data_descriptor = pickle.load(f)
    
    # Check the number of items in the dictionary
    num_items = len(query_data_descriptor)
    print(f"Total items in dictionary: {num_items}")
    
    # Randomly select and display an item and its value if the dictionary is not empty
    if num_items > 0:
        random_item = random.choice(list(query_data_descriptor.items()))
        img_filename, descriptors = random_item
        print(f"Randomly selected item:\nFilename: {img_filename}\nValue Shape: {descriptors.shape}\nValue: {descriptors}")
    else:
        print("The dictionary is empty.")

def main(partition):
    
    # Call the profiling function
    profiler = cProfile.Profile()
    profiler.enable()
    
    # set up directories
    root_dir, processed_img_dir = load_data_dirs()
    
    # Set up logging files
    log_file_std_output, log_file_err_output = log_to_file(root_dir, 'descriptors_generation')
    
    # set up filenames
    data_descriptor_filename = 'giraffes_' + partition + '_descriptors.pkl'
    data_descriptor_filepath = os.path.join(root_dir, partition + '_dir', data_descriptor_filename)
    
    # load metadata
    metadata_filepath = os.path.join(root_dir, partition + '_dir', 'metadata_' + partition + '.csv')
    metadata_table = load_metadata_file(metadata_filepath)
    
    # get sift descriptors
    if partition == 'reference':
        
        # run algorithm
        metadata_table, data_descriptors_dict = get_sift_discriptor_based_on_saved_images_labels_known(metadata_table, metadata_filepath, processed_img_dir)
        
        # save metadata
        metadata_table.to_csv(metadata_filepath, index=False)

        # save pklfile
        with open(os.path.join(root_dir, partition + '_dir', data_descriptor_filepath), 'wb') as f:
            pickle.dump(data_descriptors_dict, f)
            
        # run a check on saved data
        run_a_check_on_reference_data(data_descriptor_filepath)
        
    else:
        
        # run algorithm
        metadata_table, data_descriptors_dict = get_sift_discriptor_based_on_saved_images_labels_NOT_known(metadata_table, metadata_filepath, processed_img_dir)
        
        # save metadata
        metadata_table.to_csv(metadata_filepath, index=False)

        # save pklfile
        with open(os.path.join(root_dir, partition + '_dir', data_descriptor_filepath), 'wb') as f:
            pickle.dump(data_descriptors_dict, f)
            
        # run a check on saved data
        run_a_check_on_query_data(data_descriptor_filepath)
        
    # Disabling the profiling function
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    
    # Print memory usage
    print_memory_usage()
    
    # Restore stdout 
    restore_stdout(log_file_std_output, log_file_err_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create image descriptors for giraffes')
    parser.add_argument('--partition', type=str, help='Partition to use vision model on: Partition to use vision model on: query or reference')
    args = parser.parse_args()
    main(args.partition)