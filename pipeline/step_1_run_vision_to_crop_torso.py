# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
import torch
import pstats
import cProfile
import warnings
import argparse
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers_matching import print_memory_usage, log_to_file, restore_stdout
from utils.helpers_matching import load_data_dirs, load_metadata_file
from configs.config_matching import cropped_img_size
from utils.utils_matching import ProcessGiraffe

setup_logger()
warnings.filterwarnings("ignore", category=UserWarning, module='detectron2')  

def load_computer_vision_models(inputs_dir):
    
    model_dir = os.path.join(inputs_dir, 'models')
    instance_segmentation_weights_filepath = os.path.join(model_dir, "model_final_f10217_segmentation.pkl")    
    instance_segmentation_config_filepath = os.path.join(model_dir, "mask_rcnn_R_50_FPN_3x.yaml")
    torso_detection_weights_filepath = os.path.join(model_dir, "model_final_torso_detection.pth")
    torso_detection_config_filepath = os.path.join(model_dir, "config.yaml")
    
    # Check if model directory exists
    if not os.path.isdir(model_dir):
        print(f"Error: Model directory '{model_dir}' not found.")
        sys.exit(1)
    
    # Check each file existence
    required_files = [
        instance_segmentation_weights_filepath,
        instance_segmentation_config_filepath,
        torso_detection_weights_filepath,
        torso_detection_config_filepath
    ]
    
    for file_path in required_files:
        if not os.path.isfile(file_path):
            print(f"Error: Required model file '{file_path}' not found.")
            sys.exit(1)  # Exit if any file is missing

    # Proceed with configuration if all checks are passed
    print("All required model files are present.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load segmentation model
    cfg_seg = get_cfg()
    cfg_seg.set_new_allowed(True)
    cfg_seg.MODEL.DEVICE = device
    cfg_seg.merge_from_file(instance_segmentation_config_filepath)
    cfg_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg_seg.MODEL.WEIGHTS = instance_segmentation_weights_filepath
    giraffe_predictor = DefaultPredictor(cfg_seg)

    # Load torso detection model
    cfg_torso = get_cfg()
    cfg_torso.set_new_allowed(True)
    cfg_torso.merge_from_file(torso_detection_config_filepath)
    cfg_torso.MODEL.WEIGHTS = torso_detection_weights_filepath
    cfg_torso.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg_torso.MODEL.DEVICE = device
    torso_predictor = DefaultPredictor(cfg_torso)

    return giraffe_predictor, torso_predictor

def check_if_processed_image_exists(a_giraffe_photo_path, output_image_dir):
    
    parts = a_giraffe_photo_path.rsplit(".", 1)
    img_filename = os.path.basename(a_giraffe_photo_path)
    
    # find a new name
    cropped_torso_zoomed_dir = os.path.join(output_image_dir, "zoomed_version", img_filename).replace("." + parts[1], "_cropped_torso_zoomed." + parts[1])    
    cropped_torso_dir = os.path.join(output_image_dir, "original_size", img_filename).replace("." + parts[1], "_cropped_torso." + parts[1])
    
    return (os.path.isfile(cropped_torso_zoomed_dir) and os.path.isfile(cropped_torso_dir))

def run_vision(metadata_table, metadata_filepath, input_img_dir, processed_img_dir, giraffe_predictor, torso_predictor, cropped_img_size):

    for idx, row in tqdm(metadata_table.iterrows(), total=metadata_table.shape[0]):
        
        if idx%100 == 0:
            metadata_table.to_csv(metadata_filepath, index=False)
            
        # Run vision
        a_giraffe_photo_path = row['path_relative_to_root']
        a_serial_no = 'NA'
        a_label = 'NA'
        print(a_giraffe_photo_path)
        
        if os.path.exists(os.path.join(input_img_dir, a_giraffe_photo_path)):
            if not check_if_processed_image_exists(a_giraffe_photo_path, processed_img_dir):
                giraffe_img_obj, _, _ = ProcessGiraffe(a_giraffe_photo_path, a_serial_no, a_label, giraffe_predictor, torso_predictor, cropped_img_size, input_img_dir, processed_img_dir, False)[0]
            
                # Update the relevant columns using the identified index
                metadata_table.loc[idx, 'giraffes_count'] = giraffe_img_obj.giraffe_segment_counts
                metadata_table.loc[idx, 'detection_coverage'] = float(f"{giraffe_img_obj.torso_detection_coverage:.3g}")
                metadata_table.loc[idx, 'segmentation_coverage'] = float(f"{giraffe_img_obj.torso_segmentation_coverage:.3g}")
                metadata_table.loc[idx, 'combined_coverage'] = float(f"{giraffe_img_obj.get_torso_combined_coverage():.3g}")

                processed_image = giraffe_img_obj.image_center_torso_cropped_sq
                if (processed_image is not None) and processed_image.shape == (cropped_img_size, cropped_img_size, 3):
                    metadata_table.loc[idx, 'ai_found_torso'] = 'True'
                else:
                    metadata_table.loc[idx, 'ai_found_torso'] = 'False'
            else:
                metadata_table.loc[idx, 'ai_found_torso'] = 'existing_item'
        else:
            metadata_table.loc[idx, 'ai_found_torso'] = 'file_not_found'
            
    return metadata_table

def main(partition):
    
    # Call the profiling function
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Load data directory paths
    root_dir, processed_img_dir = load_data_dirs()
    input_img_dir = root_dir 

    # Set up logging files
    log_file_std_output, log_file_err_output = log_to_file(root_dir, 'vision_model')
            
    # Load metadata
    metadata_filepath = os.path.join(root_dir, partition + '_dir', 'metadata_' + partition + '.csv')
    metadata_table = load_metadata_file(metadata_filepath)
    
    # Load and run vision model
    giraffe_predictor, torso_predictor = load_computer_vision_models(root_dir)
    metadata_table = run_vision(metadata_table, metadata_filepath, input_img_dir, processed_img_dir, giraffe_predictor, torso_predictor, cropped_img_size)
    
    # Save metadata
    metadata_table.to_csv(metadata_filepath, index=False)
    
    # Disabling the profiling function
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    
    # Print memory usage
    print_memory_usage()
    
    # Restore stdout 
    restore_stdout(log_file_std_output, log_file_err_output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run vision model to crop giraffe torso')
    parser.add_argument('--partition', type=str, default='query', help='Partition to use vision model on for preprocessing images: query or reference')
    args = parser.parse_args()
    main(args.partition)
