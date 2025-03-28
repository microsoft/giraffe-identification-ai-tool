# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import cv2
import sys
import torch
import random
import pstats
import cProfile
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_vision import cv2_annotations_dir, giraffe_count_coverage_df_dir, metadata_path_processed
from configs.config_vision import giraffe_count, data_random_sample, small_dataset_serials, experiment_keyname
from utils.helpers_matching import print_memory_usage, log_to_file, restore_stdout, load_data_dirs
from utils.utils_vision import load_metadata, merge_metadata_count_giraffes, choose_metadata_subset, build_data_splits
from utils.utils_vision import set_up_data_splits, get_dataset_dicts
from configs.config_vision import specific_image_path

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(output_dir, cfg_torso, predictor):
    os.makedirs(os.path.join(output_dir, 'outputs_evaluation'), exist_ok=True)
    evaluator = COCOEvaluator("giraffe_torso_val", ("bbox",), False, output_dir=os.path.join(output_dir, 'outputs_evaluation'))
    val_loader = build_detection_test_loader(cfg_torso, "giraffe_torso_val")
    inference_on_dataset(predictor.model, val_loader, evaluator)
    
def inference_model(output_data_dir):
    
    # Load torso detection model
    cfg_torso = get_cfg()
    cfg_torso.OUTPUT_DIR = output_data_dir
    cfg_torso.merge_from_file(os.path.join(cfg_torso.OUTPUT_DIR, 'config.yaml'))
    cfg_torso.MODEL.WEIGHTS = os.path.join(cfg_torso.OUTPUT_DIR, 'model_final.pth')
    cfg_torso.set_new_allowed(True)
    cfg_torso.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg_torso.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg_torso)
   
    return cfg_torso, predictor

def visualize_single_image(dataset_dicts, predictor, giraffe_torso_metadata, output_data_dir):
    
    for d in random.sample(dataset_dicts, 1):
        
        # image info
        a_giraffe_photo_path = d["file_name"]
        output_file_path = os.path.join(output_data_dir, 'giraffe_' + a_giraffe_photo_path.replace('/', '___'))
        print(a_giraffe_photo_path)

        # load image, run inference and visualize
        im = cv2.imread(a_giraffe_photo_path)
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=giraffe_torso_metadata, scale=0.5)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(output_file_path, out.get_image()[:, :, ::-1])

def main():
    
    # Call the profiling function
    profiler = cProfile.Profile()
    profiler.enable()
    
    # set up directories
    root_dir, _ = load_data_dirs()
    root_output_dir = os.path.join(root_dir, 'object_detection_output_dir')    
    output_dir = os.path.join(root_output_dir, 'models_' + experiment_keyname)
    os.makedirs(output_dir, exist_ok=True)
    
    cv2_annotations_dir_full = os.path.join(root_dir, cv2_annotations_dir)
    metadata_path_processed_full = os.path.join(root_dir, metadata_path_processed)
    giraffe_count_coverage_df_dir_full = os.path.join(root_dir, giraffe_count_coverage_df_dir)
    
    # Set up logging files
    log_file_std_output, log_file_err_output = log_to_file(output_dir, 'torso_model_inference', subdir='')    
    
    # Load metadata and select a subset if needed
    metadata_df = load_metadata(metadata_path_processed_full)
    metadata_df = merge_metadata_count_giraffes(metadata_df, giraffe_count_coverage_df_dir_full)
    metadata_df = choose_metadata_subset(metadata_df, giraffe_count, data_random_sample, small_dataset_serials)
    train_df, val_df, test_df = build_data_splits(metadata_df)

    # Set up data splits in catalog
    DatasetCatalog, MetadataCatalog = set_up_data_splits(train_df, val_df, test_df, cv2_annotations_dir_full, root_dir)


    # Run inference and evaluate model 
    cfg_torso, predictor = inference_model(output_dir)
    evaluate(output_dir, cfg_torso, predictor)
    
    # # Select a specific single image from validation set for vizualization
    if specific_image_path is not None:
        val_df = val_df[val_df['path'] == specific_image_path]
    
    # Visualize on single image randomly or selected by user
    dataset_dicts = get_dataset_dicts(val_df, cv2_annotations_dir_full, root_dir)
    visualize_single_image(dataset_dicts, predictor, MetadataCatalog.get("giraffe_torso_val"), output_dir)
    
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