# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
import cv2
import pstats
import cProfile
from PIL import Image
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers_matching import print_memory_usage, log_to_file, restore_stdout, load_data_dirs
from configs.config_matching import cropped_img_size
from utils.utils_matching import ProcessGiraffe
from configs.config_vision import giraffe_segmentation_model_dir, giraffe_segmentation_model_yaml_dir
from configs.config_vision import torso_detection_model_dir, torso_detection_model_yaml_dir, input_image_dir
from utils.utils_vision import insert_subdir_with_suffix


def get_all_image_paths(output_image_dir, extensions=None):
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.tiff'}

    image_paths = []

    for root, _, files in os.walk(output_image_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                image_paths.append(os.path.join(root, file))

    return image_paths

def make_giffy(input_image_list, output_dir):

    frames = [Image.open(image) for image in input_image_list]

    # Save as GIF
    frames[0].save(
        os.path.join(output_dir, 'output_giffy.gif'),
        save_all=True,
        append_images=frames[1:],
        duration=1500,  # Duration for each frame in milliseconds
        loop=0         # Loop forever
    )

def segmentation_model(root_dir, image_path, output_image_dir):
    
    # Giraffe Segmentation
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(os.path.join(root_dir, giraffe_segmentation_model_yaml_dir))
    cfg.MODEL.WEIGHTS = os.path.join(root_dir, giraffe_segmentation_model_dir)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    predictor = DefaultPredictor(cfg)
    
    # Make a name for new processed image file
    suffix='_full_giraffe'
    image_path_to_save = insert_subdir_with_suffix(image_path, output_image_dir, suffix)
    
    return predictor, image_path_to_save
    
def torso_detection_model(root_dir, image_path, output_image_dir):
    
    # Torso Detection
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(os.path.join(root_dir, torso_detection_model_yaml_dir))
    cfg.MODEL.WEIGHTS = os.path.join(root_dir, torso_detection_model_dir)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
    predictor = DefaultPredictor(cfg)
    
    # Make a name for new processed image file
    suffix='_torso'
    image_path_to_save = insert_subdir_with_suffix(image_path, output_image_dir, suffix)
    
    return predictor, image_path_to_save

def visualize_and_save(predictor, image_path, image_path_to_save, specific_instance_arg=None):
    im = cv2.imread(image_path)
    outputs = predictor(im) 
    v = Visualizer(im[:, :, ::-1])
    
    if specific_instance_arg:

        out = v.draw_instance_predictions(outputs["instances"][int(specific_instance_arg)].to("cpu"))
        print(outputs["instances"].pred_boxes[:])
    else:
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))


    plt.imshow(out.get_image())
    plt.show()
    cv2.imwrite(image_path_to_save, out.get_image()[:, :, ::-1])

def main(image_path, output_image_dir, mode):
    
    giraffe_predictor, giraffe_image_path_to_save = segmentation_model(image_path)
    torso_predictor, torso_image_path_to_save = torso_detection_model(image_path)

    if mode == 'segmentation':
        visualize_and_save(giraffe_predictor, image_path, giraffe_image_path_to_save)

    elif mode == 'torso_detection':
        visualize_and_save(torso_predictor, image_path, torso_image_path_to_save)
    
    else:
        a_serial_no = 'NA'
        a_label = 'NA'
        image_filename = os.path.basename(image_path)
        image_input_dir = os.path.dirname(image_path)
        _, _, _ = ProcessGiraffe(image_filename, a_serial_no, a_label, giraffe_predictor, torso_predictor, cropped_img_size, image_input_dir, output_image_dir, False)[0]

if __name__ == "__main__":
    
    # Call the profiling function
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Set up directories
    root_dir, _ = load_data_dirs()
    input_image_dir_full = os.path.join(root_dir, input_image_dir)
    output_image_dir = os.path.join(input_image_dir_full, 'images_outputs')
    os.makedirs(output_image_dir, exist_ok=True)
    
    # Set up logging files
    log_file_std_output, log_file_err_output = log_to_file(output_image_dir, 'vision_models_predictions', subdir='')        
    
    # Possible options for predictions
    modes = ['segmentation', 'torso_detection', 'combined']
    selected_option = modes[2]
    make_giffy_flag = False

    # Run model on selected images
    image_list = [img_item for img_item in os.listdir(input_image_dir) if img_item.split('.')[-1] in ['JPG']]
    for image_path in image_list:
        print(f'Processing image {image_path}')
        main(image_path, output_image_dir, selected_option)
    
    # Create a giffy based on some or all of the processed images if needed
    if make_giffy_flag:
        all_images_list = get_all_image_paths(output_image_dir)
        make_giffy(all_images_list, output_image_dir)
    
    # Print memory usage
    print_memory_usage()
    
    # Disabling the profiling function
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    
    # Restore stdout 
    restore_stdout(log_file_std_output, log_file_err_output)
    