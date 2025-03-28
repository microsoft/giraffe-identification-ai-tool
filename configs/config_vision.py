# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Training: Set up experiment keyname
experiment_keyname = 'lr0004_default_aug_cosine_single_grf'

# Training: Data filtering setup
giraffe_count = 1
data_random_sample = True
small_dataset_serials = []

# Training: Data and initial model dirs
init_model_dir = 'models/model_initial_torso_detection.pkl'
init_model_config_yaml_dir = 'models/faster_rcnn_R_101_FPN_3x.yaml'
metadata_path_processed = 'processed_metadata/data_splits/metadata_with_splits_with_cropped_img_with_image_dims_with_path_relative_to_root.csv'
giraffe_count_coverage_df_dir = 'processed_metadata/data_splits/giraffe_counts_coverage_from_models_lr0004_default_aug_cosine.csv'
cv2_annotations_dir = 'processed_metadata/torso_annotations/cv2_annotations_rotation_fixed'

# Inference: Single image that should be in validation set for evaluation
specific_image_path = None #'2013_9_MAY_rawpics/2013May.IMG_3553.JPG'

# Prediction: Trained model and input image dirs
input_image_dir = 'object_detection_output_dir/images_inputs'
giraffe_segmentation_model_dir = 'models/model_final_f10217_segmentation.pkl' 
giraffe_segmentation_model_yaml_dir = 'models/mask_rcnn_R_50_FPN_3x.yaml'
torso_detection_model_dir = 'models/model_final_torso_detection.pth'
torso_detection_model_yaml_dir = 'models/config.yaml'


