# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Elephant detection config — MegaDetector replaces Detectron2

# Experiment identifier for tracking runs
experiment_keyname = 'elephant_wildfusion_v1'

# Inference: single image that should be in validation set for evaluation
specific_image_path = None

# Prediction: input image directory
input_image_dir = 'object_detection_output_dir/images_inputs'

# Detector backend
DETECTOR_BACKEND     = "megadetector"
DETECTOR_CONF        = 0.5
MEGADETECTOR_MODEL_DIR = "models/"
