# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
container_name = os.getenv("container_name")
data_root_abs_path = os.getenv("data_root_abs_path")

# Configs for sharding
min_shard_size, max_shard_size = 7000, 15000

# Some options to use pipeline for validation purposes
inference_mode = 'off' # 'on' or 'off' to adjust the starting new ID to avoid conflicts in catalog in validation mode
auto_accept_model_matching_results = True # True or False to accept model matching results automatically in validation mode

# Set up matching parameters
faiss_distance_cutoff = 0.062
faiss_mode_cutoff = 4
faiss_distance_cutoff_re_id = np.inf
faiss_mode_cutoff_re_id = 5
num_recommended_ids = 3
formatted_string_for_setup = f"distance_{faiss_distance_cutoff:.3f}_mode_{faiss_mode_cutoff}_distance_re_id_{faiss_distance_cutoff_re_id if faiss_distance_cutoff_re_id == np.inf else faiss_distance_cutoff_re_id:.3f}_mode_re_id_{faiss_mode_cutoff_re_id}"

# Set up sift parameters
cropped_img_size = 512
n_features = 1500

# Define ground truth column name if available
gt_keyname_col = 'AID2021'

# Define directory for faiss index if want to reuse an existing one
faiss_index_dir = os.path.join(data_root_abs_path, container_name, 'faiss_index')

# Options to use a pre-exising indx or start from scratch for partitioning new giraffes
partitioning_initialization = 0 #[0,1,2]

# Define pipeline dir to run codes in user interface
pipeline_code_relative_dir = 'pipeline'

# Define some image dirs for user interface
readme_ui_file = '../docs/README_UI.md'
image_dir = str(Path(__file__).resolve().parent.parent)
demo_images_files = {
    0: 'infographic/sift.png',
    1: 'infographic/all_images.png',
    2: 'infographic/header_giraffe.png',
    3: 'infographic/ms-ai-for-good-lab.png',
    4: 'infographic/re_id.png',
    5: 'infographic/identify_unknowns.png',
    6: 'infographic/update_catalogue.jpg',
    7: 'infographic/GiraffesLakeshore1.jpg',
    8: 'infographic/sample_query_data.png',
    9: 'infographic/accuracy.png',
    10: 'infographic/Infographic.jpg'}
demo_images = {k: os.path.join(image_dir, v) for k, v in demo_images_files.items()}