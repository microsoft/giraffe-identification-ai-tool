# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Elephant-specific matching config — see also configs/config_elephant.py

import os
from pathlib import Path
from dotenv import load_dotenv

from configs.config_elephant import MATCH_ACCEPT_THRESHOLD, NUM_RECOMMENDED_IDS, SHORTLIST_K, ID_COL, GT_COL

load_dotenv()
container_name      = os.getenv("container_name")
data_root_abs_path  = os.getenv("data_root_abs_path")

# Configs for sharding
min_shard_size, max_shard_size = 7000, 15000

# Pipeline execution modes
inference_mode                   = 'off'   # 'on' or 'off' to adjust starting new ID in validation mode
auto_accept_model_matching_results = True  # True or False to accept model results automatically in validation mode

# Formatted string for experiment bookkeeping
formatted_string_for_setup = f"fused_accept_{MATCH_ACCEPT_THRESHOLD}_shortlist_{SHORTLIST_K}"

# Define directory for FAISS index if reusing an existing one
faiss_index_dir = os.path.join(data_root_abs_path, container_name, 'faiss_index')

# Options to use a pre-existing index or start from scratch for partitioning new elephants
partitioning_initialization = 0  # [0, 1, 2]

# Define pipeline dir to run codes in user interface
pipeline_code_relative_dir = 'pipeline'

# Define some image dirs for user interface
readme_ui_file = '../docs/README_UI.md'
image_dir = str(Path(__file__).resolve().parent.parent)
demo_images_files = {
    0: 'infographic/all_images.png',
    1: 'infographic/all_images.png',
    2: 'infographic/header_elephant.png',
    3: 'infographic/ms-ai-for-good-lab.png',
    4: 're_id.png',
    5: 'infographic/identify_unknowns.png',
    6: 'infographic/update_catalogue.jpg',
    7: 'infographic/GiraffesLakeshore1.jpg',
    8: 'infographic/sample_query_data.png',
    9: 'infographic/accuracy.png',
    10: 'infographic/Infographic.jpg',
}
demo_images = {k: os.path.join(image_dir, v) for k, v in demo_images_files.items()}
