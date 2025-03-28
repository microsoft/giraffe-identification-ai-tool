# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

#!/usr/bin/bash

conda_env_full_path="/anaconda/envs/giraffe"
streamlit_app_file_path="/home/giraffe_ui_streamlit.py"

# Start the application
export PATH=/anaconda/condabin:$PATH

# Source the conda initialization script
source /anaconda/etc/profile.d/conda.sh

# Activate the conda environment
conda activate $conda_env_full_path

streamlit run $streamlit_app_file_path  --server.port 8088
