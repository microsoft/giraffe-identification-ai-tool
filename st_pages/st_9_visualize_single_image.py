# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
import pandas as pd
from PIL import Image
import streamlit as st
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_matching import pipeline_code_relative_dir, demo_images

from user_authentication import login_ui, authorize_users

# Check authentication status and redirect to login if not authenticated
if authorize_users() and not st.session_state.get("authenticated", False):
    # Add CSS to hide sidebar
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)
    login_ui()
    st.stop()

st.title('Visualize Single Elephant Image from Directory')

pipeline_code_dir = os.path.join(str(Path(__file__).resolve().parent.parent), str(pipeline_code_relative_dir))

def _lookup_viewpoint(image_path, metadata_filepath):
    """Returns the viewpoint tag for the given image path from step_1 metadata, if available."""
    if not image_path or not os.path.isfile(metadata_filepath):
        return None
    try:
        df = pd.read_csv(metadata_filepath)
        if 'viewpoint' not in df.columns or 'path_relative_to_root' not in df.columns:
            return None
        basename = os.path.basename(image_path)
        match = df[df['path_relative_to_root'].apply(os.path.basename) == basename]
        if not match.empty:
            return match['viewpoint'].iloc[0]
    except Exception:
        pass
    return None

def display_image(image_path=None, default_image_dir=demo_images[7], metadata_filepath=None):
    if not image_path:
        image_path = default_image_dir

    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption="", width=400)

        # Show viewpoint tag if step_1 metadata is available
        if metadata_filepath:
            vp = _lookup_viewpoint(image_path, metadata_filepath)
            if vp is not None:
                st.info(f"Viewpoint: **{vp}**")
    else:
        st.error("The file does not exist. Please check the path and try again.")


# Default image will be shown first
display_image()

# Optional: allow user to point at a metadata file to get viewpoint info
metadata_filepath_input = st.text_input(
    "Enter path to metadata CSV (optional, to display viewpoint tag):", ""
)

# Input for image path from the user
image_path = st.text_input("Enter the path to the elephant image:")

# Button to trigger the image update
if st.button("Display Image"):
    meta_fp = metadata_filepath_input.strip() if metadata_filepath_input.strip() else None
    display_image(image_path, metadata_filepath=meta_fp)
