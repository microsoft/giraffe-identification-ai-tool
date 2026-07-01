# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
from PIL import Image
import streamlit as st
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_matching import pipeline_code_relative_dir, demo_images
from utils.helpers_matching import get_img_paths_from_a_folder

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
    
st.title("Create Query Table from Elephant Image Directory")

pipeline_code_dir = os.path.join(str(Path(__file__).resolve().parent.parent), str(pipeline_code_relative_dir))


def process_directory(relative_to_root_image_dir):
    directory_path = os.path.join(st.session_state.root_dir, relative_to_root_image_dir)
    if os.path.exists(directory_path):
        st.write(f"Full Directory: {directory_path}")
        saved_file_path = get_img_paths_from_a_folder(st.session_state.root_dir, relative_to_root_image_dir)
        st.success(f"Corresponding metadata table created at:  {saved_file_path}")
    else:
        st.error(f"Directory does not exist: {directory_path}")
        
def main_get_a_dir_create_metadata_table():

    # Add a text input widget
    directory_input = st.text_input("Enter image directory path:", "")

    # Button to trigger the function
    if st.button("Process Directory"):
        if directory_input:
            process_directory(directory_input)
        else:
            st.error("Please enter a valid directory path.")
    
def display_image(image_path=None, default_image_dir=demo_images[8]):
    # If no image path is provided, display the default image
    if not image_path:
        image_path = default_image_dir
    
    # Check if the file exists
    if os.path.exists(image_path):
        
        # Open the image
        img = Image.open(image_path)

        # Resize while keeping aspect ratio
        max_width = 300  # Adjust this based on Streamlit column size
        w_percent = max_width / float(img.size[0])
        h_size = int(float(img.size[1]) * w_percent)
        resized_img = img.resize((max_width, h_size), Image.Resampling.LANCZOS)

        # Display in Streamlit
        st.image(resized_img, caption="Sample Query Table")

    else:
        # Display message if file doesn't exist
        st.error("The file does not exist. Please check the path and try again.")


input_col, info_col = st.columns([1, 2])

with info_col:

    st.markdown("""
    <style>
    .small-font {
        font-size:14px;
    }
    </style>

    <div class="small-font">
    
    Upload new images to your storage account, or keep existing images in any directory within the storage. Ensure that the image filenames and their paths (relative to the root directory) are available for processing. Use this tab to create a table with a required column for image paths relative to the root directory, based on the directory containing your images. Additionally, you can include an optional column for labels used in testing and accuracy tracking. This table will be updated throughout the process to track and monitor results.
    </div>
    """, unsafe_allow_html=True)
    
    display_image()

with input_col:
    main_get_a_dir_create_metadata_table()