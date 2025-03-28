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

st.title('Visualize Single Image from Directory')

pipeline_code_dir = os.path.join(str(Path(__file__).resolve().parent.parent), str(pipeline_code_relative_dir))

def display_image(image_path=None, default_image_dir=demo_images[7]):
    # If no image path is provided, display the default image
    if not image_path:
        image_path = default_image_dir
    
    # Check if the file exists
    if os.path.exists(image_path):
        # Open and display the image if it exists
        image = Image.open(image_path)
        st.image(image, caption="", width=400)
    else:
        # Display message if file doesn't exist
        st.error("The file does not exist. Please check the path and try again.")


# Default image will be shown first
display_image()

# Input for image path from the user
image_path = st.text_input("Enter the path to the image:")

# Button to trigger the image update
if st.button("Display Image"):
    display_image(image_path)