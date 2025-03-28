# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
import streamlit as st

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from configs.config_matching import demo_images, readme_ui_file
from utils.utils_files import read_file

# Load styles
st.html(f'<style>{read_file(os.path.join(parent_dir, "static/styles/header.css"))}</style>')
st.html(f'<style>{read_file(os.path.join(parent_dir, "static/styles/markdown.css"))}</style>')

# Display the header
st.html(read_file(os.path.join(parent_dir,'static/templates/header.html')))

# Display the image separately to ensure it loads correctly
st.divider()
st.image(demo_images[10], caption="AI Workflow Visualization", use_container_width=False)

# Read the contents of the README file
script_dir = os.path.dirname(os.path.abspath(__file__))
readme_path = os.path.join(script_dir, readme_ui_file)
with open(readme_path, "r") as file:
    readme_content = file.read()

# Display the contents of the README file
st.markdown(readme_content)