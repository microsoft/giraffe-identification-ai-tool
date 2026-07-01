# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
import subprocess
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

pipeline_code_dir = os.path.join(str(Path(__file__).resolve().parent.parent), str(pipeline_code_relative_dir))

st.title('Preprocessing Elephant Images')
st.html("""
    <style>
        .stMainBlockContainer {
            width: 100%;
            max-width: 100%;
            padding-left: 80px;
            padding-right: 80px;
            padding-top: 96px;
            padding-bottom: 160px;
        }
    </style>
""")


st.markdown("""
<style>
.small-font {
    font-size:14px;
}
</style>

<div class="small-font">

Select a dataset type and analysis method to detect elephants and generate deep embeddings for re-identification. The elephant's skin texture and body shape act as a unique fingerprint for matching.

**1. Dataset Selection**
Reference Catalogue focuses on annotated elephants used for comparison.
Query Data focuses on new images matched against the reference dataset.

**2. Elephant Detection Crop**
MegaDetector detects the animal in the scene and produces a tight whole-body crop.

**3. Image Embeddings Extraction**
MiewID and MegaDescriptor extract global deep descriptors from the cropped elephant images.
LightGlue provides local keypoint matching for fine-grained verification.

</div>
""", unsafe_allow_html=True)

def run_bash_script(code_directory, pycode_name, additional_args):
    result = subprocess.run(
        ["bash", "./setup_pipeline.sh", os.path.join(code_directory, pycode_name)] + additional_args,
        capture_output=True, text=True
    )

    # Display the output of the script in the Streamlit app
    st.subheader("Script Output:")
    st.text(result.stdout)

    if result.stderr:
        st.subheader("Script Error:")
        st.text(result.stderr)

    # Check if the execution was successful
    if result.returncode == 0:
        print("Bash script executed successfully.")
        return result.stdout, result.stderr
    else:
        print(f"Bash script execution failed with return code {result.returncode}.")
        print("Standard Error Output:")
        print(result.stderr)
        return result.stdout, result.stderr

def terminate_script():
    SESSION_NAME = "streamlit_script"
    subprocess.run(["tmux", "kill-session", "-t", SESSION_NAME])

def check_status():
    SESSION_NAME = "streamlit_script"
    result = subprocess.run(["tmux", "has-session", "-t", SESSION_NAME],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0  # 0 means the session exists

def main_run_buttons(pipeline_code_dir, pycode_name, image_path, additional_args):
    cols_buttons, col_images = st.columns([1, 2])  # Left column (buttons), right column (image)

    # Initialize session state variables if not already set
    if 'run_button_disabled' not in st.session_state:
        st.session_state.run_button_disabled = False
    if 'terminate_button_disabled' not in st.session_state:
        st.session_state.terminate_button_disabled = True
    if 'run_script' not in st.session_state:
        st.session_state.run_script = False

    # If an experiment is already running, update button states
    if check_status():
        st.session_state.run_button_disabled = True
        st.session_state.terminate_button_disabled = False

    # BUTTONS (Left Column)
    with cols_buttons:
        if st.button("▶️ Start Experiment", disabled=st.session_state.run_button_disabled):
            st.session_state.run_button_disabled = True
            st.session_state.terminate_button_disabled = False
            st.session_state.run_script = True
            st.rerun()

        if st.button("⏹️ Stop Experiment", disabled=st.session_state.terminate_button_disabled):
            st.session_state.run_button_disabled = False
            st.session_state.terminate_button_disabled = True
            st.session_state.run_script = False
            st.empty()
            terminate_script()
            st.success("Experiment terminated successfully.")
            st.rerun()

        # Execute the script if run_script flag is set
        if st.session_state.run_script:
            st.session_state.run_script = False
            st.empty()
            with st.spinner("Running script..."):
                stdout, stderr = run_bash_script(pipeline_code_dir, pycode_name, additional_args)

    # IMAGE DISPLAY (Right Column)
    with col_images:
        if os.path.isfile(image_path):
            with open(image_path, "rb") as file:
                gif_bytes = file.read()
                st.image(gif_bytes, caption='')
        else:
            st.error("Image file not found.")


def display_viewpoint_if_available(metadata_filepath):
    """Show a compact viewpoint summary for processed images if step_1 metadata exists."""
    if not os.path.isfile(metadata_filepath):
        return

    try:
        import pandas as pd
        df = pd.read_csv(metadata_filepath)
        if 'viewpoint' in df.columns:
            st.markdown("**Viewpoint distribution in processed batch:**")
            vp_counts = df['viewpoint'].value_counts().reset_index()
            vp_counts.columns = ['viewpoint', 'count']
            st.dataframe(vp_counts, use_container_width=False)
    except Exception:
        pass  # Non-critical display; silently skip on any read error


# Display the pill button selection widget for data type
st.markdown(f'<div class="radio-font-size">Select Data Batch:</div>', unsafe_allow_html=True)
data_type = st.pills("", ['Reference Catalogue', 'Query Data'], key="data_type_toggle")
additional_args = []
if data_type == 'Reference Catalogue':
    additional_args = ["--partition reference"]
elif data_type == 'Query Data':
    additional_args = ["--partition query"]

# Add a styled horizontal line below the toggle
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

if len(additional_args) != 0:

    # Display the dropdown menu with increased font size
    st.markdown(f'<div class="radio-font-size">Select Analysis:</div>', unsafe_allow_html=True)
    selected_option = st.selectbox("", ['Detect & Crop Elephants',
                                        'Extract Image Embeddings'],
                                key="dropdown_options")

    # Add a styled horizontal line below the toggle
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    if selected_option == 'Detect & Crop Elephants':
        pycode_name = 'step_1_run_detection_to_crop.py'
        main_run_buttons(pipeline_code_dir, pycode_name, demo_images[1], additional_args)

        # Show viewpoint tags produced by step_1 for the selected partition
        partition_key = 'reference' if data_type == 'Reference Catalogue' else 'query'
        from dotenv import load_dotenv
        load_dotenv()
        data_root = os.getenv("data_root_abs_path", "")
        container = os.getenv("container_name", "")
        metadata_fp = os.path.join(data_root, container, f"{partition_key}_dir", f"metadata_{partition_key}.csv")
        display_viewpoint_if_available(metadata_fp)

    elif selected_option == 'Extract Image Embeddings':
        pycode_name = 'step_2_create_embeddings.py'
        main_run_buttons(pipeline_code_dir, pycode_name, demo_images[0], additional_args)
