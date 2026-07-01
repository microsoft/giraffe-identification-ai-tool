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
    
st.title('Update Elephant Reference Catalogue based on Query Data')

pipeline_code_dir = os.path.join(str(Path(__file__).resolve().parent.parent), str(pipeline_code_relative_dir))

def run_bash_script(code_directory, pycode_name):
    result = subprocess.run(
        ["bash", "./setup_pipeline.sh", os.path.join(code_directory, pycode_name)], 
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
    # Terminate the tmux session
    subprocess.run(["tmux", "kill-session", "-t", SESSION_NAME])

def check_status():
    SESSION_NAME = "streamlit_script"
    # Check if the tmux session is running
    result = subprocess.run(["tmux", "has-session", "-t", SESSION_NAME],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0  # 0 means the session exists

def main_run_buttons(pipeline_code_dir, pycode_name, image_path):
    cols_buttons, col_images = st.columns([1, 2])  # Left column (buttons), right column (image)

    # Initialize session state variables if not already set
    if 'run_button_disabled' not in st.session_state:
        st.session_state.run_button_disabled = False  # "Run" button enabled by default
    if 'terminate_button_disabled' not in st.session_state:
        st.session_state.terminate_button_disabled = True  # "Terminate" button disabled by default
    if 'run_script' not in st.session_state:
        st.session_state.run_script = False  # Control variable to ensure one-time execution

    # If an experiment is already running, update button states
    if check_status():
        st.session_state.run_button_disabled = True
        st.session_state.terminate_button_disabled = False

    # BUTTONS (Left Column)
    with cols_buttons:
        if st.button("▶️ Start Experiment", disabled=st.session_state.run_button_disabled):
            # Set flags and disable/enable buttons
            st.session_state.run_button_disabled = True
            st.session_state.terminate_button_disabled = False
            st.session_state.run_script = True  # Flag to execute the script

            # Refresh UI to apply button state changes
            st.rerun()

        if st.button("⏹️ Stop Experiment", disabled=st.session_state.terminate_button_disabled):
            # Reset button states
            st.session_state.run_button_disabled = False
            st.session_state.terminate_button_disabled = True
            st.session_state.run_script = False  # Ensure no script runs

            # Clear outputs and show termination message
            st.empty()
            terminate_script()  # Terminate the script
            st.success("Experiment terminated successfully.")

            # Refresh UI to apply changes
            st.rerun()

        # Execute the script if run_script flag is set
        if st.session_state.run_script:
            # Reset the flag to avoid multiple executions
            st.session_state.run_script = False
            st.empty()  # Clear previous outputs

            # Execute the script
            with st.spinner("Running script..."):
                stdout, stderr = run_bash_script(pipeline_code_dir, pycode_name)

    # IMAGE DISPLAY (Right Column)
    with col_images:
        st.markdown("""
        <style>
        .small-font {
            font-size:14px;
        }
        </style>

        <div class="small-font">
        
        **Update Reference Catalogue:** All re-identified elephants, along with newly partitioned individuals reviewed and approved by a human expert, are assigned identification labels.
        These labels are aligned with the existing labels in the reference dataset.
        After labeling, the reference catalogue is updated accordingly.
        This process is repeated for each survey round, ensuring the catalogue remains accurate and up-to-date with new data and observations.
        </div>
        """, unsafe_allow_html=True)
        if os.path.isfile(image_path):
            with open(image_path, "rb") as file:
                gif_bytes = file.read()
                st.image(gif_bytes, caption='')
        else:
            st.error("Image file not found.")


# Add a styled horizontal line below the radio button section
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

# Run the main function
pycode_name = 'step_4_partition_new_items.py'
main_run_buttons(pipeline_code_dir, pycode_name, demo_images[6])