# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import subprocess
import streamlit as st
from dotenv import load_dotenv
from user_authentication import login_ui, authorize_users 

load_dotenv()
container_name, storage_account_name, mount_type, app_id, data_root_abs_path = map(os.getenv, 
                ["container_name", "storage_account_name", "mount_type", "app_id", "data_root_abs_path"])

from utils.utils_files import read_file
from utils.helpers_matching import load_data_dirs

st.session_state.mounting_success = False
st.session_state.num_id_recomms = 3
    
def mount_data(container_name, storage_account_name, mount_type, app_id):
    
    mounted_dir = os.path.join(data_root_abs_path, container_name)

    # check if already mounted
    if os.path.exists(mounted_dir):
        if os.listdir(mounted_dir):
            st.session_state.mounting_success = True
            return

    # run mounting script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, 'mount_blob_gen2.sh')
    print(script_path)
    result = subprocess.run(
        ["sudo", "bash", script_path, container_name, storage_account_name, mount_type, app_id], 
        capture_output=True, text=True
    )
    
    # print the output of the script
    print(result.stderr)
    print(result.stdout)
    
    # Check if mounted successfully, keeep record to avoid mounting multiple times
    if os.path.exists(mounted_dir):
        if os.listdir(mounted_dir):
            print(f"{mounted_dir} is NOT empty.")
            st.session_state.mounting_success = True
        else:
            print(f"{mounted_dir} is empty.")    

def remove_directory(path):
    if os.path.exists(path):
        try:
            # Use sudo to remove the directory and all its contents
            result = subprocess.run(
                ["sudo", "rm", "-r", path],
                capture_output=True, text=True, check=True
            )
            print(f"Successfully removed directory: {path}")
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error removing directory {path}: {e.stderr}")
            return e.stderr

def remove_if_empty(path):
    # Check if the directory exists
    if os.path.exists(path):
        # Check if the directory is empty
        if not os.listdir(path):  # Returns an empty list if the directory is empty
            print(f"{path} is empty, removing it.")
            try:
                # Use sudo to remove the empty directory
                result = subprocess.run(
                    ["sudo", "rm", "-r", path],
                    capture_output=True, text=True, check=True
                )
                print(f"Successfully removed directory: {path}")
                return result.stdout
            except subprocess.CalledProcessError as e:
                print(f"Error removing directory {path}: {e.stderr}")
                return e.stderr
        else:
            print(f"{path} is not empty, not removing.")
    else:
        print(f"Directory {path} does not exist.")

def main():
    if data_root_abs_path == '/mnt/':
        # Mount data if not done already
        if not st.session_state.mounting_success:
            remove_directory(os.path.join(data_root_abs_path, "blobfusecache"))
            remove_if_empty(os.path.join(data_root_abs_path, container_name))
            mount_data(container_name, storage_account_name, mount_type, app_id)
    else:
        # Data is available in local directory
        st.session_state.mounting_success = True
    st.session_state.root_dir, st.session_state.processed_img_dir = load_data_dirs()

    if not st.session_state.get("authenticated", False):
        st.markdown("""
            <style>
                [data-testid="stSidebar"] {
                    display: none;
                }
            </style>
        """, unsafe_allow_html=True)
        login_ui()
        return

    # Load global styles
    st.html(f'<style>{read_file(os.path.join(os.path.dirname(__file__), "static/styles/styles.css"))}</style>')
    st.html(f'<style>{read_file(os.path.join(os.path.dirname(__file__), "static/styles/sidebar.css"))}</style>')
    st.html(f'<style>{read_file(os.path.join(os.path.dirname(__file__), "static/styles/fonts.css"))}</style>')

    # Build the app
    app = st.navigation([
        st.Page("st_pages/st_0_home.py", title="Home"),
        st.Page("st_pages/st_1_create_query_table.py", title="Create Query Table"),
        st.Page("st_pages/st_2_preprocess_images.py", title="Preprocess Images"),
        st.Page("st_pages/st_3_run_reidentification.py", title="Run Reidentification"),
        st.Page("st_pages/st_4_verify_reidentification.py", title="Verify Reidentification"),
        st.Page("st_pages/st_5_identify_unknown_individuals.py", title="Identify Unknown"),
        st.Page("st_pages/st_6_verify_new_identifications.py", title="Verify New IDs"),
        st.Page("st_pages/st_7_update_catalogue.py", title="Update Catalogue"),
        st.Page("st_pages/st_8_validate_based_on_ground_truth.py", title="Validate Ground Truth"),
        st.Page("st_pages/st_9_visualize_single_image.py", title="Visualize Image")
    ])
    app.run()
        

if __name__ == "__main__":
    if not authorize_users():
        print("Not authorizing")
        st.session_state["authenticated"] = True   

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False   
    main()