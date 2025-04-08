# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_matching import pipeline_code_relative_dir, demo_images
from utils.helpers_matching import load_metadata_file, load_data_dirs

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
    
st.title('Review Accuracy and Incorrect Results based on Ground Truth')
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


st.markdown("""This page offers a detailed breakdown of accuracy results based on ground truth data for query images. 
            Additionally, you can visualize and analyze falsely matched results to gain deeper insights into identification errors and improve performance.""")


pipeline_code_dir = os.path.join(str(Path(__file__).resolve().parent.parent), str(pipeline_code_relative_dir))
st.session_state.root_dir, st.session_state.processed_img_dir = load_data_dirs()

def find_algorithm_mistakes(query_metadata):
    
    required_cols = ['AID2021', 'matched_label_1', 'out_of_sample', 'matching_status']
    known_matched_false, fp_table, fn_table = None, None, None
    
    if all(col in query_metadata.columns for col in required_cols):
    
        # items known but matched incorrectly  
        known_matched_false = query_metadata[(query_metadata['matching_status'] == 'matched') & (query_metadata['AID2021'] != query_metadata['matched_label_1'])]

        # items not known but matched incorrectly
        fp_table = query_metadata[(query_metadata['out_of_sample']==True) & (query_metadata['matching_status']=='matched')]
        
        # items known but incorrectly not matched 
        fn_table  = query_metadata[(query_metadata['out_of_sample']==False) & (query_metadata['matching_status']=='not_matched')]
        
    else:
        st.write('query_metadata does not contain all required columns')

    return known_matched_false, fp_table, fn_table

def main_compute_matching_accuracy(pipeline_code_dir, image_path):
    col_buttons, col_info = st.columns([0.5, 1.5])
    
    with col_buttons:
        st.write("")
        # Button to trigger bash script execution
        if st.button("Compute Matching Accuracy Metrics"):
            with st.spinner("Running script..."):
                pycode_name = 'step_5_evaluate_matching_results.py'
                stdout, stderr = run_bash_script(pipeline_code_dir, pycode_name)
                acc_results_file = os.path.join(st.session_state.root_dir, 'query_dir', 'accuracy_results.csv')
                if os.path.isfile(acc_results_file):
                    st.write(pd.read_csv(acc_results_file))
    with col_info:    
        st.markdown("""
        ## Accuracy Metrics Overview  

        We categorize known individuals as the **positive class** and unknown individuals as the **negative class**. The evaluation considers two key aspects:  

        - **Reidentification Accuracy:** Measures how accurately known individuals are identified.  
        - **Partitioning Accuracy:** Assesses how well unknown individuals are grouped into distinct clusters.  

        To capture these aspects, we report the following metrics:  

        ### Class-Specific Metrics  
        These metrics evaluate the classification performance for known (**positive class**) and unknown (**negative class**) individuals:  

        - **Precision (Positive):** The proportion of correctly identified known individuals among all predicted known individuals.  
        - **Recall (Positive):** The proportion of correctly identified known individuals among all actual known individuals.  
        - **F1 Score (Positive):** The harmonic mean of precision and recall for known individuals.  
        - **Precision (Negative):** The proportion of correctly identified unknown individuals among all predicted unknown individuals.  
        - **Recall (Negative):** The proportion of correctly identified unknown individuals among all actual unknown individuals.  
        - **F1 Score (Negative):** The harmonic mean of precision and recall for unknown individuals.  

        ### Overall Performance Metrics  
        These metrics provide a broader evaluation of system accuracy:  

        - **Overall Accuracy:** The proportion of correctly classified known and unknown individuals across all query images.  
        - **Accuracy (Reidentified Items):** Measures how accurately known individuals are reidentified based on ground truth labels.  
        - **Adjusted Rand Index (Partitioning):** Evaluates the quality of partitioning for unknown individuals, comparing the predicted clusters with the ground truth.  

        These metrics offer a comprehensive assessment of identification accuracy and clustering effectiveness, ensuring robust evaluation of the system’s performance.
        """)

        # Display an image
        if not os.path.isfile(image_path):
            raise FileNotFoundError("Image file '{}' not found.".format(image_path))
        image = Image.open(image_path)
        st.image(image, caption='')

def run_bash_script(code_directory, pycode_name):
    result = subprocess.run(
        ["bash", "./setup_pipeline.sh", os.path.join(code_directory, pycode_name)], 
        capture_output=True, text=True
    )
            
    # Check if the execution was successful
    if result.returncode == 0:
        print("Bash script executed successfully.")
        return result.stdout, result.stderr
    else:
        print(f"Bash script execution failed with return code {result.returncode}.")
        print("Standard Error Output:")
        print(result.stderr)
        return result.stdout, result.stderr
    
def reset_state_vars():
    
    st.session_state.current_query_index = 0
    st.session_state.current_reference_index = 0
    st.session_state.reference_image_paths = []  
    st.session_state.recomms_rank_selected = 1
    st.session_state.current_query_not_matched_index = 0
    
    # related to visualization of falsely matched items
    st.session_state.curr_query_idx_known_matched_false = 0
    st.session_state.curr_ref_idx_known_matched_false = 0
    st.session_state.curr_ref_idx_gt_known_matched_false = 0
    st.session_state.ref_img_paths_known_matched_false = []
    st.session_state.ref_img_paths_gt_known_matched_false = []
    st.session_state.recomms_rank_selected_known_matched_false = 1
    
 
    # related to visualization of falsely unmatched items
    st.session_state.curr_query_idx_fn_table = 0
    st.session_state.curr_ref_idx_fn_table = 0
    st.session_state.curr_ref_idx_gt_fn_table = 0
    st.session_state.ref_img_paths_fn_table = []
    st.session_state.ref_img_paths_gt_fn_table = []
    st.session_state.recomms_rank_selected_fn_table = 1
    
    # related to visualization of falsely matched unknown items
    st.session_state.curr_query_idx_fp_table = 0
    st.session_state.curr_ref_idx_fp_table = 0
    st.session_state.curr_ref_idx_gt_fp_table = 0
    st.session_state.ref_img_paths_fp_table = []
    st.session_state.ref_img_paths_gt_fp_table = []
    st.session_state.recomms_rank_selected_fp_table = 1
   
def initialize_vizualization_project_validation():
    st.session_state.metadata = {}
    for partition in ['query', 'reference']:
        metadata_filepath = os.path.join(st.session_state.root_dir, partition + '_dir', 'metadata_' + partition + '.csv')
        st.session_state.metadata[partition] = load_metadata_file(metadata_filepath)
    
    # user can use this option to reload the query metadata and start fresh
    st.session_state.metadata_table = st.session_state.metadata['reference'].copy()
    st.session_state.matching_results_table = st.session_state.metadata['query'].copy()
    
def get_human_input_single(key_query):
    base_names = st.session_state.matching_results_table['path_relative_to_root'].apply(os.path.basename)
    mask = base_names == os.path.basename(key_query)

    if 'human_input' in st.session_state.matching_results_table.columns and len(st.session_state.matching_results_table.loc[mask, 'human_input'])!=0 and not st.session_state.matching_results_table.loc[mask, 'human_input'].isnull().all():
        return st.session_state.matching_results_table.loc[mask, 'human_input']
    else:
        return None
    
def get_corresponding_torso_image(image_path, subdir_name="original_size", suffix=""):
    
    # find path for _cropped_torso_zoomed
    parts = image_path.rsplit(".", 1)
    img_filename = os.path.basename(image_path)
    cropped_torso_zoomed_image_dir = os.path.join(st.session_state.processed_img_dir, subdir_name, img_filename).replace("." + parts[1], "_cropped_torso" + suffix + "." + parts[1])
            
    return cropped_torso_zoomed_image_dir

def display_left_query_images_matched_images(image_paths, captions):
    # Create a single column
    col = st.columns(1)[0]

    # Iterate through the images and display them in the single column
    for i in [1, 0]:
        image_path = image_paths[i]
        
        # Check if the image file exists
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file '{image_path}' not found.")
        
        # Open and handle EXIF rotation
        img = ImageOps.exif_transpose(Image.open(image_path))

        # Resize images based on index
        if i == 0:  # First image: Resized dynamically
            scale_factor = 0.15
            img_resized = img.resize(
                (int(img.width * scale_factor), int(img.height * scale_factor))
            )
            # Display image with caption in the single column
            col.image(img_resized, caption=captions[i], use_container_width=True)
            
        else:  # Second image: Fixed size
            img_resized = img.resize((256, 256))  # Adjusted to a more viewable size
            
            # Display image with caption in the single column
            col.image(img_resized, caption='Cropped Torso Image', use_container_width=True)
 
def display_right_reference_image(data, starting_idx, num_rows, num_cols):
    image_paths = data[starting_idx:]
    for i in range(num_rows):
        cols = st.columns(num_cols)
        for j, col in enumerate(cols):
            index = i * num_cols + j
            if index < len(image_paths):
                image_path = image_paths[index]
                
                # torso photo
                torso_img_path = get_corresponding_torso_image(image_path, "zoomed_version", "_zoomed")
                if not os.path.isfile(torso_img_path):
                    raise FileNotFoundError("Image file '{}' not found.".format(torso_img_path))
                torso_img = ImageOps.exif_transpose(Image.open(torso_img_path))
                torso_img = torso_img.resize((256, 256))
                col.image(torso_img, caption='Cropped Torso Image', use_container_width=True)
                
                # full original photo
                if not os.path.isfile(image_path):
                    raise FileNotFoundError("Image file '{}' not found.".format(image_path))
                actual_img = ImageOps.exif_transpose(Image.open(image_path))
                actual_img_resized = actual_img.resize((int(actual_img.width * 0.15), int(actual_img.height * 0.15)))
                actual_image_caption = os.path.basename(image_path)
                col.image(actual_img_resized, caption=f"Reference Image: {actual_image_caption}", use_container_width=True)

def display_custom_table(assigned_id, image_serial_matched, matched_dist, matched_mode, descriptors_size, recom_rank):
    table_html = f"""
    <style>
    .custom-table {{
        border-collapse: collapse;
        table-layout: auto;
        margin-top: 10px;
    }}
    .custom-table td {{
        border: 1px solid black;
        padding: 10px;
        text-align: center;
        font-size: 16px;
    }}
    .custom-table .header {{
        background-color: #f2f2f2;
        font-weight: bold;
    }}
    </style>
    <table class="custom-table">
        <tr>
            <td class="header">Rank</td>
            <td>{recom_rank}</td>
        </tr>
        <tr>
            <td class="header">Matched ID</td>
            <td>{assigned_id}</td>
        </tr>
        <tr>
            <td class="header">Matched Serial</td>
            <td>{image_serial_matched}</td>
        </tr>
        <tr>
            <td class="header">Matched Distance</td>
            <td>{matched_dist}</td>
        </tr>
        <tr>
            <td class="header">Matched Mode</td>
            <td>{matched_mode}</td>
        </tr>
        <tr>
            <td class="header">Descriptors Size</td>
            <td>{descriptors_size}</td>
        </tr>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

def display_custom_table_human_input(key_query):
    value = get_human_input_single(key_query)
    if value is not None:
        value = list(value)[0]
    else:
        value = "None"
    table_html = f"""
    <style>
    .custom-table {{
        border-collapse: collapse;
        table-layout: auto;
        margin-top: 10px;
    }}
    .custom-table td {{
        border: 1px solid black;
        padding: 10px;
        text-align: center;
        font-size: 16px;
    }}
    .custom-table .header {{
        background-color: #f2f2f2;
        font-weight: bold;
    }}
    </style>
    <table class="custom-table">
        <tr>
            <td class="header">Human Expert Input</td>
            <td>{value}</td>
        </tr>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    
def display_custom_table_ground_truth(ground_truth):
    table_html = f"""
    <style>
    .custom-table {{
        border-collapse: collapse;
        table-layout: auto;
        margin-top: 10px;
    }}
    .custom-table td {{
        border: 1px solid black;
        padding: 10px;
        text-align: center;
        font-size: 16px;
    }}
    .custom-table .header {{
        background-color: #f2f2f2;
        font-weight: bold;
    }}
    </style>
    <table class="custom-table">
        <tr>
            <td class="header">Ground Truth</td>
            <td>{ground_truth}</td>
        </tr>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

def get_matched_label(current_query_image, recom_rank):
    df = st.session_state.matching_results_table
    # Extract the basename from the current query image path
    current_image_name = os.path.basename(current_query_image)

    # Find the row that matches the current image name
    matching_row = df[df['path_relative_to_root'].apply(lambda x: os.path.basename(x)) == current_image_name]

    # Check if a matching row exists and return matched_label_, else return None
    if not matching_row.empty:
        ground_truth = None
        if 'AID2021' in list(matching_row.columns) and not np.isnan(matching_row['AID2021'].iloc[0]):
            ground_truth = int(matching_row['AID2021'].iloc[0])
        return int(matching_row['matched_label_' + str(recom_rank)].iloc[0]), int(matching_row['matched_img_serial_' + str(recom_rank)].iloc[0]), float("{:.2f}".format(matching_row['matching_mean_dist_' + str(recom_rank)].iloc[0])), int(matching_row['matching_mode_' + str(recom_rank)].iloc[0]), matching_row['descriptors_size'].iloc[0], ground_truth
    else:
        return None, None, None, None, None, None

def get_paths_by_matched_label_id(matched_label, matched_serial):
    df = st.session_state.metadata_table
    base_dir = st.session_state.root_dir
    
    # Find rows where the value in the AID2021 column matches matched_label
    matching_rows = df[df['AID2021'] == matched_label]

    # Find the row that exactly matches the provided serial number
    matching_row_exact = df[df['#Serial'] == matched_serial]

    # Get the exact path and prepend the base directory
    exact_path = None
    if not matching_row_exact.empty:
        exact_path = os.path.join(base_dir, matching_row_exact.iloc[0]['path_relative_to_root'])

    # Remove the exact match row from the matching_rows dataframe
    matching_rows = matching_rows[matching_rows['#Serial'] != matched_serial]

    # Get the remaining paths and prepend the base directory
    remaining_paths = [os.path.join(base_dir, path) for path in matching_rows['path_relative_to_root'].tolist()]
    remaining_paths_serials = [serial for serial in matching_rows['#Serial'].tolist()]

    # Combine the exact path (if found) with the remaining paths
    if exact_path is not None:
        all_paths = [exact_path] + remaining_paths
        all_serials = [matched_serial] + remaining_paths_serials
    else:
        all_paths = remaining_paths
        all_serials = remaining_paths_serials
    
    return all_paths, all_serials

def update_reference_images(current_query_image, recom_rank):
    matched_label, matched_serial, _, _, _, _ = get_matched_label(current_query_image, recom_rank)
    reference_image_paths, reference_image_serials = get_paths_by_matched_label_id(matched_label, matched_serial)
    return reference_image_paths

def update_reference_images_ground_truth(current_query_image, recom_rank):
    _, _, _, _, _, ground_truth = get_matched_label(current_query_image, recom_rank)
    reference_image_paths, reference_image_serials = get_paths_by_matched_label_id(ground_truth, None)
    return reference_image_paths

def update_human_input_single(key_query, human_input_value):
    base_names = st.session_state.matching_results_table['path_relative_to_root'].apply(os.path.basename)
    mask = base_names == os.path.basename(key_query)
    st.session_state.matching_results_table.loc[mask, 'human_input'] = human_input_value

def update_human_inputs_all(query_image_paths, human_input_value):
    df = st.session_state.matching_results_table
    df['filename'] = df['path_relative_to_root'].apply(os.path.basename)

    query_basenames = [os.path.basename(query_item) for query_item in query_image_paths]

    # Create a mask for all matching rows
    mask = df['filename'].isin(query_basenames)

    # Update the 'human_input' column for all matching rows
    df.loc[mask, 'human_input'] = human_input_value

def main_visualize_known_matched_false():
    
    # Get the paths of filtered query images
    st.session_state.known_matched_false, st.session_state.fp_table, st.session_state.fn_table =  find_algorithm_mistakes(st.session_state.matching_results_table)

    if st.session_state.known_matched_false is None:
        st.warning("Table for falsely matched in-sample items is empty. Try re-loading.")
    else:
        query_image_paths = list(os.path.join(st.session_state.root_dir, x) for x in st.session_state.known_matched_false['path_relative_to_root'])

        # Load all query images and their corresponding cropped torso images
        cropped_torso_paths_zoomed = [get_corresponding_torso_image(p, "zoomed_version" , "_zoomed") for p in query_image_paths]
        cropped_torso_paths = [get_corresponding_torso_image(p) for p in query_image_paths]

        # Create a session state variable to track the index of the currently displayed query image pair
        if 'curr_query_idx_known_matched_false' not in st.session_state:
            st.session_state.curr_query_idx_known_matched_false = 0

        # Create a session state variable to track the index of the currently displayed reference image set
        if 'curr_ref_idx_known_matched_false' not in st.session_state:
            st.session_state.curr_ref_idx_known_matched_false = 0
        
        if 'curr_ref_idx_gt_known_matched_false' not in st.session_state:
            st.session_state.curr_ref_idx_gt_known_matched_false = 0
            
        if 'ref_img_paths_known_matched_false' not in st.session_state:
            st.session_state.ref_img_paths_known_matched_false = []
            
        if 'ref_img_paths_gt_known_matched_false' not in st.session_state:
            st.session_state.ref_img_paths_gt_known_matched_false = []

        if 'recomms_rank_selected_known_matched_false' not in st.session_state:
            st.session_state.recomms_rank_selected_known_matched_false = 1

        # Initialize some vars
        matched_label = None
        
        # Create columns for layout
        col_query_img, col_predicted_match, col_ground_truth, col_info = st.columns([2, 2, 2, 1.5])
        
        with col_query_img:
            st.markdown("<h2 style='font-size:20px;'>Matched Query Images</h2>", unsafe_allow_html=True)
            # st.subheader('Matched Query Images')
            
            if len(query_image_paths) > 0:
                
                # Add button below the left column for the next query image
                if st.button('Next Query Image'):
                    st.session_state.recomms_rank_selected_known_matched_false = 1
                    if st.session_state.curr_query_idx_known_matched_false == (len(query_image_paths) - 1):
                        st.warning("You have looped over all query images once.")
                    st.session_state.curr_query_idx_known_matched_false = (st.session_state.curr_query_idx_known_matched_false + 1) % len(query_image_paths)
                    st.session_state.curr_ref_idx_known_matched_false = 0 
                    st.session_state.curr_ref_idx_gt_known_matched_false = 0 

                # Add button below the left column for previous query image
                if st.button('Previous Query Image'):
                    if (st.session_state.curr_query_idx_known_matched_false - 1) >= 0:
                        st.session_state.recomms_rank_selected_known_matched_false = 1
                        st.session_state.curr_query_idx_known_matched_false -= 1
                        st.session_state.curr_ref_idx_known_matched_false = 0 
                        st.session_state.curr_ref_idx_gt_known_matched_false = 0 
                    else:
                        st.warning("Not enough remained data for this action.")
                    
                # Progress bar
                st.write(f"{st.session_state.curr_query_idx_known_matched_false + 1} / {len(query_image_paths)}")
                
                # Get the current image paths based on the current index
                current_query_image = query_image_paths[st.session_state.curr_query_idx_known_matched_false]
                current_cropped_torso_zoomed_image = cropped_torso_paths_zoomed[st.session_state.curr_query_idx_known_matched_false]
                current_cropped_torso_image = cropped_torso_paths[st.session_state.curr_query_idx_known_matched_false]
                
                # Display the left panel images
                display_left_query_images_matched_images(
                    [current_query_image, current_cropped_torso_zoomed_image, current_cropped_torso_image], 
                    ['Query Image: ' + os.path.basename(current_query_image),
                    'Cropped Torso Zoomed Image: ' + os.path.basename(current_cropped_torso_zoomed_image),
                    'Cropped Torso Image: ' + os.path.basename(current_cropped_torso_image)]
                )
                st.session_state.ref_img_paths_known_matched_false = update_reference_images(current_query_image, st.session_state.recomms_rank_selected_known_matched_false)
                st.session_state.ref_img_paths_gt_known_matched_false = update_reference_images_ground_truth(current_query_image, st.session_state.recomms_rank_selected_known_matched_false)
            else:
                st.warning("No query images found in the specified directory.")

        with col_info:  
            st.markdown("<h2 style='font-size:20px;'>Action</h2>", unsafe_allow_html=True)                    
            # st.subheader('Action')

            if len(query_image_paths) > 0:
                
                if st.button('Next Algorithmic Matched ID'):
                    st.session_state.recomms_rank_selected_known_matched_false = max (1, (st.session_state.recomms_rank_selected_known_matched_false + 1) % (st.session_state.num_id_recomms + 1))
                    st.session_state.curr_ref_idx_known_matched_false = 0  # Reset reference index for new reference image
                    st.session_state.curr_ref_idx_gt_known_matched_false = 0
                    st.session_state.ref_img_paths_known_matched_false = update_reference_images(current_query_image, st.session_state.recomms_rank_selected_known_matched_false)
                    st.session_state.ref_img_paths_gt_known_matched_false = update_reference_images_ground_truth(current_query_image, st.session_state.recomms_rank_selected_known_matched_false)
                    st.success("Next Algorithmic Matched ID clicked! Rank: {}".format(st.session_state.recomms_rank_selected_known_matched_false))
                    
                # Show matching results
                st.markdown("<h2 style='font-size:20px;'>Matching Results</h2>", unsafe_allow_html=True)
                # st.subheader('Matching Results')
                matched_label, matched_serial, matched_dist, matched_mode, descriptors_size, ground_truth = get_matched_label(current_query_image, st.session_state.recomms_rank_selected_known_matched_false)
                display_custom_table(matched_label, matched_serial, matched_dist, matched_mode, descriptors_size, st.session_state.recomms_rank_selected_known_matched_false)
                
                # Show ground truth
                display_custom_table_ground_truth(ground_truth)
                    
        with col_predicted_match:
            st.markdown("<h2 style='font-size:20px;'>Matched Reference Images</h2>", unsafe_allow_html=True)
            # st.subheader('Matched Reference Images')
        
            num_rows, num_cols = 1, 1
            
            # Get the current query image for reference image retrieval
            if len(query_image_paths) > 0:
                if len(st.session_state.ref_img_paths_known_matched_false) > 0:
                    
                    # Add button below the right image column for the next reference image
                    if st.button('Next Matched Reference Image'):
                        
                        st.session_state.curr_ref_idx_known_matched_false += 1  # Move to the next set
                        
                        if st.session_state.curr_ref_idx_known_matched_false * num_rows * num_cols >= len(st.session_state.ref_img_paths_known_matched_false):
                            st.session_state.curr_ref_idx_known_matched_false = 0
                    
                    # Add button below the left column for previous reference image
                    if st.button('Previous Matched Reference Image'):
                        if (st.session_state.curr_ref_idx_known_matched_false - 1) >= 0:
                            st.session_state.curr_ref_idx_known_matched_false -= 1
                        else:
                            st.warning("You have reached the first item.")
                            
                    # Progress bar
                    st.write(f"{st.session_state.curr_ref_idx_known_matched_false + 1} / {len(st.session_state.ref_img_paths_known_matched_false)}")

                    # Display images
                    starting_idx = st.session_state.curr_ref_idx_known_matched_false * num_rows * num_cols
                    display_right_reference_image(st.session_state.ref_img_paths_known_matched_false, starting_idx, num_rows, num_cols)
                else:
                    st.warning("No reference images found for the current query image.")
            else:
                st.warning("No query images found in the specified directory.")
        
        with col_ground_truth:
            st.markdown("<h2 style='font-size:20px;'>Ground Truth</h2>", unsafe_allow_html=True)
            # st.subheader('Ground Truth')
            
            num_rows, num_cols = 1, 1
            
            # Get the current query image for reference image retrieval
            if len(query_image_paths) > 0:
                if len(st.session_state.ref_img_paths_gt_known_matched_false) > 0:
                    
                    # Add button below the right image column for the next reference image
                    if st.button('Next Ground Truth Reference Image'):
                        
                        st.session_state.curr_ref_idx_gt_known_matched_false += 1  # Move to the next set
                        
                        if st.session_state.curr_ref_idx_gt_known_matched_false * num_rows * num_cols >= len(st.session_state.ref_img_paths_gt_known_matched_false):
                            st.session_state.curr_ref_idx_gt_known_matched_false = 0
                    
                    # Add button below the left column for previous reference image
                    if st.button('Previous Ground Truth Reference Image'):
                        if (st.session_state.curr_ref_idx_gt_known_matched_false - 1) >= 0:
                            st.session_state.curr_ref_idx_gt_known_matched_false -= 1
                        else:
                            st.warning("You have reached the first item.")
                            
                    # Progress bar
                    st.write(f"{st.session_state.curr_ref_idx_gt_known_matched_false + 1} / {len(st.session_state.ref_img_paths_gt_known_matched_false)}")

                    # Display images
                    starting_idx = st.session_state.curr_ref_idx_gt_known_matched_false * num_rows * num_cols
                    display_right_reference_image(st.session_state.ref_img_paths_gt_known_matched_false, starting_idx, num_rows, num_cols)
                else:
                    st.warning("No reference images found for the current query image.")
            else:
                st.warning("No query images found in the specified directory.")

def main_visualize_fn_table():
    
    # Get the paths of filtered query images
    st.session_state.known_matched_false, st.session_state.fp_table, st.session_state.fn_table =  find_algorithm_mistakes(st.session_state.matching_results_table)
    
    if st.session_state.fn_table is None:
        st.warning("Table for falsely unmatched in-sample items is empty. Try re-loading.")
    else:
        query_image_paths = list(os.path.join(st.session_state.root_dir, x) for x in st.session_state.fn_table['path_relative_to_root'])

        # Load all query images and their corresponding cropped torso images
        cropped_torso_paths_zoomed = [get_corresponding_torso_image(p, "zoomed_version" , "_zoomed") for p in query_image_paths]
        cropped_torso_paths = [get_corresponding_torso_image(p) for p in query_image_paths]

        # Create a session state variable to track the index of the currently displayed query image pair
        if 'curr_query_idx_fn_table' not in st.session_state:
            st.session_state.curr_query_idx_fn_table = 0

        # Create a session state variable to track the index of the currently displayed reference image set
        if 'curr_ref_idx_fn_table' not in st.session_state:
            st.session_state.curr_ref_idx_fn_table = 0
        
        if 'curr_ref_idx_gt_fn_table' not in st.session_state:
            st.session_state.curr_ref_idx_gt_fn_table = 0
            
        if 'ref_img_paths_fn_table' not in st.session_state:
            st.session_state.ref_img_paths_fn_table = []
            
        if 'ref_img_paths_gt_fn_table' not in st.session_state:
            st.session_state.ref_img_paths_gt_fn_table = []

        if 'recomms_rank_selected_fn_table' not in st.session_state:
            st.session_state.recomms_rank_selected_fn_table = 1

        # Initialize some vars
        matched_label = None
        
        # Create columns for layout
        col_query_img, col_predicted_match, col_ground_truth, col_info = st.columns([2, 2, 2, 1.5])
        
        with col_query_img:
            st.markdown("<h2 style='font-size:20px;'>Matched Query Images</h2>", unsafe_allow_html=True)
            # st.subheader('Matched Query Images')
            
            if len(query_image_paths) > 0:
                
                # Add button below the left column for the next query image
                if st.button('Next Query Image'):
                    st.session_state.recomms_rank_selected_fn_table = 1
                    if st.session_state.curr_query_idx_fn_table == (len(query_image_paths) - 1):
                        st.warning("You have looped over all query images once.")
                    st.session_state.curr_query_idx_fn_table = (st.session_state.curr_query_idx_fn_table + 1) % len(query_image_paths)
                    st.session_state.curr_ref_idx_fn_table = 0 
                    st.session_state.curr_ref_idx_gt_fn_table = 0 

                # Add button below the left column for previous query image
                if st.button('Previous Query Image'):
                    if (st.session_state.curr_query_idx_fn_table - 1) >= 0:
                        st.session_state.recomms_rank_selected_fn_table = 1
                        st.session_state.curr_query_idx_fn_table -= 1
                        st.session_state.curr_ref_idx_fn_table = 0 
                        st.session_state.curr_ref_idx_gt_fn_table = 0 
                    else:
                        st.warning("Not enough remained data for this action.")
                    
                # Progress bar
                st.write(f"{st.session_state.curr_query_idx_fn_table + 1} / {len(query_image_paths)}")
                
                # Get the current image paths based on the current index
                current_query_image = query_image_paths[st.session_state.curr_query_idx_fn_table]
                current_cropped_torso_zoomed_image = cropped_torso_paths_zoomed[st.session_state.curr_query_idx_fn_table]
                current_cropped_torso_image = cropped_torso_paths[st.session_state.curr_query_idx_fn_table]
                
                # Display the left panel images
                display_left_query_images_matched_images(
                    [current_query_image, current_cropped_torso_zoomed_image, current_cropped_torso_image], 
                    ['Query Image: ' + os.path.basename(current_query_image),
                    'Cropped Torso Zoomed Image: ' + os.path.basename(current_cropped_torso_zoomed_image),
                    'Cropped Torso Image: ' + os.path.basename(current_cropped_torso_image)]
                )
                st.session_state.ref_img_paths_fn_table = update_reference_images(current_query_image, st.session_state.recomms_rank_selected_fn_table)
                st.session_state.ref_img_paths_gt_fn_table = update_reference_images_ground_truth(current_query_image, st.session_state.recomms_rank_selected_fn_table)
            else:
                st.warning("No query images found in the specified directory.")

        with col_info:  
            st.markdown("<h2 style='font-size:20px;'>Actions</h2>", unsafe_allow_html=True)           
            # st.subheader('Action')

            if len(query_image_paths) > 0:
                
                if st.button('Next Algorithmic Matched ID'):
                    st.session_state.recomms_rank_selected_fn_table = max (1, (st.session_state.recomms_rank_selected_fn_table + 1) % (st.session_state.num_id_recomms + 1))
                    st.session_state.curr_ref_idx_fn_table = 0  # Reset reference index for new reference image
                    st.session_state.curr_ref_idx_gt_fn_table = 0
                    st.session_state.ref_img_paths_fn_table = update_reference_images(current_query_image, st.session_state.recomms_rank_selected_fn_table)
                    st.session_state.ref_img_paths_gt_fn_table = update_reference_images_ground_truth(current_query_image, st.session_state.recomms_rank_selected_fn_table)
                    st.success("Next Algorithmic Matched ID clicked! Rank: {}".format(st.session_state.recomms_rank_selected_fn_table))
                    
                # Show matching results
                st.markdown("<h2 style='font-size:20px;'>Matching Results</h2>", unsafe_allow_html=True)  
                # st.subheader('Matching Results')
                matched_label, matched_serial, matched_dist, matched_mode, descriptors_size, ground_truth = get_matched_label(current_query_image, st.session_state.recomms_rank_selected_fn_table)
                display_custom_table(matched_label, matched_serial, matched_dist, matched_mode, descriptors_size, st.session_state.recomms_rank_selected_fn_table)
                
                # Show ground truth
                display_custom_table_ground_truth(ground_truth)

        with col_predicted_match:
            st.markdown("<h2 style='font-size:20px;'>Matched Reference (Rejected)</h2>", unsafe_allow_html=True)
            # st.subheader('Matched Reference Images (Rejected)')
        
            num_rows, num_cols = 1, 1
            
            # Get the current query image for reference image retrieval
            if len(query_image_paths) > 0:
                if len(st.session_state.ref_img_paths_fn_table) > 0:
                    
                    # Add button below the right image column for the next reference image
                    if st.button('Next Matched Reference Image'):
                        
                        st.session_state.curr_ref_idx_fn_table += 1  # Move to the next set
                        
                        if st.session_state.curr_ref_idx_fn_table * num_rows * num_cols >= len(st.session_state.ref_img_paths_fn_table):
                            st.session_state.curr_ref_idx_fn_table = 0
                    
                    # Add button below the left column for previous reference image
                    if st.button('Previous Matched Reference Image'):
                        if (st.session_state.curr_ref_idx_fn_table - 1) >= 0:
                            st.session_state.curr_ref_idx_fn_table -= 1
                        else:
                            st.warning("You have reached the first item.")
                            
                    # Progress bar
                    st.write(f"{st.session_state.curr_ref_idx_fn_table + 1} / {len(st.session_state.ref_img_paths_fn_table)}")

                    # Display images
                    starting_idx = st.session_state.curr_ref_idx_fn_table * num_rows * num_cols
                    display_right_reference_image(st.session_state.ref_img_paths_fn_table, starting_idx, num_rows, num_cols)
                else:
                    st.warning("No reference images found for the current query image.")
            else:
                st.warning("No query images found in the specified directory.")
                
        with col_ground_truth:
            st.markdown("<h2 style='font-size:20px;'>Ground Truth</h2>", unsafe_allow_html=True)

            # st.subheader('Ground Truth')
            
            num_rows, num_cols = 1, 1
            
            # Get the current query image for reference image retrieval
            if len(query_image_paths) > 0:
                if len(st.session_state.ref_img_paths_gt_fn_table) > 0:
                    
                    # Add button below the right image column for the next reference image
                    if st.button('Next Ground Truth Reference Image'):
                        
                        st.session_state.curr_ref_idx_gt_fn_table += 1  # Move to the next set
                        
                        if st.session_state.curr_ref_idx_gt_fn_table * num_rows * num_cols >= len(st.session_state.ref_img_paths_gt_fn_table):
                            st.session_state.curr_ref_idx_gt_fn_table = 0
                    
                    # Add button below the left column for previous reference image
                    if st.button('Previous Ground Truth Reference Image'):
                        if (st.session_state.curr_ref_idx_gt_fn_table - 1) >= 0:
                            st.session_state.curr_ref_idx_gt_fn_table -= 1
                        else:
                            st.warning("You have reached the first item.")
                            
                    # Progress bar
                    st.write(f"{st.session_state.curr_ref_idx_gt_fn_table + 1} / {len(st.session_state.ref_img_paths_gt_fn_table)}")

                    # Display images
                    starting_idx = st.session_state.curr_ref_idx_gt_fn_table * num_rows * num_cols
                    display_right_reference_image(st.session_state.ref_img_paths_gt_fn_table, starting_idx, num_rows, num_cols)
                else:
                    st.warning("No reference images found for the current query image.")
            else:
                st.warning("No query images found in the specified directory.")

def main_visualize_unknown_matched_false():
    
    # Get the paths of filtered query images
    st.session_state.known_matched_false, st.session_state.fp_table, st.session_state.fn_table =  find_algorithm_mistakes(st.session_state.matching_results_table)

    if st.session_state.fp_table is None:
        st.warning("Table for falsely matched out-of-sample items is empty. Try re-loading.")
    else:
        query_image_paths = list(os.path.join(st.session_state.root_dir, x) for x in st.session_state.fp_table['path_relative_to_root'])

        # Load all query images and their corresponding cropped torso images
        cropped_torso_paths_zoomed = [get_corresponding_torso_image(p, "zoomed_version" , "_zoomed") for p in query_image_paths]
        cropped_torso_paths = [get_corresponding_torso_image(p) for p in query_image_paths]

        # Create a session state variable to track the index of the currently displayed query image pair
        if 'curr_query_idx_fp_table' not in st.session_state:
            st.session_state.curr_query_idx_fp_table = 0

        # Create a session state variable to track the index of the currently displayed reference image set
        if 'curr_ref_idx_fp_table' not in st.session_state:
            st.session_state.curr_ref_idx_fp_table = 0
        
        if 'curr_ref_idx_gt_fp_table' not in st.session_state:
            st.session_state.curr_ref_idx_gt_fp_table = 0
            
        if 'ref_img_paths_fp_table' not in st.session_state:
            st.session_state.ref_img_paths_fp_table = []
            
        if 'ref_img_paths_gt_fp_table' not in st.session_state:
            st.session_state.ref_img_paths_gt_fp_table = []

        if 'recomms_rank_selected_fp_table' not in st.session_state:
            st.session_state.recomms_rank_selected_fp_table = 1

        # Initialize some vars
        matched_label = None
        
        # Create columns for layout
        col_query_img, col_predicted_match, col_info = st.columns([2, 2, 1.5])
        
        with col_query_img:
            st.markdown("<h2 style='font-size:20px;'>Matched Query Images</h2>", unsafe_allow_html=True)
            # st.subheader('Matched Query Images')
            
            if len(query_image_paths) > 0:
                
                # Add button below the left column for the next query image
                if st.button('Next Query Image'):
                    st.session_state.recomms_rank_selected_fp_table = 1
                    if st.session_state.curr_query_idx_fp_table == (len(query_image_paths) - 1):
                        st.warning("You have looped over all query images once.")
                    st.session_state.curr_query_idx_fp_table = (st.session_state.curr_query_idx_fp_table + 1) % len(query_image_paths)
                    st.session_state.curr_ref_idx_fp_table = 0 
                    st.session_state.curr_ref_idx_gt_fp_table = 0 

                # Add button below the left column for previous query image
                if st.button('Previous Query Image'):
                    if (st.session_state.curr_query_idx_fp_table - 1) >= 0:
                        st.session_state.recomms_rank_selected_fp_table = 1
                        st.session_state.curr_query_idx_fp_table -= 1
                        st.session_state.curr_ref_idx_fp_table = 0 
                        st.session_state.curr_ref_idx_gt_fp_table = 0 
                    else:
                        st.warning("Not enough remained data for this action.")
                    
                # Progress bar
                st.write(f"{st.session_state.curr_query_idx_fp_table + 1} / {len(query_image_paths)}")
                
                # Get the current image paths based on the current index
                current_query_image = query_image_paths[st.session_state.curr_query_idx_fp_table]
                current_cropped_torso_zoomed_image = cropped_torso_paths_zoomed[st.session_state.curr_query_idx_fp_table]
                current_cropped_torso_image = cropped_torso_paths[st.session_state.curr_query_idx_fp_table]
                
                # Display the left panel images
                display_left_query_images_matched_images(
                    [current_query_image, current_cropped_torso_zoomed_image, current_cropped_torso_image], 
                    ['Query Image: ' + os.path.basename(current_query_image),
                    'Cropped Torso Zoomed Image: ' + os.path.basename(current_cropped_torso_zoomed_image),
                    'Cropped Torso Image: ' + os.path.basename(current_cropped_torso_image)]
                )
                st.session_state.ref_img_paths_fp_table = update_reference_images(current_query_image, st.session_state.recomms_rank_selected_fp_table)
                st.session_state.ref_img_paths_gt_fp_table = update_reference_images_ground_truth(current_query_image, st.session_state.recomms_rank_selected_fp_table)
            else:
                st.warning("No query images found in the specified directory.")

        with col_info:  
            st.markdown("<h2 style='font-size:20px;'>Action</h2>", unsafe_allow_html=True)
            # st.subheader('Action')

            if len(query_image_paths) > 0:
                
                if st.button('Next Algorithmic Matched ID'):
                    st.session_state.recomms_rank_selected_fp_table = max (1, (st.session_state.recomms_rank_selected_fp_table + 1) % (st.session_state.num_id_recomms + 1))
                    st.session_state.curr_ref_idx_fp_table = 0  # Reset reference index for new reference image
                    st.session_state.curr_ref_idx_gt_fp_table = 0
                    st.session_state.ref_img_paths_fp_table = update_reference_images(current_query_image, st.session_state.recomms_rank_selected_fp_table)
                    st.session_state.ref_img_paths_gt_fp_table = update_reference_images_ground_truth(current_query_image, st.session_state.recomms_rank_selected_fp_table)
                    st.success("Next Algorithmic Matched ID clicked! Rank: {}".format(st.session_state.recomms_rank_selected_fp_table))
                    
                # Show matching results
                st.markdown("<h2 style='font-size:20px;'>Matching Results</h2>", unsafe_allow_html=True)
                # st.subheader('Matching Results')
                matched_label, matched_serial, matched_dist, matched_mode, descriptors_size, ground_truth = get_matched_label(current_query_image, st.session_state.recomms_rank_selected_fp_table)
                display_custom_table(matched_label, matched_serial, matched_dist, matched_mode, descriptors_size, st.session_state.recomms_rank_selected_fp_table)
                
                # Show ground truth
                display_custom_table_ground_truth(ground_truth)

        with col_predicted_match:
            st.markdown("<h2 style='font-size:20px;'>Matched Reference (False)</h2>", unsafe_allow_html=True)
            # st.subheader('Matched Reference Images (False)')
        
            num_rows, num_cols = 1, 1
            
            # Get the current query image for reference image retrieval
            if len(query_image_paths) > 0:
                if len(st.session_state.ref_img_paths_fp_table) > 0:
                    
                    # Add button below the right image column for the next reference image
                    if st.button('Next Matched Reference Image'):
                        
                        st.session_state.curr_ref_idx_fp_table += 1  # Move to the next set
                        
                        if st.session_state.curr_ref_idx_fp_table * num_rows * num_cols >= len(st.session_state.ref_img_paths_fp_table):
                            st.session_state.curr_ref_idx_fp_table = 0
                    
                    # Add button below the left column for previous reference image
                    if st.button('Previous Matched Reference Image'):
                        if (st.session_state.curr_ref_idx_fp_table - 1) >= 0:
                            st.session_state.curr_ref_idx_fp_table -= 1
                        else:
                            st.warning("You have reached the first item.")
                        
                    # Progress bar
                    st.write(f"{st.session_state.curr_ref_idx_fp_table + 1} / {len(st.session_state.ref_img_paths_fp_table)}")

                    # Display images
                    starting_idx = st.session_state.curr_ref_idx_fp_table * num_rows * num_cols
                    display_right_reference_image(st.session_state.ref_img_paths_fp_table, starting_idx, num_rows, num_cols)
                else:
                    st.warning("No reference images found for the current query image.")
            else:
                st.warning("No query images found in the specified directory.")


# Display the dropdown menu with increased font size
st.markdown(f'<div class="radio-font-size">Select Analysis:</div>', unsafe_allow_html=True)
selected_option = st.selectbox("", ['Evaluate Model Results',
                                'Visualize Falsely Matched Images (in-sample)',
                                'Visualize Falsely Matched Images (out-of-sample)',
                                'Visualize Falsely Unmatched Images (in-sample)',
                                ],
                            key="dropdown_options")

# Add a styled horizontal line below the radio button section
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

# Run the corresponding function based on the selected option

if selected_option == 'Evaluate Model Results':

    main_compute_matching_accuracy(pipeline_code_dir, demo_images[9])

elif selected_option == 'Visualize Falsely Matched Images (in-sample)':
    
    # Load new metadata if needed
    initialize_vizualization_project_validation()

    if ('matching_results_table' in st.session_state) and ('matching_attempt' in st.session_state.matching_results_table.columns):
        
        main_visualize_known_matched_false()
    
    else:
        st.warning("Matching results not available or you need to start project from matched results tab.")

elif selected_option == 'Visualize Falsely Matched Images (out-of-sample)':
    
    # Load new metadata if needed
    initialize_vizualization_project_validation()

    if ('matching_results_table' in st.session_state) and ('matching_attempt' in st.session_state.matching_results_table.columns):
        
        main_visualize_unknown_matched_false()
    
    else:
        st.warning("Matching results not available or you need to start project from matched results tab.")
        
elif selected_option == 'Visualize Falsely Unmatched Images (in-sample)':
    
    # Load new metadata if needed
    initialize_vizualization_project_validation()

    if ('matching_results_table' in st.session_state) and ('matching_attempt' in st.session_state.matching_results_table.columns):
        
        main_visualize_fn_table()
    
    else:
        st.warning("Matching results not available or you need to start project from matched results tab.")
            