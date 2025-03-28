# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_matching import pipeline_code_relative_dir
from utils.helpers_matching import load_data_dirs, load_metadata_file


st.title("Review & Revise Re-identification Results")
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


pipeline_code_dir = os.path.join(str(Path(__file__).resolve().parent.parent), str(pipeline_code_relative_dir))
st.session_state.root_dir, st.session_state.processed_img_dir = load_data_dirs()

def save_human_inputs():
    print('Human Expert Inputs Saved!')
    partition = 'query'
    metadata_filepath = os.path.join(st.session_state.root_dir, 'query_dir', 'metadata_' + partition + '.csv')
    st.session_state.matching_results_table.to_csv(metadata_filepath, index=False)
    
def get_all_query_image_paths(matching_status_key):
    
    query_metadata = st.session_state.matching_results_table
    query_metadata = query_metadata.sort_values(by='matching_mode_1', ascending=False)
    selected_items = list(os.path.join(st.session_state.root_dir, x) for x in query_metadata.loc[query_metadata['matching_status']==matching_status_key, 'path_relative_to_root'])
    
    return selected_items

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
    
def initialize_vizualization_project():
    st.session_state.metadata = {}
    for partition in ['query', 'reference']:
        metadata_filepath = os.path.join(st.session_state.root_dir, partition + '_dir', 'metadata_' + partition + '.csv')
        st.session_state.metadata[partition] = load_metadata_file(metadata_filepath)
        
    st.markdown(f'<div class="radio-font-size">Load Project:</div>', unsafe_allow_html=True)
    st.text(
        "How do you want to proceed? "
        "If Resume, you will only lose human inputs entered and not saved in this session. "
        "If Restart, you will lose human inputs entered and not saved in this session as well as any input previsouly saved in query metadata. "
        "Once selected choose Lock the Choice to contiune reviewing for both sets of results."
    )
    response = st.radio(
        "",
        options=["Lock the Choice", "Resume Project", "Start New Project"]
    )    
    if response == "Start New Project":
        if st.button("Confirm"):
            # user can use this option to reload the query metadata and start fresh
            st.session_state.metadata_table = st.session_state.metadata['reference'].copy()
            st.session_state.matching_results_table = st.session_state.metadata['query'].copy()
            
            reset_state_vars()
            
            if 'human_input' in st.session_state.matching_results_table.columns:
                st.session_state.matching_results_table['human_input'] = None
            
            st.success("Matching data updated and human_input column erased.")
        
    
    elif response == "Resume Project":
        if st.button("Confirm"):
            # user can use this option to reload the query metadata and start fresh
            st.session_state.metadata_table = st.session_state.metadata['reference'].copy()
            st.session_state.matching_results_table = st.session_state.metadata['query'].copy()
            
            reset_state_vars()
                    
            st.success("Matching data updated and human_input column is initialized.")
            
def overwrite_matching_results(key_query):
    base_names = st.session_state.matching_results_table['path_relative_to_root'].apply(os.path.basename)
    mask = base_names == os.path.basename(key_query)
    st.session_state.matching_results_table.loc[mask, 'matching_status'] = 'matched'
    
def get_human_input_single(key_query):
    base_names = st.session_state.matching_results_table['path_relative_to_root'].apply(os.path.basename)
    mask = base_names == os.path.basename(key_query)
    
    if 'human_input' in st.session_state.matching_results_table.columns and len(st.session_state.matching_results_table.loc[mask, 'human_input'])!=0 and not st.session_state.matching_results_table.loc[mask, 'human_input'].isnull().all():
        return st.session_state.matching_results_table.loc[mask, 'human_input']
    else:
        return None
     
def display_left_query_images_matched_images(image_paths, captions):
    # Create a single column
    col = st.columns(1)[0]

    # Iterate through the images and display them in the single column
    for i in [1,0]:
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
            col.image(img_resized, caption=captions[i], use_container_width=False)
            
        else:  # Second image: Fixed size
            img_resized = img.resize((256, 256))  # Adjusted to a more viewable size
            
            # Display image with caption in the single column
            col.image(img_resized, caption='Cropped Torso Image', use_container_width=True)
 
def display_left_query_images_not_matched_images(image_paths, captions):
    cols = st.columns(1)[0]
    for i in [1,0]:
        if i ==1:
            image_path = image_paths[i]
            if not os.path.isfile(image_path):
                raise FileNotFoundError("Image file '{}' not found.".format(image_path))
            img = ImageOps.exif_transpose(Image.open(image_path))
            img_resized = img.resize((256, 256))
            cols.image(img_resized, caption='Cropped Torso Image', use_container_width=True)
        elif i==0:
            image_path = image_paths[i]
            if not os.path.isfile(image_path):
                raise FileNotFoundError("Image file '{}' not found.".format(image_path))
            img = ImageOps.exif_transpose(Image.open(image_path))
            img_resized = img.resize((int(img.width * 0.15), int(img.height * 0.15)))
            cols.image(img_resized, caption=captions[i], use_container_width=False)
 
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
                col.image(torso_img, caption="Cropped Torso Image", use_container_width=True)
                
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

def update_human_inputs_all(query_image_paths, human_input_value):
    df = st.session_state.matching_results_table
    df['filename'] = df['path_relative_to_root'].apply(os.path.basename)

    query_basenames = [os.path.basename(query_item) for query_item in query_image_paths]

    # Create a mask for all matching rows
    mask = df['filename'].isin(query_basenames)

    # Update the 'human_input' column for all matching rows
    df.loc[mask, 'human_input'] = human_input_value
    
def get_corresponding_torso_image(image_path, subdir_name="original_size", suffix=""):
    
    # find path for _cropped_torso_zoomed
    parts = image_path.rsplit(".", 1)
    img_filename = os.path.basename(image_path)
    cropped_torso_zoomed_image_dir = os.path.join(st.session_state.processed_img_dir, subdir_name, img_filename).replace("." + parts[1], "_cropped_torso" + suffix + "." + parts[1])
            
    return cropped_torso_zoomed_image_dir

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
    
def main_analyze_matched_images(query_image_paths):
    
    # Load all query images and their corresponding cropped torso images
    cropped_torso_paths_zoomed = [get_corresponding_torso_image(p, "zoomed_version" , "_zoomed") for p in query_image_paths]
    cropped_torso_paths = [get_corresponding_torso_image(p) for p in query_image_paths]

    # Create a session state variable to track the index of the currently displayed query image pair
    if 'current_query_index' not in st.session_state:
        st.session_state.current_query_index = 0

    # Create a session state variable to track the index of the currently displayed reference image set
    if 'current_reference_index' not in st.session_state:
        st.session_state.current_reference_index = 0
    
    if 'reference_image_paths' not in st.session_state:
        st.session_state.reference_image_paths = []  

    if 'recomms_rank_selected' not in st.session_state:
        st.session_state.recomms_rank_selected = 1

    # Initialize some vars
    matched_label = None
    
    # Create columns for layout
    col_middle, col_right, col_left = st.columns([3, 3, 1.5])
    
    with col_middle:

        st.subheader('Matched Query Images')
        
        if len(query_image_paths) > 0:
            
            # Add button below the left column for the next query image
            if st.button('Next Query Image'):
                st.session_state.recomms_rank_selected = 1
                if st.session_state.current_query_index == (len(query_image_paths) - 1):
                    st.warning("You have looped over all query images once and saved human inputs if action items were pressed.")
                st.session_state.current_query_index = (st.session_state.current_query_index + 1) % len(query_image_paths)
                st.session_state.current_reference_index = 0 

            # Add button below the left column for previous query image
            if st.button('Previous Query Image'):
                if (st.session_state.current_query_index - 1) >= 0:
                    st.session_state.recomms_rank_selected = 1
                    st.session_state.current_query_index -= 1
                    st.session_state.current_reference_index = 0 
                else:
                    st.warning("You reached the first item.")
            
            # Add button below the left column to jump forward 100 images
            if st.button('Jump Forward 100 Images'):
                if (st.session_state.current_query_index + 100) < len(query_image_paths):
                    st.session_state.recomms_rank_selected = 1
                    st.session_state.current_query_index += 100
                    st.session_state.current_reference_index = 0  
                else:
                    st.warning("Not enough data available for this action.")
            
            # Add button below the left column to jump forward 100 images
            if st.button('Jump Backward 100 Images'):
                if (st.session_state.current_query_index - 100) >= 0:
                    st.session_state.recomms_rank_selected = 1
                    st.session_state.current_query_index -= 100
                    st.session_state.current_reference_index = 0  
                else:
                    st.warning("Not enough data available for this action.")
                
            # Progress bar
            st.write(f"{st.session_state.current_query_index + 1} / {len(query_image_paths)}")
            
            # Get the current image paths based on the current index
            current_query_image = query_image_paths[st.session_state.current_query_index]
            current_cropped_torso_zoomed_image = cropped_torso_paths_zoomed[st.session_state.current_query_index]
            current_cropped_torso_image = cropped_torso_paths[st.session_state.current_query_index]
            
            # Display the left panel images
            display_left_query_images_matched_images(
                [current_query_image, current_cropped_torso_zoomed_image, current_cropped_torso_image], 
                ['Query Image: ' + os.path.basename(current_query_image),
                 'Cropped Torso Zoomed Image: ' + os.path.basename(current_cropped_torso_zoomed_image),
                 'Cropped Torso Image: ' + os.path.basename(current_cropped_torso_image)]
            )
            st.session_state.reference_image_paths = update_reference_images(current_query_image, st.session_state.recomms_rank_selected)
        else:
            st.warning("No query images found in the specified directory.")

    with col_left:  
                              
        st.subheader('Actions')

        if len(query_image_paths) > 0:
            
            if st.button('Accept This Matched ID'):
                human_input_value = 'AcceptId'
                update_human_input_single(current_query_image, human_input_value)
                st.success("Accept Matched ID Clicked!")
                
            if st.button('Assign New ID'):
                human_input_value = 'AssignNewId'
                update_human_input_single(current_query_image, 'AssignNewId')
                st.success(f"Assign New ID Clicked!")

            if st.button('Skip This Query Image'):
                human_input_value = 'SkipImage'
                update_human_input_single(current_query_image, human_input_value)
                st.success("Skip This Query Image Clicked!")
                
            if st.button('Next Algorithmic Matched ID'):
                st.session_state.recomms_rank_selected = max (1, (st.session_state.recomms_rank_selected + 1) % (st.session_state.num_id_recomms + 1))
                st.session_state.current_reference_index = 0  # Reset reference index for new reference image
                st.session_state.reference_image_paths = update_reference_images(current_query_image, st.session_state.recomms_rank_selected)
                st.success("Next Algorithmic Matched ID clicked! Rank: {}".format(st.session_state.recomms_rank_selected))
            
            if st.button('Accept All Matched IDs'):
                human_input_value = 'AcceptId'
                update_human_inputs_all(query_image_paths, human_input_value)
                st.success(f"All Matched IDs Accepted!")
            
            if st.button('Save Analyzed Results'):
                save_human_inputs()
                st.success("Results Saved!")
    
            # Show matching results
            st.subheader('Matching Results')
            matched_label, matched_serial, matched_dist, matched_mode, descriptors_size, ground_truth = get_matched_label(current_query_image, st.session_state.recomms_rank_selected)
            display_custom_table(matched_label, matched_serial, matched_dist, matched_mode, descriptors_size, st.session_state.recomms_rank_selected)
            
            # Show human input
            display_custom_table_human_input(current_query_image)
            
            # Show ground truth
            display_custom_table_ground_truth(ground_truth)
                
    with col_right:
        st.subheader('Matched Reference Images')
    
        num_rows, num_cols = 1, 1
        
        # Get the current query image for reference image retrieval
        if len(query_image_paths) > 0:
            if len(st.session_state.reference_image_paths) > 0:
                
                # Add button below the right image column for the next reference image
                if st.button('Next Reference Image'):
                    
                    st.session_state.current_reference_index += 1  # Move to the next set
                    
                    if st.session_state.current_reference_index * num_rows * num_cols >= len(st.session_state.reference_image_paths):
                        st.session_state.current_reference_index = 0
                
                # Add button below the left column for previous reference image
                if st.button('Previous Reference Image'):
                    if (st.session_state.current_reference_index - 1) >= 0:
                        st.session_state.current_reference_index -= 1
                    else:
                        st.warning("You have reached the first item.")

                # Add button below the left column to jump forward 10 images
                if st.button('Jump Forward 10 Images'):
                    if (st.session_state.current_reference_index + 10) < len(st.session_state.reference_image_paths):
                        st.session_state.current_reference_index += 10
                    else:
                        st.warning("Not enough data available for this action.")
                
                # Add button below the left column to jump forward 10 images
                if st.button('Jump Backward 10 Images'):
                    if (st.session_state.current_reference_index - 10) >= 0:
                        st.session_state.current_reference_index -= 10
                    else:
                        st.warning("Not enough data available for this action.")
                    
                # Progress bar
                st.write(f"{st.session_state.current_reference_index + 1} / {len(st.session_state.reference_image_paths)}")

                # Display images
                starting_idx = st.session_state.current_reference_index * num_rows * num_cols
                display_right_reference_image(st.session_state.reference_image_paths, starting_idx, num_rows, num_cols)
            else:
                st.warning("No reference images found for the current query image.")
        else:
            st.warning("No query images found in the specified directory.")
            
def main_analyze_not_matched_images(query_image_not_matched_paths):
    
    # Load all query images and their corresponding cropped torso images
    cropped_torso_paths_zoomed = [get_corresponding_torso_image(p, "zoomed_version", "_zoomed") for p in query_image_not_matched_paths]
    cropped_torso_paths = [get_corresponding_torso_image(p) for p in query_image_not_matched_paths]

    # Create a session state variable to track the index of the currently displayed query image pair
    if 'current_query_not_matched_index' not in st.session_state:
        st.session_state.current_query_not_matched_index = 0
        
    if 'curr_ref_not_matched_idx' not in st.session_state:
        st.session_state.curr_ref_not_matched_idx = 0
        
    if 'ref_img_paths_not_matched' not in st.session_state:
        st.session_state.ref_img_paths_not_matched = []
    
    if 'not_matched_recomms_rank_selected' not in st.session_state:
        st.session_state.not_matched_recomms_rank_selected  = 1
    
    # Create columns for layout
    col_right, col_ref, col_buttons = st.columns([3, 3, 1.5])
    
    with col_right:

        st.subheader('Not-matched Query Images')
        
        if len(query_image_not_matched_paths) > 0:
            
            # Progress bar
            st.write(f"{st.session_state.current_query_not_matched_index + 1} / {len(query_image_not_matched_paths)}")

            # Add button below the left column for the next query image
            if st.button('Next Query Image'):
                if st.session_state.current_query_not_matched_index == (len(query_image_not_matched_paths) - 1):
                    st.warning("You have looped over all query images once and saved human inputs if action items were pressed.")
                st.session_state.current_query_not_matched_index = (st.session_state.current_query_not_matched_index + 1) % len(query_image_not_matched_paths)
                st.session_state.curr_ref_not_matched_idx = 0
                st.session_state.not_matched_recomms_rank_selected  = 1
                
            # Add button below the left column for Previous query image
            if st.button('Previous Query Image'):
                if (st.session_state.current_query_not_matched_index - 1) >= 0:
                    st.session_state.current_query_not_matched_index -= 1
                else:
                    st.warning("Not enough remained data for this action.")
                st.session_state.curr_ref_not_matched_idx = 0
                st.session_state.not_matched_recomms_rank_selected  = 1
                
            # Add button below the left column to jump forward 100 images
            if st.button('Jump Forward 100 Images'):
                if (st.session_state.current_query_not_matched_index + 100) < len(query_image_not_matched_paths):
                    st.session_state.current_query_not_matched_index += 100
                    st.session_state.not_matched_recomms_rank_selected  = 1
                else:
                    st.warning("Not enough remained data for this action.")
                st.session_state.curr_ref_not_matched_idx = 0
            
            # Add button below the left column to jump forward 100 images
            if st.button('Jump Backward 100 Images'):
                if (st.session_state.current_query_not_matched_index - 100) >= 0:
                    st.session_state.current_query_not_matched_index -= 100
                    st.session_state.not_matched_recomms_rank_selected  = 1
                else:
                    st.warning("Not enough remained data for this action.")
                st.session_state.curr_ref_not_matched_idx = 0
                    
            # Get the current image paths based on the current index
            current_query_image = query_image_not_matched_paths[st.session_state.current_query_not_matched_index]
            current_cropped_torso_zoomed_image = cropped_torso_paths_zoomed[st.session_state.current_query_not_matched_index]
            current_cropped_torso_image = cropped_torso_paths[st.session_state.current_query_not_matched_index]
            
            # Display the left panel images
            display_left_query_images_not_matched_images(
                [current_query_image, current_cropped_torso_zoomed_image, current_cropped_torso_image], 
                ['Query Image: ' + os.path.basename(current_query_image),
                 'Cropped Torso Zoomed Image: ' + os.path.basename(current_cropped_torso_zoomed_image),
                 'Cropped Torso Image: ' + os.path.basename(current_cropped_torso_image)]
            )
            st.session_state.ref_img_paths_not_matched = update_reference_images(current_query_image, st.session_state.not_matched_recomms_rank_selected)
        else:
            st.warning("No query images found in the specified directory.")
    
    with col_ref:
    
        st.subheader('Matched Reference Images (Rejected)')

        num_rows, num_cols = 1, 1
        
        # Get the current query image for reference image retrieval
        if len(query_image_not_matched_paths) > 0:
            if len(st.session_state.ref_img_paths_not_matched) > 0:
                
                # Add button below the right image column for the next reference image
                if st.button('Next Matched Reference Image'):
                    
                    st.session_state.curr_ref_not_matched_idx += 1  # Move to the next set
                    
                    if st.session_state.curr_ref_not_matched_idx * num_rows * num_cols >= len(st.session_state.ref_img_paths_not_matched):
                        st.session_state.curr_ref_not_matched_idx = 0
                
                # Add button below the left column for previous reference image
                if st.button('Previous Matched Reference Image'):
                    if (st.session_state.curr_ref_not_matched_idx - 1) >= 0:
                        st.session_state.curr_ref_not_matched_idx -= 1
                    else:
                        st.warning("You have reached the first item.")
                
                # Add button below the left column to jump forward 10 images
                if st.button('Jump Forward 10 Images'):
                    if (st.session_state.curr_ref_not_matched_idx + 10) < len(st.session_state.ref_img_paths_not_matched):
                        st.session_state.curr_ref_not_matched_idx += 10
                    else:
                        st.warning("Not enough data available for this action.")
                
                # Add button below the left column to jump forward 10 images
                if st.button('Jump Backward 10 Images'):
                    if (st.session_state.curr_ref_not_matched_idx - 10) >= 0:
                        st.session_state.curr_ref_not_matched_idx -= 10
                    else:
                        st.warning("Not enough data available for this action.")
                        
                # Progress bar
                st.write(f"{st.session_state.curr_ref_not_matched_idx + 1} / {len(st.session_state.ref_img_paths_not_matched)}")

                # Display images
                starting_idx = st.session_state.curr_ref_not_matched_idx * num_rows * num_cols
                display_right_reference_image(st.session_state.ref_img_paths_not_matched, starting_idx, num_rows, num_cols)
            else:
                st.warning("No reference images found for the current query image.")
        else:
            st.warning("No query images found in the specified directory.")
        
    with col_buttons:
                
        st.subheader('Actions')
        
        if len(query_image_not_matched_paths) > 0:
            
            if st.button('Assign New ID'):
                update_human_input_single(current_query_image, 'AssignNewId')
                st.success(f"Assign New ID Clicked!")
                
                
            if st.button('Accept AI Matched ID'):
                human_input_value = 'AcceptId'
                update_human_input_single(current_query_image, human_input_value)
                overwrite_matching_results(current_query_image)
                st.success("Matched ID Overwritten!")
            
            
            if st.button('Skip This Query Image'):
                update_human_input_single(current_query_image, 'SkipImage')
                st.success("Skip This Query Image Clicked!")
                
                
            if st.button('Next Algorithmic Matched ID'):
                st.session_state.not_matched_recomms_rank_selected = max (1, (st.session_state.not_matched_recomms_rank_selected + 1) % (st.session_state.num_id_recomms + 1))
                st.session_state.curr_ref_not_matched_idx = 0  # Reset reference index for new reference image
                st.session_state.ref_img_paths_not_matched = update_reference_images(current_query_image, st.session_state.not_matched_recomms_rank_selected)
                st.success("Next Algorithmic Matched ID clicked! Rank: {}".format(st.session_state.not_matched_recomms_rank_selected))


            if st.button('Assign New ID to All'):
                human_input_value = 'AssignNewId'
                update_human_inputs_all(query_image_not_matched_paths, human_input_value)
                st.success(f"All Matched IDs Accepted!")
            
            
            # Show matching results
            st.subheader('Matching Results')
            matched_label, matched_serial, matched_dist, matched_mode, descriptors_size, ground_truth = get_matched_label(current_query_image, st.session_state.not_matched_recomms_rank_selected)
            display_custom_table(matched_label, matched_serial, matched_dist, matched_mode, descriptors_size, st.session_state.not_matched_recomms_rank_selected)

            # Show human input
            display_custom_table_human_input(current_query_image)
            
            # Show ground truth
            display_custom_table_ground_truth(ground_truth)


# Load new metadata if needed
initialize_vizualization_project()

# Display the dropdown menu with increased font size
st.markdown(f'<div class="radio-font-size">Select Analysis:</div>', unsafe_allow_html=True)
selected_option = st.selectbox("", ['Analyze Matched Images', 
                                    'Analyze Not Matched Images'],
                            key="dropdown_options")

# Add a styled horizontal line below the radio button section
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

# Run the corresponding function based on the selected option
if selected_option == 'Analyze Matched Images':

    if ('matching_results_table' in st.session_state) and ('matching_attempt' in st.session_state.matching_results_table.columns):
        
        query_image_paths = get_all_query_image_paths('matched')
        main_analyze_matched_images(query_image_paths)
    
    else:
        st.warning("Matching results not available or you need to start project from matched results tab.")
    
elif selected_option == 'Analyze Not Matched Images':
        
    if ('matching_results_table' in st.session_state) and ('matching_attempt' in st.session_state.matching_results_table.columns):
        query_image_paths = get_all_query_image_paths('not_matched')
        main_analyze_not_matched_images(query_image_paths)
    else:
        st.warning("Matching results not available or need to start project from matched results tab.")
    