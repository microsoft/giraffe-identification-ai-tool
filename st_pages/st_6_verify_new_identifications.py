# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_matching import pipeline_code_relative_dir
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
    
st.title("Verify Identification of Unknown Individuals")
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

def get_corresponding_torso_image(image_path, subdir_name="original_size", suffix=""):
    
    # find path for _cropped_torso_zoomed
    parts = image_path.rsplit(".", 1)
    img_filename = os.path.basename(image_path)
    cropped_torso_zoomed_image_dir = os.path.join(st.session_state.processed_img_dir, subdir_name, img_filename).replace("." + parts[1], "_cropped_torso" + suffix + "." + parts[1])
            
    return cropped_torso_zoomed_image_dir

def load_metadata_with_partitioning_results():
    # Load query metadata with partitioning results
    partition = 'query'
    metadata_filepath = os.path.join(st.session_state.root_dir, partition + '_dir', 'metadata_' + partition + '.csv')
    metadata_query = load_metadata_file(metadata_filepath)
    
    # Filter needed columns if they exist in dataframe
    needed_cols = ['#Serial', 'path_relative_to_root', 'descriptors_size', 'new_id_aligned_with_ref', 'human_input']
    missing_cols = [col for col in needed_cols if col not in metadata_query.columns]
    if missing_cols:
        st.write(f"WARNING! The following required columns are missing from metadata_query: {', '.join(missing_cols)}")
        st.stop()
    # Filter dataframe based on conditions
    filtered_df = metadata_query[
        (metadata_query['new_id_aligned_with_ref'].notna()) &
        (metadata_query['human_input'] == 'AssignNewId')]

    # Group by 'new_id_aligned_with_ref' column and extract each column into a list
    st.session_state.partitions = filtered_df.groupby('new_id_aligned_with_ref')[needed_cols].agg(list).to_dict('index')

def display_all_images_within_partition(partition_data):
    # Get data related to images from partition data
    query_image_paths = partition_data['path_relative_to_root']
    serial_numbers = partition_data['#Serial']
    descriptors_sizes = partition_data['descriptors_size']
    
    if 'AID2021' in partition_data:
        aid2021_numbers = partition_data['AID2021']
    else:
        aid2021_numbers = ['NA'] * len(serial_numbers)

    
    # Load all query images and their corresponding cropped torso images
    cropped_torso_paths_zoomed = [get_corresponding_torso_image(p, "zoomed_version", "_zoomed") for p in query_image_paths]
    
    
    num_images = len(query_image_paths)
    

    # Iterate over each image and display them
    for idx in range(num_images):
        
        # Create two columns for layout
        cols = st.columns(2)
        
        # Create a meaningful caption using the partition data
        serial = serial_numbers[idx]
        aid2021 = aid2021_numbers[idx]
        descriptor_size = descriptors_sizes[idx]
        caption = (f"Query Image {idx + 1} - Serial: {serial}, AID2021: {aid2021}, "
                   f"Descriptors Size: {descriptor_size}")
        
        # Load the original query image
        query_image = ImageOps.exif_transpose(Image.open(os.path.join(st.session_state.root_dir, query_image_paths[idx])))
        
        # Resize the query image so that the height is 256, maintaining the aspect ratio
        width, height = query_image.size
        # Check if the height is zero (avoid division by zero)
        if height > 0:
            new_height = 256
            new_width = int((new_height / height) * width)
            query_image = query_image.resize((new_width, new_height))
        
        # Load the cropped torso images
        cropped_torso_image_zoomed = Image.open(cropped_torso_paths_zoomed[idx])
        cropped_torso_image_zoomed = cropped_torso_image_zoomed.resize((256, 256))

        # Display original query image in the left column
        cols[0].image(query_image, caption=caption, use_container_width=False)
        cols[1].image(cropped_torso_image_zoomed, caption=f"Cropped Torso Zoomed {idx + 1}", use_container_width=False)

def main_analyze_partitioned_results():
    # Initialize session state variables if not already initialized
    if 'current_right_partition_index' not in st.session_state:
        st.session_state.current_right_partition_index = 0
    
    if 'current_left_partition_index' not in st.session_state:
        st.session_state.current_left_partition_index = 0

    if 'partitions' not in st.session_state:
        st.session_state.partitions = {}
    
    load_metadata_with_partitioning_results()

    # Create columns for layout
    col_left, col_right = st.columns([3, 3])
            
    with col_left:
        st.subheader('Partition - Left')
        
        # Ensure there is data in the partitions
        partition_keys_left = list(st.session_state.partitions.keys())
        if len(partition_keys_left) > 0:

            prev_col, next_col = st.columns([0.75, 1.5])
            with next_col:
                # Display "Show Next Partition" button to update the index and data
                if st.button('Next | Left'):
                    if st.session_state.current_left_partition_index + 1 < len(partition_keys_left):
                        st.session_state.current_left_partition_index += 1  
                    else:
                        st.warning("You have reached the last partition.")
            with prev_col:       
                # Display "Show Previous Partition" button to update the index and data
                if st.button('Previous | Left'):
                    if st.session_state.current_left_partition_index - 1 >= 0:
                        st.session_state.current_left_partition_index -= 1  
                    else:
                        st.warning("You have reached the first partition.")

            # Get data for partition
            st.session_state.current_partition_key_left = partition_keys_left[st.session_state.current_left_partition_index]
            partition_data_left = st.session_state.partitions[st.session_state.current_partition_key_left]
            
            # Progress bar showing which partition is being displayed
            st.write(f"Displaying partition # {int(st.session_state.current_partition_key_left)} | {st.session_state.current_left_partition_index + 1} of {len(partition_keys_left)}")
            
            # Display data for the current partition
            display_all_images_within_partition(partition_data_left)

        else:
            st.warning("No partitions found.")
    
    with col_right:
        st.subheader('Partition - Right')
        
        # Ensure there is data in the partitions
        partition_keys_right = list(st.session_state.partitions.keys())
        if len(partition_keys_right ) > 0:
            
            prev_col, next_col = st.columns([0.75, 1.5])
            with next_col:
                # Display "Show Next Partition" button to update the index and data
                if st.button('Next | Right'):
                    if st.session_state.current_right_partition_index + 1 < len(partition_keys_right):
                        st.session_state.current_right_partition_index += 1
                    else:
                        st.warning("You have reached the last partition.")
            
            with prev_col:
                # Display "Show Previous Partition" button to update the index and data
                if st.button('Previous | Right'):
                    if st.session_state.current_right_partition_index - 1 >= 0:
                        st.session_state.current_right_partition_index -= 1 
                    else:
                        st.warning("You have reached the first partition.")

            # Get data for partition
            st.session_state.current_partition_key_right = partition_keys_right [st.session_state.current_right_partition_index]
            partition_data_right = st.session_state.partitions[st.session_state.current_partition_key_right]
            
            # Progress bar showing which partition is being displayed
            st.write(f"Displaying partition # {int(st.session_state.current_partition_key_right)} |  {st.session_state.current_right_partition_index + 1} of {len(partition_keys_right)}")
            
            # Display data for the current partition
            display_all_images_within_partition(partition_data_right)

        else:
            st.warning("No partitions found.")


# Run the main function
pycode_name = 'step_4_partition_new_items.py'
main_analyze_partitioned_results()