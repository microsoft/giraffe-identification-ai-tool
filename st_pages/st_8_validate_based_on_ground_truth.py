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


st.markdown("""This page offers a detailed breakdown of accuracy results based on ground truth data for query elephant images.
            Additionally, you can visualize and analyze falsely matched results to gain deeper insights into identification errors and improve performance.""")


pipeline_code_dir = os.path.join(str(Path(__file__).resolve().parent.parent), str(pipeline_code_relative_dir))
st.session_state.root_dir, st.session_state.processed_img_dir = load_data_dirs()

def find_algorithm_mistakes(query_metadata):

    required_cols = ['individual_id', 'match_individual_1', 'out_of_sample', 'matching_status']
    known_matched_false, fp_table, fn_table = None, None, None

    if all(col in query_metadata.columns for col in required_cols):

        known_matched_false = query_metadata[
            (query_metadata['matching_status'] == 'matched') &
            (query_metadata['individual_id'] != query_metadata['match_individual_1'])
        ]
        fp_table = query_metadata[(query_metadata['out_of_sample'] == True) & (query_metadata['matching_status'] == 'matched')]
        fn_table = query_metadata[(query_metadata['out_of_sample'] == False) & (query_metadata['matching_status'] == 'not_matched')]

    else:
        st.write('query_metadata does not contain all required columns')

    return known_matched_false, fp_table, fn_table

def main_compute_matching_accuracy(pipeline_code_dir, image_path):
    col_buttons, col_info = st.columns([0.5, 1.5])

    with col_buttons:
        st.write("")
        if st.button("Compute Matching Accuracy Metrics"):
            with st.spinner("Running script..."):
                pycode_name = 'step_5_evaluate_matching_results.py'
                stdout, stderr = run_bash_script(pipeline_code_dir, pycode_name)
                acc_results_file = os.path.join(st.session_state.root_dir, 'query_dir', 'accuracy_results_full.csv')
                if not os.path.isfile(acc_results_file):
                    acc_results_file = os.path.join(st.session_state.root_dir, 'query_dir', 'accuracy_results.csv')
                if os.path.isfile(acc_results_file):
                    st.write(pd.read_csv(acc_results_file))

    with col_info:
        st.markdown("""
        ## Accuracy Metrics Overview

        We categorize known individuals as the **positive class** and unknown individuals as the **negative class**. The evaluation considers two key aspects:

        - **Re-identification Accuracy:** Measures how accurately known elephants are identified.
        - **Partitioning Accuracy:** Assesses how well unknown individuals are grouped into distinct clusters.

        ### Class-Specific Metrics
        - **Precision (Positive):** Proportion of correctly identified known elephants among all predicted known.
        - **Recall (Positive):** Proportion of correctly identified known elephants among all actual known.
        - **F1 Score (Positive):** Harmonic mean of precision and recall for known individuals.
        - **Precision (Negative):** Proportion of correctly identified unknown elephants among all predicted unknown.
        - **Recall (Negative):** Proportion of correctly identified unknown elephants among all actual unknown.
        - **F1 Score (Negative):** Harmonic mean of precision and recall for unknown individuals.

        ### WildFusion-Specific Metrics
        - **Top-1 / Top-3 Accuracy:** Fraction of queries where the correct individual appears in top-1 or top-3 ranked candidates.
        - **mAP@3:** Mean Average Precision at rank 3.
        - **ECE (Fused Score):** Expected Calibration Error of the fused similarity score.
        - **Same-view / Cross-view Top-1:** Top-1 accuracy split by whether query and candidate share the same viewpoint.

        ### Overall Performance Metrics
        - **Overall Accuracy:** Proportion of correctly classified individuals.
        - **Accuracy (Re-identified Items):** Accuracy for known elephants given ground truth labels.
        - **Adjusted Rand Index (Partitioning):** Clustering quality for unknown individuals.
        """)

        if os.path.isfile(image_path):
            image = Image.open(image_path)
            st.image(image, caption='')

def run_bash_script(code_directory, pycode_name):
    result = subprocess.run(
        ["bash", "./setup_pipeline.sh", os.path.join(code_directory, pycode_name)],
        capture_output=True, text=True
    )

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

    st.session_state.curr_query_idx_known_matched_false = 0
    st.session_state.curr_ref_idx_known_matched_false = 0
    st.session_state.curr_ref_idx_gt_known_matched_false = 0
    st.session_state.ref_img_paths_known_matched_false = []
    st.session_state.ref_img_paths_gt_known_matched_false = []
    st.session_state.recomms_rank_selected_known_matched_false = 1

    st.session_state.curr_query_idx_fn_table = 0
    st.session_state.curr_ref_idx_fn_table = 0
    st.session_state.curr_ref_idx_gt_fn_table = 0
    st.session_state.ref_img_paths_fn_table = []
    st.session_state.ref_img_paths_gt_fn_table = []
    st.session_state.recomms_rank_selected_fn_table = 1

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

    st.session_state.metadata_table = st.session_state.metadata['reference'].copy()
    st.session_state.matching_results_table = st.session_state.metadata['query'].copy()

def get_human_input_single(key_query):
    base_names = st.session_state.matching_results_table['path_relative_to_root'].apply(os.path.basename)
    mask = base_names == os.path.basename(key_query)
    if 'human_input' in st.session_state.matching_results_table.columns and len(st.session_state.matching_results_table.loc[mask, 'human_input']) != 0 and not st.session_state.matching_results_table.loc[mask, 'human_input'].isnull().all():
        return st.session_state.matching_results_table.loc[mask, 'human_input']
    else:
        return None

def get_corresponding_torso_image(image_path, subdir_name="original_size", suffix=""):
    parts = image_path.rsplit(".", 1)
    img_filename = os.path.basename(image_path)
    cropped_torso_zoomed_image_dir = os.path.join(st.session_state.processed_img_dir, subdir_name, img_filename).replace("." + parts[1], "_cropped_torso" + suffix + "." + parts[1])
    return cropped_torso_zoomed_image_dir

def display_left_query_images_matched_images(image_paths, captions):
    col = st.columns(1)[0]
    for i in [1, 0]:
        image_path = image_paths[i]
        if not os.path.isfile(image_path):
            col.warning(f"Image not found: {os.path.basename(image_path)}")
            continue
        img = ImageOps.exif_transpose(Image.open(image_path))
        if i == 0:
            scale_factor = 0.15
            img_resized = img.resize(
                (int(img.width * scale_factor), int(img.height * scale_factor))
            )
            col.image(img_resized, caption=captions[i], use_container_width=True)
        else:
            img_resized = img.resize((256, 256))
            col.image(img_resized, caption='Cropped Elephant Image', use_container_width=True)

def display_right_reference_image(data, starting_idx, num_rows, num_cols):
    image_paths = data[starting_idx:]
    for i in range(num_rows):
        cols = st.columns(num_cols)
        for j, col in enumerate(cols):
            index = i * num_cols + j
            if index < len(image_paths):
                image_path = image_paths[index]

                torso_img_path = get_corresponding_torso_image(image_path, "zoomed_version", "_zoomed")
                if os.path.isfile(torso_img_path):
                    torso_img = ImageOps.exif_transpose(Image.open(torso_img_path))
                    torso_img = torso_img.resize((256, 256))
                    col.image(torso_img, caption='Cropped Elephant Image', use_container_width=True)
                else:
                    col.warning("Crop not available")

                if not os.path.isfile(image_path):
                    col.warning(f"Image not found: {os.path.basename(image_path)}")
                    continue
                actual_img = ImageOps.exif_transpose(Image.open(image_path))
                actual_img_resized = actual_img.resize((int(actual_img.width * 0.15), int(actual_img.height * 0.15)))
                actual_image_caption = os.path.basename(image_path)
                col.image(actual_img_resized, caption=f"Reference Image: {actual_image_caption}", use_container_width=True)

def display_custom_table(assigned_id, image_id_matched, fused_sim, global_sim, local_count, recom_rank):
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
            <td class="header">Matched Image ID</td>
            <td>{image_id_matched}</td>
        </tr>
        <tr>
            <td class="header">Fused Score (higher=better)</td>
            <td>{fused_sim}</td>
        </tr>
        <tr>
            <td class="header">Global Similarity</td>
            <td>{global_sim}</td>
        </tr>
        <tr>
            <td class="header">Local Inlier Count</td>
            <td>{local_count}</td>
        </tr>
    </table>
    """
    st.html(table_html)

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
    st.html(table_html)

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
    st.html(table_html)

def get_matched_label(current_query_image, recom_rank):
    df = st.session_state.matching_results_table
    current_image_name = os.path.basename(current_query_image)
    matching_row = df[df['path_relative_to_root'].apply(lambda x: os.path.basename(x)) == current_image_name]

    if not matching_row.empty:
        ground_truth = None
        if 'individual_id' in list(matching_row.columns):
            val = matching_row['individual_id'].iloc[0]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                ground_truth = val

        matched_individual = matching_row[f'match_individual_{recom_rank}'].iloc[0] if f'match_individual_{recom_rank}' in matching_row.columns else None
        matched_image      = matching_row[f'match_image_{recom_rank}'].iloc[0]      if f'match_image_{recom_rank}' in matching_row.columns else None
        fused_sim   = matching_row[f'match_fused_sim_{recom_rank}'].iloc[0]   if f'match_fused_sim_{recom_rank}'   in matching_row.columns else None
        global_sim  = matching_row[f'match_global_sim_{recom_rank}'].iloc[0]  if f'match_global_sim_{recom_rank}'  in matching_row.columns else None
        local_count = matching_row[f'match_local_count_{recom_rank}'].iloc[0] if f'match_local_count_{recom_rank}' in matching_row.columns else None

        fused_sim  = float("{:.4f}".format(fused_sim))  if fused_sim  is not None and not (isinstance(fused_sim, float)  and np.isnan(fused_sim))  else fused_sim
        global_sim = float("{:.4f}".format(global_sim)) if global_sim is not None and not (isinstance(global_sim, float) and np.isnan(global_sim)) else global_sim

        return matched_individual, matched_image, fused_sim, global_sim, local_count, ground_truth
    else:
        return None, None, None, None, None, None

def get_paths_by_matched_label_id(matched_label, matched_image_id):
    df = st.session_state.metadata_table
    base_dir = st.session_state.root_dir

    matching_rows = df[df['individual_id'] == matched_label]
    matching_row_exact = df[df['image_id'] == matched_image_id] if matched_image_id is not None else df.iloc[:0]

    exact_path = None
    if not matching_row_exact.empty:
        exact_path = os.path.join(base_dir, matching_row_exact.iloc[0]['path_relative_to_root'])

    matching_rows = matching_rows[matching_rows['image_id'] != matched_image_id] if matched_image_id is not None else matching_rows

    remaining_paths = [os.path.join(base_dir, path) for path in matching_rows['path_relative_to_root'].tolist()]

    if exact_path is not None:
        all_paths = [exact_path] + remaining_paths
    else:
        all_paths = remaining_paths

    return all_paths

def update_reference_images(current_query_image, recom_rank):
    matched_label, matched_image, _, _, _, _ = get_matched_label(current_query_image, recom_rank)
    reference_image_paths = get_paths_by_matched_label_id(matched_label, matched_image)
    return reference_image_paths

def update_reference_images_ground_truth(current_query_image, recom_rank):
    _, _, _, _, _, ground_truth = get_matched_label(current_query_image, recom_rank)
    reference_image_paths = get_paths_by_matched_label_id(ground_truth, None)
    return reference_image_paths

def update_human_input_single(key_query, human_input_value):
    base_names = st.session_state.matching_results_table['path_relative_to_root'].apply(os.path.basename)
    mask = base_names == os.path.basename(key_query)
    st.session_state.matching_results_table.loc[mask, 'human_input'] = human_input_value

def update_human_inputs_all(query_image_paths, human_input_value):
    df = st.session_state.matching_results_table
    df['filename'] = df['path_relative_to_root'].apply(os.path.basename)
    query_basenames = [os.path.basename(query_item) for query_item in query_image_paths]
    mask = df['filename'].isin(query_basenames)
    df.loc[mask, 'human_input'] = human_input_value

def main_visualize_known_matched_false():

    st.session_state.known_matched_false, st.session_state.fp_table, st.session_state.fn_table = \
        find_algorithm_mistakes(st.session_state.matching_results_table)

    if st.session_state.known_matched_false is None:
        st.warning("Table for falsely matched in-sample items is empty. Try re-loading.")
    else:
        query_image_paths = list(os.path.join(st.session_state.root_dir, x) for x in st.session_state.known_matched_false['path_relative_to_root'])
        cropped_torso_paths_zoomed = [get_corresponding_torso_image(p, "zoomed_version", "_zoomed") for p in query_image_paths]
        cropped_torso_paths = [get_corresponding_torso_image(p) for p in query_image_paths]

        if 'curr_query_idx_known_matched_false' not in st.session_state:
            st.session_state.curr_query_idx_known_matched_false = 0
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

        matched_label = None

        col_query_img, col_predicted_match, col_ground_truth, col_info = st.columns([2, 2, 2, 1.5])

        with col_query_img:
            st.markdown("<h2 style='font-size:20px;'>Matched Query Elephant Images</h2>", unsafe_allow_html=True)

            if len(query_image_paths) > 0:

                if st.button('Next Query Image'):
                    st.session_state.recomms_rank_selected_known_matched_false = 1
                    if st.session_state.curr_query_idx_known_matched_false == (len(query_image_paths) - 1):
                        st.warning("You have looped over all query images once.")
                    st.session_state.curr_query_idx_known_matched_false = (st.session_state.curr_query_idx_known_matched_false + 1) % len(query_image_paths)
                    st.session_state.curr_ref_idx_known_matched_false = 0
                    st.session_state.curr_ref_idx_gt_known_matched_false = 0

                if st.button('Previous Query Image'):
                    if (st.session_state.curr_query_idx_known_matched_false - 1) >= 0:
                        st.session_state.recomms_rank_selected_known_matched_false = 1
                        st.session_state.curr_query_idx_known_matched_false -= 1
                        st.session_state.curr_ref_idx_known_matched_false = 0
                        st.session_state.curr_ref_idx_gt_known_matched_false = 0
                    else:
                        st.warning("Not enough remained data for this action.")

                st.write(f"{st.session_state.curr_query_idx_known_matched_false + 1} / {len(query_image_paths)}")

                current_query_image = query_image_paths[st.session_state.curr_query_idx_known_matched_false]
                current_cropped_torso_zoomed_image = cropped_torso_paths_zoomed[st.session_state.curr_query_idx_known_matched_false]
                current_cropped_torso_image = cropped_torso_paths[st.session_state.curr_query_idx_known_matched_false]

                display_left_query_images_matched_images(
                    [current_query_image, current_cropped_torso_zoomed_image, current_cropped_torso_image],
                    ['Query Image: ' + os.path.basename(current_query_image),
                     'Cropped Elephant Zoomed Image: ' + os.path.basename(current_cropped_torso_zoomed_image),
                     'Cropped Elephant Image: ' + os.path.basename(current_cropped_torso_image)]
                )
                st.session_state.ref_img_paths_known_matched_false = update_reference_images(current_query_image, st.session_state.recomms_rank_selected_known_matched_false)
                st.session_state.ref_img_paths_gt_known_matched_false = update_reference_images_ground_truth(current_query_image, st.session_state.recomms_rank_selected_known_matched_false)
            else:
                st.warning("No query images found in the specified directory.")

        with col_info:
            st.markdown("<h2 style='font-size:20px;'>Action</h2>", unsafe_allow_html=True)

            if len(query_image_paths) > 0:

                if st.button('Next Algorithmic Matched ID'):
                    st.session_state.recomms_rank_selected_known_matched_false = max(1, (st.session_state.recomms_rank_selected_known_matched_false + 1) % (st.session_state.num_id_recomms + 1))
                    st.session_state.curr_ref_idx_known_matched_false = 0
                    st.session_state.curr_ref_idx_gt_known_matched_false = 0
                    st.session_state.ref_img_paths_known_matched_false = update_reference_images(current_query_image, st.session_state.recomms_rank_selected_known_matched_false)
                    st.session_state.ref_img_paths_gt_known_matched_false = update_reference_images_ground_truth(current_query_image, st.session_state.recomms_rank_selected_known_matched_false)
                    st.success("Next Algorithmic Matched ID clicked! Rank: {}".format(st.session_state.recomms_rank_selected_known_matched_false))

                st.markdown("<h2 style='font-size:20px;'>Matching Results</h2>", unsafe_allow_html=True)
                matched_individual, matched_image, fused_sim, global_sim, local_count, ground_truth = get_matched_label(current_query_image, st.session_state.recomms_rank_selected_known_matched_false)
                display_custom_table(matched_individual, matched_image, fused_sim, global_sim, local_count, st.session_state.recomms_rank_selected_known_matched_false)
                display_custom_table_ground_truth(ground_truth)

        with col_predicted_match:
            st.markdown("<h2 style='font-size:20px;'>Matched Reference Images</h2>", unsafe_allow_html=True)

            num_rows, num_cols = 1, 1

            if len(query_image_paths) > 0:
                if len(st.session_state.ref_img_paths_known_matched_false) > 0:

                    if st.button('Next Matched Reference Image'):
                        st.session_state.curr_ref_idx_known_matched_false += 1
                        if st.session_state.curr_ref_idx_known_matched_false * num_rows * num_cols >= len(st.session_state.ref_img_paths_known_matched_false):
                            st.session_state.curr_ref_idx_known_matched_false = 0

                    if st.button('Previous Matched Reference Image'):
                        if (st.session_state.curr_ref_idx_known_matched_false - 1) >= 0:
                            st.session_state.curr_ref_idx_known_matched_false -= 1
                        else:
                            st.warning("You have reached the first item.")

                    st.write(f"{st.session_state.curr_ref_idx_known_matched_false + 1} / {len(st.session_state.ref_img_paths_known_matched_false)}")

                    starting_idx = st.session_state.curr_ref_idx_known_matched_false * num_rows * num_cols
                    display_right_reference_image(st.session_state.ref_img_paths_known_matched_false, starting_idx, num_rows, num_cols)
                else:
                    st.warning("No reference images found for the current query image.")
            else:
                st.warning("No query images found in the specified directory.")

        with col_ground_truth:
            st.markdown("<h2 style='font-size:20px;'>Ground Truth</h2>", unsafe_allow_html=True)

            num_rows, num_cols = 1, 1

            if len(query_image_paths) > 0:
                if len(st.session_state.ref_img_paths_gt_known_matched_false) > 0:

                    if st.button('Next Ground Truth Reference Image'):
                        st.session_state.curr_ref_idx_gt_known_matched_false += 1
                        if st.session_state.curr_ref_idx_gt_known_matched_false * num_rows * num_cols >= len(st.session_state.ref_img_paths_gt_known_matched_false):
                            st.session_state.curr_ref_idx_gt_known_matched_false = 0

                    if st.button('Previous Ground Truth Reference Image'):
                        if (st.session_state.curr_ref_idx_gt_known_matched_false - 1) >= 0:
                            st.session_state.curr_ref_idx_gt_known_matched_false -= 1
                        else:
                            st.warning("You have reached the first item.")

                    st.write(f"{st.session_state.curr_ref_idx_gt_known_matched_false + 1} / {len(st.session_state.ref_img_paths_gt_known_matched_false)}")

                    starting_idx = st.session_state.curr_ref_idx_gt_known_matched_false * num_rows * num_cols
                    display_right_reference_image(st.session_state.ref_img_paths_gt_known_matched_false, starting_idx, num_rows, num_cols)
                else:
                    st.warning("No reference images found for the current query image.")
            else:
                st.warning("No query images found in the specified directory.")

def main_visualize_fn_table():

    st.session_state.known_matched_false, st.session_state.fp_table, st.session_state.fn_table = \
        find_algorithm_mistakes(st.session_state.matching_results_table)

    if st.session_state.fn_table is None:
        st.warning("Table for falsely unmatched in-sample items is empty. Try re-loading.")
    else:
        query_image_paths = list(os.path.join(st.session_state.root_dir, x) for x in st.session_state.fn_table['path_relative_to_root'])
        cropped_torso_paths_zoomed = [get_corresponding_torso_image(p, "zoomed_version", "_zoomed") for p in query_image_paths]
        cropped_torso_paths = [get_corresponding_torso_image(p) for p in query_image_paths]

        if 'curr_query_idx_fn_table' not in st.session_state:
            st.session_state.curr_query_idx_fn_table = 0
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

        matched_label = None
        col_query_img, col_predicted_match, col_ground_truth, col_info = st.columns([2, 2, 2, 1.5])

        with col_query_img:
            st.markdown("<h2 style='font-size:20px;'>Matched Query Elephant Images</h2>", unsafe_allow_html=True)

            if len(query_image_paths) > 0:

                if st.button('Next Query Image'):
                    st.session_state.recomms_rank_selected_fn_table = 1
                    if st.session_state.curr_query_idx_fn_table == (len(query_image_paths) - 1):
                        st.warning("You have looped over all query images once.")
                    st.session_state.curr_query_idx_fn_table = (st.session_state.curr_query_idx_fn_table + 1) % len(query_image_paths)
                    st.session_state.curr_ref_idx_fn_table = 0
                    st.session_state.curr_ref_idx_gt_fn_table = 0

                if st.button('Previous Query Image'):
                    if (st.session_state.curr_query_idx_fn_table - 1) >= 0:
                        st.session_state.recomms_rank_selected_fn_table = 1
                        st.session_state.curr_query_idx_fn_table -= 1
                        st.session_state.curr_ref_idx_fn_table = 0
                        st.session_state.curr_ref_idx_gt_fn_table = 0
                    else:
                        st.warning("Not enough remained data for this action.")

                st.write(f"{st.session_state.curr_query_idx_fn_table + 1} / {len(query_image_paths)}")

                current_query_image = query_image_paths[st.session_state.curr_query_idx_fn_table]
                current_cropped_torso_zoomed_image = cropped_torso_paths_zoomed[st.session_state.curr_query_idx_fn_table]
                current_cropped_torso_image = cropped_torso_paths[st.session_state.curr_query_idx_fn_table]

                display_left_query_images_matched_images(
                    [current_query_image, current_cropped_torso_zoomed_image, current_cropped_torso_image],
                    ['Query Image: ' + os.path.basename(current_query_image),
                     'Cropped Elephant Zoomed Image: ' + os.path.basename(current_cropped_torso_zoomed_image),
                     'Cropped Elephant Image: ' + os.path.basename(current_cropped_torso_image)]
                )
                st.session_state.ref_img_paths_fn_table = update_reference_images(current_query_image, st.session_state.recomms_rank_selected_fn_table)
                st.session_state.ref_img_paths_gt_fn_table = update_reference_images_ground_truth(current_query_image, st.session_state.recomms_rank_selected_fn_table)
            else:
                st.warning("No query images found in the specified directory.")

        with col_info:
            st.markdown("<h2 style='font-size:20px;'>Actions</h2>", unsafe_allow_html=True)

            if len(query_image_paths) > 0:

                if st.button('Next Algorithmic Matched ID'):
                    st.session_state.recomms_rank_selected_fn_table = max(1, (st.session_state.recomms_rank_selected_fn_table + 1) % (st.session_state.num_id_recomms + 1))
                    st.session_state.curr_ref_idx_fn_table = 0
                    st.session_state.curr_ref_idx_gt_fn_table = 0
                    st.session_state.ref_img_paths_fn_table = update_reference_images(current_query_image, st.session_state.recomms_rank_selected_fn_table)
                    st.session_state.ref_img_paths_gt_fn_table = update_reference_images_ground_truth(current_query_image, st.session_state.recomms_rank_selected_fn_table)
                    st.success("Next Algorithmic Matched ID clicked! Rank: {}".format(st.session_state.recomms_rank_selected_fn_table))

                st.markdown("<h2 style='font-size:20px;'>Matching Results</h2>", unsafe_allow_html=True)
                matched_individual, matched_image, fused_sim, global_sim, local_count, ground_truth = get_matched_label(current_query_image, st.session_state.recomms_rank_selected_fn_table)
                display_custom_table(matched_individual, matched_image, fused_sim, global_sim, local_count, st.session_state.recomms_rank_selected_fn_table)
                display_custom_table_ground_truth(ground_truth)

        with col_predicted_match:
            st.markdown("<h2 style='font-size:20px;'>Matched Reference (Rejected)</h2>", unsafe_allow_html=True)

            num_rows, num_cols = 1, 1

            if len(query_image_paths) > 0:
                if len(st.session_state.ref_img_paths_fn_table) > 0:

                    if st.button('Next Matched Reference Image'):
                        st.session_state.curr_ref_idx_fn_table += 1
                        if st.session_state.curr_ref_idx_fn_table * num_rows * num_cols >= len(st.session_state.ref_img_paths_fn_table):
                            st.session_state.curr_ref_idx_fn_table = 0

                    if st.button('Previous Matched Reference Image'):
                        if (st.session_state.curr_ref_idx_fn_table - 1) >= 0:
                            st.session_state.curr_ref_idx_fn_table -= 1
                        else:
                            st.warning("You have reached the first item.")

                    st.write(f"{st.session_state.curr_ref_idx_fn_table + 1} / {len(st.session_state.ref_img_paths_fn_table)}")

                    starting_idx = st.session_state.curr_ref_idx_fn_table * num_rows * num_cols
                    display_right_reference_image(st.session_state.ref_img_paths_fn_table, starting_idx, num_rows, num_cols)
                else:
                    st.warning("No reference images found for the current query image.")
            else:
                st.warning("No query images found in the specified directory.")

        with col_ground_truth:
            st.markdown("<h2 style='font-size:20px;'>Ground Truth</h2>", unsafe_allow_html=True)

            num_rows, num_cols = 1, 1

            if len(query_image_paths) > 0:
                if len(st.session_state.ref_img_paths_gt_fn_table) > 0:

                    if st.button('Next Ground Truth Reference Image'):
                        st.session_state.curr_ref_idx_gt_fn_table += 1
                        if st.session_state.curr_ref_idx_gt_fn_table * num_rows * num_cols >= len(st.session_state.ref_img_paths_gt_fn_table):
                            st.session_state.curr_ref_idx_gt_fn_table = 0

                    if st.button('Previous Ground Truth Reference Image'):
                        if (st.session_state.curr_ref_idx_gt_fn_table - 1) >= 0:
                            st.session_state.curr_ref_idx_gt_fn_table -= 1
                        else:
                            st.warning("You have reached the first item.")

                    st.write(f"{st.session_state.curr_ref_idx_gt_fn_table + 1} / {len(st.session_state.ref_img_paths_gt_fn_table)}")

                    starting_idx = st.session_state.curr_ref_idx_gt_fn_table * num_rows * num_cols
                    display_right_reference_image(st.session_state.ref_img_paths_gt_fn_table, starting_idx, num_rows, num_cols)
                else:
                    st.warning("No reference images found for the current query image.")
            else:
                st.warning("No query images found in the specified directory.")

def main_visualize_unknown_matched_false():

    st.session_state.known_matched_false, st.session_state.fp_table, st.session_state.fn_table = \
        find_algorithm_mistakes(st.session_state.matching_results_table)

    if st.session_state.fp_table is None:
        st.warning("Table for falsely matched out-of-sample items is empty. Try re-loading.")
    else:
        query_image_paths = list(os.path.join(st.session_state.root_dir, x) for x in st.session_state.fp_table['path_relative_to_root'])
        cropped_torso_paths_zoomed = [get_corresponding_torso_image(p, "zoomed_version", "_zoomed") for p in query_image_paths]
        cropped_torso_paths = [get_corresponding_torso_image(p) for p in query_image_paths]

        if 'curr_query_idx_fp_table' not in st.session_state:
            st.session_state.curr_query_idx_fp_table = 0
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

        matched_label = None
        col_query_img, col_predicted_match, col_info = st.columns([2, 2, 1.5])

        with col_query_img:
            st.markdown("<h2 style='font-size:20px;'>Matched Query Elephant Images</h2>", unsafe_allow_html=True)

            if len(query_image_paths) > 0:

                if st.button('Next Query Image'):
                    st.session_state.recomms_rank_selected_fp_table = 1
                    if st.session_state.curr_query_idx_fp_table == (len(query_image_paths) - 1):
                        st.warning("You have looped over all query images once.")
                    st.session_state.curr_query_idx_fp_table = (st.session_state.curr_query_idx_fp_table + 1) % len(query_image_paths)
                    st.session_state.curr_ref_idx_fp_table = 0
                    st.session_state.curr_ref_idx_gt_fp_table = 0

                if st.button('Previous Query Image'):
                    if (st.session_state.curr_query_idx_fp_table - 1) >= 0:
                        st.session_state.recomms_rank_selected_fp_table = 1
                        st.session_state.curr_query_idx_fp_table -= 1
                        st.session_state.curr_ref_idx_fp_table = 0
                        st.session_state.curr_ref_idx_gt_fp_table = 0
                    else:
                        st.warning("Not enough remained data for this action.")

                st.write(f"{st.session_state.curr_query_idx_fp_table + 1} / {len(query_image_paths)}")

                current_query_image = query_image_paths[st.session_state.curr_query_idx_fp_table]
                current_cropped_torso_zoomed_image = cropped_torso_paths_zoomed[st.session_state.curr_query_idx_fp_table]
                current_cropped_torso_image = cropped_torso_paths[st.session_state.curr_query_idx_fp_table]

                display_left_query_images_matched_images(
                    [current_query_image, current_cropped_torso_zoomed_image, current_cropped_torso_image],
                    ['Query Image: ' + os.path.basename(current_query_image),
                     'Cropped Elephant Zoomed Image: ' + os.path.basename(current_cropped_torso_zoomed_image),
                     'Cropped Elephant Image: ' + os.path.basename(current_cropped_torso_image)]
                )
                st.session_state.ref_img_paths_fp_table = update_reference_images(current_query_image, st.session_state.recomms_rank_selected_fp_table)
                st.session_state.ref_img_paths_gt_fp_table = update_reference_images_ground_truth(current_query_image, st.session_state.recomms_rank_selected_fp_table)
            else:
                st.warning("No query images found in the specified directory.")

        with col_info:
            st.markdown("<h2 style='font-size:20px;'>Action</h2>", unsafe_allow_html=True)

            if len(query_image_paths) > 0:

                if st.button('Next Algorithmic Matched ID'):
                    st.session_state.recomms_rank_selected_fp_table = max(1, (st.session_state.recomms_rank_selected_fp_table + 1) % (st.session_state.num_id_recomms + 1))
                    st.session_state.curr_ref_idx_fp_table = 0
                    st.session_state.curr_ref_idx_gt_fp_table = 0
                    st.session_state.ref_img_paths_fp_table = update_reference_images(current_query_image, st.session_state.recomms_rank_selected_fp_table)
                    st.session_state.ref_img_paths_gt_fp_table = update_reference_images_ground_truth(current_query_image, st.session_state.recomms_rank_selected_fp_table)
                    st.success("Next Algorithmic Matched ID clicked! Rank: {}".format(st.session_state.recomms_rank_selected_fp_table))

                st.markdown("<h2 style='font-size:20px;'>Matching Results</h2>", unsafe_allow_html=True)
                matched_individual, matched_image, fused_sim, global_sim, local_count, ground_truth = get_matched_label(current_query_image, st.session_state.recomms_rank_selected_fp_table)
                display_custom_table(matched_individual, matched_image, fused_sim, global_sim, local_count, st.session_state.recomms_rank_selected_fp_table)
                display_custom_table_ground_truth(ground_truth)

        with col_predicted_match:
            st.markdown("<h2 style='font-size:20px;'>Matched Reference (False)</h2>", unsafe_allow_html=True)

            num_rows, num_cols = 1, 1

            if len(query_image_paths) > 0:
                if len(st.session_state.ref_img_paths_fp_table) > 0:

                    if st.button('Next Matched Reference Image'):
                        st.session_state.curr_ref_idx_fp_table += 1
                        if st.session_state.curr_ref_idx_fp_table * num_rows * num_cols >= len(st.session_state.ref_img_paths_fp_table):
                            st.session_state.curr_ref_idx_fp_table = 0

                    if st.button('Previous Matched Reference Image'):
                        if (st.session_state.curr_ref_idx_fp_table - 1) >= 0:
                            st.session_state.curr_ref_idx_fp_table -= 1
                        else:
                            st.warning("You have reached the first item.")

                    st.write(f"{st.session_state.curr_ref_idx_fp_table + 1} / {len(st.session_state.ref_img_paths_fp_table)}")

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

    initialize_vizualization_project_validation()

    if ('matching_results_table' in st.session_state) and ('matching_attempt' in st.session_state.matching_results_table.columns):
        main_visualize_known_matched_false()
    else:
        st.warning("Matching results not available or you need to start project from matched results tab.")

elif selected_option == 'Visualize Falsely Matched Images (out-of-sample)':

    initialize_vizualization_project_validation()

    if ('matching_results_table' in st.session_state) and ('matching_attempt' in st.session_state.matching_results_table.columns):
        main_visualize_unknown_matched_false()
    else:
        st.warning("Matching results not available or you need to start project from matched results tab.")

elif selected_option == 'Visualize Falsely Unmatched Images (in-sample)':

    initialize_vizualization_project_validation()

    if ('matching_results_table' in st.session_state) and ('matching_attempt' in st.session_state.matching_results_table.columns):
        main_visualize_fn_table()
    else:
        st.warning("Matching results not available or you need to start project from matched results tab.")
