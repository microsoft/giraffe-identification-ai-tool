# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import io
import os
import sys
import json
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_matching import pipeline_code_relative_dir
from utils.helpers_matching import load_data_dirs, load_metadata_file

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

st.title("Review & Revise Elephant Re-identification Results")
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
    query_metadata = query_metadata.sort_values(by='match_local_count_1', ascending=False)
    selected_items = list(os.path.join(st.session_state.root_dir, x) for x in query_metadata.loc[query_metadata['matching_status'] == matching_status_key, 'path_relative_to_root'])
    return selected_items

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

def initialize_vizualization_project():
    st.session_state.metadata = {}
    for partition in ['query', 'reference']:
        metadata_filepath = os.path.join(st.session_state.root_dir, partition + '_dir', 'metadata_' + partition + '.csv')
        st.session_state.metadata[partition] = load_metadata_file(metadata_filepath)

    st.markdown(f'<div class="radio-font-size">Load Project:</div>', unsafe_allow_html=True)
    st.text(
        "How do you want to proceed? "
        "If Resume, you will only lose human inputs entered and not saved in this session. "
        "If Restart, you will lose human inputs entered and not saved in this session as well as any input previously saved in query metadata. "
        "Once selected choose Lock the Choice to continue reviewing for both sets of results."
    )
    response = st.radio(
        "",
        options=["Lock the Choice", "Resume Project", "Start New Project"]
    )
    if response == "Start New Project":
        if st.button("Confirm"):
            st.session_state.metadata_table = st.session_state.metadata['reference'].copy()
            st.session_state.matching_results_table = st.session_state.metadata['query'].copy()
            reset_state_vars()
            if 'human_input' in st.session_state.matching_results_table.columns:
                st.session_state.matching_results_table['human_input'] = None
            st.success("Matching data updated and human_input column erased.")

    elif response == "Resume Project":
        if st.button("Confirm"):
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
    if 'human_input' in st.session_state.matching_results_table.columns and len(st.session_state.matching_results_table.loc[mask, 'human_input']) != 0 and not st.session_state.matching_results_table.loc[mask, 'human_input'].isnull().all():
        return st.session_state.matching_results_table.loc[mask, 'human_input']
    else:
        return None

def display_left_query_images_matched_images(image_paths, captions):
    col = st.columns(1)[0]
    for i in [1, 0]:
        image_path = image_paths[i]
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file '{image_path}' not found.")
        img = ImageOps.exif_transpose(Image.open(image_path))
        if i == 0:
            scale_factor = 0.15
            img_resized = img.resize(
                (int(img.width * scale_factor), int(img.height * scale_factor))
            )
            col.image(img_resized, caption=captions[i], use_container_width=False)
        else:
            img_resized = img.resize((256, 256))
            col.image(img_resized, caption='Cropped Elephant Image', use_container_width=True)

def display_left_query_images_not_matched_images(image_paths, captions):
    cols = st.columns(1)[0]
    for i in [1, 0]:
        if i == 1:
            image_path = image_paths[i]
            if not os.path.isfile(image_path):
                raise FileNotFoundError("Image file '{}' not found.".format(image_path))
            img = ImageOps.exif_transpose(Image.open(image_path))
            img_resized = img.resize((256, 256))
            cols.image(img_resized, caption='Cropped Elephant Image', use_container_width=True)
        elif i == 0:
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

                torso_img_path = get_corresponding_torso_image(image_path, "zoomed_version", "_zoomed")
                if not os.path.isfile(torso_img_path):
                    raise FileNotFoundError("Image file '{}' not found.".format(torso_img_path))
                torso_img = ImageOps.exif_transpose(Image.open(torso_img_path))
                torso_img = torso_img.resize((256, 256))
                col.image(torso_img, caption="Cropped Elephant Image", use_container_width=True)

                if not os.path.isfile(image_path):
                    raise FileNotFoundError("Image file '{}' not found.".format(image_path))
                actual_img = ImageOps.exif_transpose(Image.open(image_path))
                actual_img_resized = actual_img.resize((int(actual_img.width * 0.15), int(actual_img.height * 0.15)))
                actual_image_caption = os.path.basename(image_path)
                col.image(actual_img_resized, caption=f"Reference Image: {actual_image_caption}", use_container_width=True)


def render_keypoint_overlay(query_img, ref_img, query_kpts, ref_kpts, matches):
    """
    Renders a side-by-side image with lines connecting matched keypoints.
    Returns image bytes (PNG) suitable for st.image().
    Lines are drawn from query keypoint coordinates to shifted ref keypoint coordinates
    so both images sit on a shared canvas.
    """
    q_arr = np.array(query_img.convert("RGB"))
    r_arr = np.array(ref_img.convert("RGB"))

    q_h, q_w = q_arr.shape[:2]
    r_h, r_w = r_arr.shape[:2]
    canvas_h = max(q_h, r_h)
    canvas_w = q_w + r_w

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:q_h, :q_w] = q_arr
    canvas[:r_h, q_w:q_w + r_w] = r_arr

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(canvas)
    ax.axis("off")

    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(matches), 1)))
    for (qi, ri), color in zip(matches, colors):
        qx, qy = query_kpts[qi]
        rx, ry = ref_kpts[ri]
        rx_shifted = rx + q_w
        ax.plot([qx, rx_shifted], [qy, ry], color=color, linewidth=0.8, alpha=0.7)
        ax.scatter([qx], [qy], c=[color], s=8, zorder=5)
        ax.scatter([rx_shifted], [ry], c=[color], s=8, zorder=5)

    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def try_render_keypoint_overlay(current_query_image, recom_rank):
    """
    Attempts to render a keypoint overlay from viz_payload_{rank} metadata column.
    Gracefully degrades if the column or data is absent.
    """
    df = st.session_state.matching_results_table
    current_image_name = os.path.basename(current_query_image)
    matching_row = df[df['path_relative_to_root'].apply(lambda x: os.path.basename(x)) == current_image_name]

    viz_col = f"viz_payload_{recom_rank}"
    if matching_row.empty or viz_col not in matching_row.columns:
        st.caption("Keypoint overlay not available (run with local matcher enabled).")
        return

    raw = matching_row[viz_col].iloc[0]
    if not isinstance(raw, str) or not raw.strip():
        st.caption("Keypoint overlay not available (run with local matcher enabled).")
        return

    try:
        payload = json.loads(raw)
        query_kpts  = payload.get("query_kpts", [])
        ref_kpts    = payload.get("ref_kpts", [])
        matches     = payload.get("matches", [])

        matched_img_col = f"match_image_{recom_rank}"
        ref_serial = matching_row[matched_img_col].iloc[0] if matched_img_col in matching_row.columns else None

        # Look up reference image path from reference metadata
        ref_df = st.session_state.metadata_table
        ref_row = ref_df[ref_df['image_id'] == ref_serial] if ref_serial is not None else ref_df.iloc[:0]

        if ref_row.empty:
            st.caption("Keypoint overlay: reference image path not found.")
            return

        ref_img_path = os.path.join(st.session_state.root_dir, ref_row.iloc[0]['path_relative_to_root'])
        query_crop = get_corresponding_torso_image(current_query_image)

        if not os.path.isfile(query_crop) or not os.path.isfile(ref_img_path):
            st.caption("Keypoint overlay: crop files not found on disk.")
            return

        q_img = ImageOps.exif_transpose(Image.open(query_crop)).resize((256, 256))
        r_img = ImageOps.exif_transpose(Image.open(ref_img_path)).resize((256, 256))

        overlay_bytes = render_keypoint_overlay(q_img, r_img, query_kpts, ref_kpts, matches)
        st.image(overlay_bytes, caption=f"LightGlue keypoint matches — rank {recom_rank}", use_container_width=True)

    except Exception as exc:
        st.caption(f"Keypoint overlay rendering failed: {exc}")


def display_custom_table(assigned_id, image_serial_matched, fused_sim, global_sim, local_count, recom_rank,
                         query_viewpoint=None, candidate_viewpoint=None):
    vp_row_query = f"""
        <tr>
            <td class="header">Query Viewpoint</td>
            <td>{query_viewpoint if query_viewpoint is not None else 'N/A'}</td>
        </tr>""" if query_viewpoint is not None else ""
    vp_row_cand = f"""
        <tr>
            <td class="header">Candidate Viewpoint</td>
            <td>{candidate_viewpoint if candidate_viewpoint is not None else 'N/A'}</td>
        </tr>""" if candidate_viewpoint is not None else ""

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
            <td>{image_serial_matched}</td>
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
        {vp_row_query}
        {vp_row_cand}
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
    mask = df['filename'].isin(query_basenames)
    df.loc[mask, 'human_input'] = human_input_value

def get_corresponding_torso_image(image_path, subdir_name="original_size", suffix=""):
    parts = image_path.rsplit(".", 1)
    img_filename = os.path.basename(image_path)
    cropped_torso_zoomed_image_dir = os.path.join(st.session_state.processed_img_dir, subdir_name, img_filename).replace("." + parts[1], "_cropped_torso" + suffix + "." + parts[1])
    return cropped_torso_zoomed_image_dir

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
        matched_image      = matching_row[f'match_image_{recom_rank}'].iloc[0] if f'match_image_{recom_rank}' in matching_row.columns else None

        fused_sim   = matching_row[f'match_fused_sim_{recom_rank}'].iloc[0] if f'match_fused_sim_{recom_rank}' in matching_row.columns else None
        global_sim  = matching_row[f'match_global_sim_{recom_rank}'].iloc[0] if f'match_global_sim_{recom_rank}' in matching_row.columns else None
        local_count = matching_row[f'match_local_count_{recom_rank}'].iloc[0] if f'match_local_count_{recom_rank}' in matching_row.columns else None

        fused_sim  = float("{:.4f}".format(fused_sim))  if fused_sim  is not None and not (isinstance(fused_sim, float)  and np.isnan(fused_sim))  else fused_sim
        global_sim = float("{:.4f}".format(global_sim)) if global_sim is not None and not (isinstance(global_sim, float) and np.isnan(global_sim)) else global_sim

        query_viewpoint     = matching_row['viewpoint'].iloc[0] if 'viewpoint' in matching_row.columns else None
        candidate_viewpoint = matching_row[f'match_viewpoint_{recom_rank}'].iloc[0] if f'match_viewpoint_{recom_rank}' in matching_row.columns else None

        return matched_individual, matched_image, fused_sim, global_sim, local_count, ground_truth, query_viewpoint, candidate_viewpoint
    else:
        return None, None, None, None, None, None, None, None

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
    remaining_ids   = [img_id for img_id in matching_rows['image_id'].tolist()]

    if exact_path is not None:
        all_paths = [exact_path] + remaining_paths
    else:
        all_paths = remaining_paths

    return all_paths

def update_reference_images(current_query_image, recom_rank):
    matched_label, matched_image, _, _, _, _, _, _ = get_matched_label(current_query_image, recom_rank)
    reference_image_paths = get_paths_by_matched_label_id(matched_label, matched_image)
    return reference_image_paths

def update_reference_images_ground_truth(current_query_image, recom_rank):
    _, _, _, _, _, ground_truth, _, _ = get_matched_label(current_query_image, recom_rank)
    reference_image_paths = get_paths_by_matched_label_id(ground_truth, None)
    return reference_image_paths

def update_human_input_single(key_query, human_input_value):
    base_names = st.session_state.matching_results_table['path_relative_to_root'].apply(os.path.basename)
    mask = base_names == os.path.basename(key_query)
    st.session_state.matching_results_table.loc[mask, 'human_input'] = human_input_value

def main_analyze_matched_images(query_image_paths):
    cropped_torso_paths_zoomed = [get_corresponding_torso_image(p, "zoomed_version", "_zoomed") for p in query_image_paths]
    cropped_torso_paths = [get_corresponding_torso_image(p) for p in query_image_paths]

    if 'current_query_index' not in st.session_state:
        st.session_state.current_query_index = 0
    if 'current_reference_index' not in st.session_state:
        st.session_state.current_reference_index = 0
    if 'reference_image_paths' not in st.session_state:
        st.session_state.reference_image_paths = []
    if 'recomms_rank_selected' not in st.session_state:
        st.session_state.recomms_rank_selected = 1

    matched_label = None
    col_middle, col_right, col_left = st.columns([3, 3, 1.5])

    with col_middle:

        st.subheader('Matched Query Elephant Images')

        if len(query_image_paths) > 0:

            if st.button('Next Query Image'):
                st.session_state.recomms_rank_selected = 1
                if st.session_state.current_query_index == (len(query_image_paths) - 1):
                    st.warning("You have looped over all query images once and saved human inputs if action items were pressed.")
                st.session_state.current_query_index = (st.session_state.current_query_index + 1) % len(query_image_paths)
                st.session_state.current_reference_index = 0

            if st.button('Previous Query Image'):
                if (st.session_state.current_query_index - 1) >= 0:
                    st.session_state.recomms_rank_selected = 1
                    st.session_state.current_query_index -= 1
                    st.session_state.current_reference_index = 0
                else:
                    st.warning("You reached the first item.")

            if st.button('Jump Forward 100 Images'):
                if (st.session_state.current_query_index + 100) < len(query_image_paths):
                    st.session_state.recomms_rank_selected = 1
                    st.session_state.current_query_index += 100
                    st.session_state.current_reference_index = 0
                else:
                    st.warning("Not enough data available for this action.")

            if st.button('Jump Backward 100 Images'):
                if (st.session_state.current_query_index - 100) >= 0:
                    st.session_state.recomms_rank_selected = 1
                    st.session_state.current_query_index -= 100
                    st.session_state.current_reference_index = 0
                else:
                    st.warning("Not enough data available for this action.")

            st.write(f"{st.session_state.current_query_index + 1} / {len(query_image_paths)}")

            current_query_image = query_image_paths[st.session_state.current_query_index]
            current_cropped_torso_zoomed_image = cropped_torso_paths_zoomed[st.session_state.current_query_index]
            current_cropped_torso_image = cropped_torso_paths[st.session_state.current_query_index]

            display_left_query_images_matched_images(
                [current_query_image, current_cropped_torso_zoomed_image, current_cropped_torso_image],
                ['Query Image: ' + os.path.basename(current_query_image),
                 'Cropped Elephant Zoomed Image: ' + os.path.basename(current_cropped_torso_zoomed_image),
                 'Cropped Elephant Image: ' + os.path.basename(current_cropped_torso_image)]
            )
            st.session_state.reference_image_paths = update_reference_images(current_query_image, st.session_state.recomms_rank_selected)

            # Keypoint overlay below the images
            st.subheader('Keypoint Match Overlay')
            try_render_keypoint_overlay(current_query_image, st.session_state.recomms_rank_selected)

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
                st.session_state.recomms_rank_selected = max(1, (st.session_state.recomms_rank_selected + 1) % (st.session_state.num_id_recomms + 1))
                st.session_state.current_reference_index = 0
                st.session_state.reference_image_paths = update_reference_images(current_query_image, st.session_state.recomms_rank_selected)
                st.success("Next Algorithmic Matched ID clicked! Rank: {}".format(st.session_state.recomms_rank_selected))

            if st.button('Accept All Matched IDs'):
                human_input_value = 'AcceptId'
                update_human_inputs_all(query_image_paths, human_input_value)
                st.success(f"All Matched IDs Accepted!")

            if st.button('Save Analyzed Results'):
                save_human_inputs()
                st.success("Results Saved!")

            # Show matching results with new WildFusion score columns
            st.subheader('Matching Results')
            matched_individual, matched_image, fused_sim, global_sim, local_count, ground_truth, query_vp, cand_vp = \
                get_matched_label(current_query_image, st.session_state.recomms_rank_selected)
            display_custom_table(matched_individual, matched_image, fused_sim, global_sim, local_count,
                                 st.session_state.recomms_rank_selected,
                                 query_viewpoint=query_vp, candidate_viewpoint=cand_vp)

            display_custom_table_human_input(current_query_image)
            display_custom_table_ground_truth(ground_truth)

    with col_right:
        st.subheader('Matched Reference Images')

        num_rows, num_cols = 1, 1

        if len(query_image_paths) > 0:
            if len(st.session_state.reference_image_paths) > 0:

                if st.button('Next Reference Image'):
                    st.session_state.current_reference_index += 1
                    if st.session_state.current_reference_index * num_rows * num_cols >= len(st.session_state.reference_image_paths):
                        st.session_state.current_reference_index = 0

                if st.button('Previous Reference Image'):
                    if (st.session_state.current_reference_index - 1) >= 0:
                        st.session_state.current_reference_index -= 1
                    else:
                        st.warning("You have reached the first item.")

                if st.button('Jump Forward 10 Images'):
                    if (st.session_state.current_reference_index + 10) < len(st.session_state.reference_image_paths):
                        st.session_state.current_reference_index += 10
                    else:
                        st.warning("Not enough data available for this action.")

                if st.button('Jump Backward 10 Images'):
                    if (st.session_state.current_reference_index - 10) >= 0:
                        st.session_state.current_reference_index -= 10
                    else:
                        st.warning("Not enough data available for this action.")

                st.write(f"{st.session_state.current_reference_index + 1} / {len(st.session_state.reference_image_paths)}")

                starting_idx = st.session_state.current_reference_index * num_rows * num_cols
                display_right_reference_image(st.session_state.reference_image_paths, starting_idx, num_rows, num_cols)
            else:
                st.warning("No reference images found for the current query image.")
        else:
            st.warning("No query images found in the specified directory.")

def main_analyze_not_matched_images(query_image_not_matched_paths):
    cropped_torso_paths_zoomed = [get_corresponding_torso_image(p, "zoomed_version", "_zoomed") for p in query_image_not_matched_paths]
    cropped_torso_paths = [get_corresponding_torso_image(p) for p in query_image_not_matched_paths]

    if 'current_query_not_matched_index' not in st.session_state:
        st.session_state.current_query_not_matched_index = 0
    if 'curr_ref_not_matched_idx' not in st.session_state:
        st.session_state.curr_ref_not_matched_idx = 0
    if 'ref_img_paths_not_matched' not in st.session_state:
        st.session_state.ref_img_paths_not_matched = []
    if 'not_matched_recomms_rank_selected' not in st.session_state:
        st.session_state.not_matched_recomms_rank_selected = 1

    col_right, col_ref, col_buttons = st.columns([3, 3, 1.5])

    with col_right:

        st.subheader('Not-matched Query Elephant Images')

        if len(query_image_not_matched_paths) > 0:

            st.write(f"{st.session_state.current_query_not_matched_index + 1} / {len(query_image_not_matched_paths)}")

            if st.button('Next Query Image'):
                if st.session_state.current_query_not_matched_index == (len(query_image_not_matched_paths) - 1):
                    st.warning("You have looped over all query images once and saved human inputs if action items were pressed.")
                st.session_state.current_query_not_matched_index = (st.session_state.current_query_not_matched_index + 1) % len(query_image_not_matched_paths)
                st.session_state.curr_ref_not_matched_idx = 0
                st.session_state.not_matched_recomms_rank_selected = 1

            if st.button('Previous Query Image'):
                if (st.session_state.current_query_not_matched_index - 1) >= 0:
                    st.session_state.current_query_not_matched_index -= 1
                else:
                    st.warning("Not enough remained data for this action.")
                st.session_state.curr_ref_not_matched_idx = 0
                st.session_state.not_matched_recomms_rank_selected = 1

            if st.button('Jump Forward 100 Images'):
                if (st.session_state.current_query_not_matched_index + 100) < len(query_image_not_matched_paths):
                    st.session_state.current_query_not_matched_index += 100
                    st.session_state.not_matched_recomms_rank_selected = 1
                else:
                    st.warning("Not enough remained data for this action.")
                st.session_state.curr_ref_not_matched_idx = 0

            if st.button('Jump Backward 100 Images'):
                if (st.session_state.current_query_not_matched_index - 100) >= 0:
                    st.session_state.current_query_not_matched_index -= 100
                    st.session_state.not_matched_recomms_rank_selected = 1
                else:
                    st.warning("Not enough remained data for this action.")
                st.session_state.curr_ref_not_matched_idx = 0

            current_query_image = query_image_not_matched_paths[st.session_state.current_query_not_matched_index]
            current_cropped_torso_zoomed_image = cropped_torso_paths_zoomed[st.session_state.current_query_not_matched_index]
            current_cropped_torso_image = cropped_torso_paths[st.session_state.current_query_not_matched_index]

            display_left_query_images_not_matched_images(
                [current_query_image, current_cropped_torso_zoomed_image, current_cropped_torso_image],
                ['Query Image: ' + os.path.basename(current_query_image),
                 'Cropped Elephant Zoomed Image: ' + os.path.basename(current_cropped_torso_zoomed_image),
                 'Cropped Elephant Image: ' + os.path.basename(current_cropped_torso_image)]
            )
            st.session_state.ref_img_paths_not_matched = update_reference_images(current_query_image, st.session_state.not_matched_recomms_rank_selected)

            # Keypoint overlay
            st.subheader('Keypoint Match Overlay')
            try_render_keypoint_overlay(current_query_image, st.session_state.not_matched_recomms_rank_selected)

        else:
            st.warning("No query images found in the specified directory.")

    with col_ref:

        st.subheader('Matched Reference Images (Rejected)')

        num_rows, num_cols = 1, 1

        if len(query_image_not_matched_paths) > 0:
            if len(st.session_state.ref_img_paths_not_matched) > 0:

                if st.button('Next Matched Reference Image'):
                    st.session_state.curr_ref_not_matched_idx += 1
                    if st.session_state.curr_ref_not_matched_idx * num_rows * num_cols >= len(st.session_state.ref_img_paths_not_matched):
                        st.session_state.curr_ref_not_matched_idx = 0

                if st.button('Previous Matched Reference Image'):
                    if (st.session_state.curr_ref_not_matched_idx - 1) >= 0:
                        st.session_state.curr_ref_not_matched_idx -= 1
                    else:
                        st.warning("You have reached the first item.")

                if st.button('Jump Forward 10 Images'):
                    if (st.session_state.curr_ref_not_matched_idx + 10) < len(st.session_state.ref_img_paths_not_matched):
                        st.session_state.curr_ref_not_matched_idx += 10
                    else:
                        st.warning("Not enough data available for this action.")

                if st.button('Jump Backward 10 Images'):
                    if (st.session_state.curr_ref_not_matched_idx - 10) >= 0:
                        st.session_state.curr_ref_not_matched_idx -= 10
                    else:
                        st.warning("Not enough data available for this action.")

                st.write(f"{st.session_state.curr_ref_not_matched_idx + 1} / {len(st.session_state.ref_img_paths_not_matched)}")

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
                st.session_state.not_matched_recomms_rank_selected = max(1, (st.session_state.not_matched_recomms_rank_selected + 1) % (st.session_state.num_id_recomms + 1))
                st.session_state.curr_ref_not_matched_idx = 0
                st.session_state.ref_img_paths_not_matched = update_reference_images(current_query_image, st.session_state.not_matched_recomms_rank_selected)
                st.success("Next Algorithmic Matched ID clicked! Rank: {}".format(st.session_state.not_matched_recomms_rank_selected))

            if st.button('Assign New ID to All'):
                human_input_value = 'AssignNewId'
                update_human_inputs_all(query_image_not_matched_paths, human_input_value)
                st.success(f"All Matched IDs Accepted!")

            # Show matching results with WildFusion scores
            st.subheader('Matching Results')
            matched_individual, matched_image, fused_sim, global_sim, local_count, ground_truth, query_vp, cand_vp = \
                get_matched_label(current_query_image, st.session_state.not_matched_recomms_rank_selected)
            display_custom_table(matched_individual, matched_image, fused_sim, global_sim, local_count,
                                 st.session_state.not_matched_recomms_rank_selected,
                                 query_viewpoint=query_vp, candidate_viewpoint=cand_vp)

            display_custom_table_human_input(current_query_image)
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
