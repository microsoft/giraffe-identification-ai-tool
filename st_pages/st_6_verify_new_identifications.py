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

st.title("Verify Identification of Unknown Elephant Individuals")
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
    parts = image_path.rsplit(".", 1)
    img_filename = os.path.basename(image_path)
    cropped_dir = os.path.join(st.session_state.processed_img_dir, subdir_name, img_filename).replace(
        "." + parts[1], "_cropped_torso" + suffix + "." + parts[1]
    )
    return cropped_dir

def load_metadata_with_partitioning_results():
    partition = 'query'
    metadata_filepath = os.path.join(st.session_state.root_dir, partition + '_dir', 'metadata_' + partition + '.csv')
    metadata_query = load_metadata_file(metadata_filepath)

    needed_cols = ['image_id', 'path_relative_to_root', 'assigned_individual_id', 'human_input']
    optional_cols = ['viewpoint', 'match_fused_sim_1', 'match_global_sim_1', 'match_local_count_1', 'match_viewpoint_1', 'individual_id']
    available_optional = [c for c in optional_cols if c in metadata_query.columns]
    cols_to_use = needed_cols + available_optional

    missing_cols = [col for col in needed_cols if col not in metadata_query.columns]
    if missing_cols:
        st.write(f"WARNING! The following required columns are missing from metadata_query: {', '.join(missing_cols)}")
        st.stop()

    filtered_df = metadata_query[
        (metadata_query['assigned_individual_id'].notna()) &
        (metadata_query['human_input'] == 'AssignNewId')
    ]

    # Group by assigned_individual_id; pull in whatever optional columns are present
    st.session_state.partitions = filtered_df.groupby('assigned_individual_id')[cols_to_use].agg(list).to_dict('index')


def render_keypoint_overlay(query_img, ref_img, query_kpts, ref_kpts, matches):
    """Side-by-side images with lines connecting matched LightGlue keypoints. Returns PNG bytes."""
    q_arr = np.array(query_img.convert("RGB"))
    r_arr = np.array(ref_img.convert("RGB"))
    q_h, q_w = q_arr.shape[:2]
    r_h, r_w = r_arr.shape[:2]
    canvas_h = max(q_h, r_h)
    canvas = np.zeros((canvas_h, q_w + r_w, 3), dtype=np.uint8)
    canvas[:q_h, :q_w] = q_arr
    canvas[:r_h, q_w:] = r_arr

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(canvas)
    ax.axis("off")

    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(matches), 1)))
    for (qi, ri), color in zip(matches, colors):
        qx, qy = query_kpts[qi]
        rx, ry = ref_kpts[ri]
        ax.plot([qx, rx + q_w], [qy, ry], color=color, linewidth=0.8, alpha=0.7)
        ax.scatter([qx], [qy], c=[color], s=8, zorder=5)
        ax.scatter([rx + q_w], [ry], c=[color], s=8, zorder=5)

    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def try_render_keypoint_overlay_for_image(query_image_path, recom_rank=1):
    """Attempts keypoint overlay for a single image; gracefully degrades if unavailable."""
    df = st.session_state.get('matching_results_table', None)
    if df is None:
        # st_6 doesn't always have matching_results_table loaded; silently skip
        return

    current_image_name = os.path.basename(query_image_path)
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
        payload    = json.loads(raw)
        query_kpts = payload.get("query_kpts", [])
        ref_kpts   = payload.get("ref_kpts", [])
        matches    = payload.get("matches", [])

        match_img_col = f"match_image_{recom_rank}"
        ref_image_id  = matching_row[match_img_col].iloc[0] if match_img_col in matching_row.columns else None

        ref_df = st.session_state.get('metadata_table', None)
        if ref_df is None or ref_image_id is None:
            st.caption("Keypoint overlay: reference metadata not loaded.")
            return

        ref_row = ref_df[ref_df['image_id'] == ref_image_id]
        if ref_row.empty:
            st.caption("Keypoint overlay: reference image not found.")
            return

        ref_img_path = os.path.join(st.session_state.root_dir, ref_row.iloc[0]['path_relative_to_root'])
        query_crop   = get_corresponding_torso_image(query_image_path)

        if not os.path.isfile(query_crop) or not os.path.isfile(ref_img_path):
            st.caption("Keypoint overlay: crop files not found on disk.")
            return

        q_img = ImageOps.exif_transpose(Image.open(query_crop)).resize((256, 256))
        r_img = ImageOps.exif_transpose(Image.open(ref_img_path)).resize((256, 256))

        overlay_bytes = render_keypoint_overlay(q_img, r_img, query_kpts, ref_kpts, matches)
        st.image(overlay_bytes, caption=f"LightGlue keypoint matches — rank {recom_rank}", use_container_width=True)

    except Exception as exc:
        st.caption(f"Keypoint overlay rendering failed: {exc}")


def display_all_images_within_partition(partition_data):
    query_image_paths = partition_data['path_relative_to_root']
    image_ids         = partition_data['image_id']
    individual_ids    = partition_data.get('individual_id', ['NA'] * len(image_ids))
    viewpoints        = partition_data.get('viewpoint', ['NA'] * len(image_ids))
    fused_sims        = partition_data.get('match_fused_sim_1', [None] * len(image_ids))
    global_sims       = partition_data.get('match_global_sim_1', [None] * len(image_ids))
    local_counts      = partition_data.get('match_local_count_1', [None] * len(image_ids))
    cand_viewpoints   = partition_data.get('match_viewpoint_1', ['NA'] * len(image_ids))

    cropped_torso_paths_zoomed = [get_corresponding_torso_image(p, "zoomed_version", "_zoomed") for p in query_image_paths]

    for idx in range(len(query_image_paths)):
        cols = st.columns(2)

        image_id   = image_ids[idx]
        individual = individual_ids[idx] if isinstance(individual_ids, list) else 'NA'
        viewpoint  = viewpoints[idx]     if isinstance(viewpoints, list)     else 'NA'
        fused_sim  = fused_sims[idx]     if isinstance(fused_sims, list)     else None
        global_sim = global_sims[idx]    if isinstance(global_sims, list)    else None
        local_cnt  = local_counts[idx]   if isinstance(local_counts, list)   else None
        cand_vp    = cand_viewpoints[idx] if isinstance(cand_viewpoints, list) else 'NA'

        score_parts = []
        if fused_sim  is not None: score_parts.append(f"Fused={fused_sim:.3f}" if isinstance(fused_sim, float) else f"Fused={fused_sim}")
        if global_sim is not None: score_parts.append(f"Global={global_sim:.3f}" if isinstance(global_sim, float) else f"Global={global_sim}")
        if local_cnt  is not None: score_parts.append(f"Inliers={local_cnt}")
        score_str = " | ".join(score_parts) if score_parts else ""

        caption = (
            f"Elephant {idx + 1} — Image ID: {image_id} | Individual: {individual} | "
            f"Viewpoint: {viewpoint} | Cand. VP: {cand_vp}"
        )
        if score_str:
            caption += f"\n{score_str}"

        query_image = ImageOps.exif_transpose(Image.open(os.path.join(st.session_state.root_dir, query_image_paths[idx])))
        width, height = query_image.size
        if height > 0:
            new_height = 256
            new_width  = int((new_height / height) * width)
            query_image = query_image.resize((new_width, new_height))

        cropped_torso_image_zoomed = Image.open(cropped_torso_paths_zoomed[idx])
        cropped_torso_image_zoomed = cropped_torso_image_zoomed.resize((256, 256))

        cols[0].image(query_image, caption=caption, use_container_width=False)
        cols[1].image(cropped_torso_image_zoomed, caption=f"Cropped Elephant {idx + 1}", use_container_width=False)

        # Keypoint overlay if viz payload is available
        full_path = os.path.join(st.session_state.root_dir, query_image_paths[idx])
        try_render_keypoint_overlay_for_image(full_path, recom_rank=1)

def main_analyze_partitioned_results():
    if 'current_right_partition_index' not in st.session_state:
        st.session_state.current_right_partition_index = 0
    if 'current_left_partition_index' not in st.session_state:
        st.session_state.current_left_partition_index = 0
    if 'partitions' not in st.session_state:
        st.session_state.partitions = {}

    load_metadata_with_partitioning_results()

    col_left, col_right = st.columns([3, 3])

    with col_left:
        st.subheader('Partition - Left')

        partition_keys_left = list(st.session_state.partitions.keys())
        if len(partition_keys_left) > 0:

            prev_col, next_col = st.columns([0.75, 1.5])
            with next_col:
                if st.button('Next | Left'):
                    if st.session_state.current_left_partition_index + 1 < len(partition_keys_left):
                        st.session_state.current_left_partition_index += 1
                    else:
                        st.warning("You have reached the last partition.")
            with prev_col:
                if st.button('Previous | Left'):
                    if st.session_state.current_left_partition_index - 1 >= 0:
                        st.session_state.current_left_partition_index -= 1
                    else:
                        st.warning("You have reached the first partition.")

            st.session_state.current_partition_key_left = partition_keys_left[st.session_state.current_left_partition_index]
            partition_data_left = st.session_state.partitions[st.session_state.current_partition_key_left]

            st.write(f"Displaying partition # {st.session_state.current_partition_key_left} | {st.session_state.current_left_partition_index + 1} of {len(partition_keys_left)}")

            display_all_images_within_partition(partition_data_left)

        else:
            st.warning("No partitions found.")

    with col_right:
        st.subheader('Partition - Right')

        partition_keys_right = list(st.session_state.partitions.keys())
        if len(partition_keys_right) > 0:

            prev_col, next_col = st.columns([0.75, 1.5])
            with next_col:
                if st.button('Next | Right'):
                    if st.session_state.current_right_partition_index + 1 < len(partition_keys_right):
                        st.session_state.current_right_partition_index += 1
                    else:
                        st.warning("You have reached the last partition.")
            with prev_col:
                if st.button('Previous | Right'):
                    if st.session_state.current_right_partition_index - 1 >= 0:
                        st.session_state.current_right_partition_index -= 1
                    else:
                        st.warning("You have reached the first partition.")

            st.session_state.current_partition_key_right = partition_keys_right[st.session_state.current_right_partition_index]
            partition_data_right = st.session_state.partitions[st.session_state.current_partition_key_right]

            st.write(f"Displaying partition # {st.session_state.current_partition_key_right} | {st.session_state.current_right_partition_index + 1} of {len(partition_keys_right)}")

            display_all_images_within_partition(partition_data_right)

        else:
            st.warning("No partitions found.")


# Run the main function
pycode_name = 'step_4_partition_new_items.py'
main_analyze_partitioned_results()
