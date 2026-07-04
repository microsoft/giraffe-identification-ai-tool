# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_elephant import NEW_ID_PREFIX
from utils.helpers_matching import load_data_dirs, load_metadata_file

from user_authentication import login_ui, authorize_users

if authorize_users() and not st.session_state.get("authenticated", False):
    st.markdown("""<style>[data-testid="stSidebar"] { display: none; }</style>""", unsafe_allow_html=True)
    login_ui()
    st.stop()

if "root_dir" not in st.session_state or not st.session_state.root_dir:
    st.session_state.root_dir, st.session_state.processed_img_dir = load_data_dirs()

st.html("""
<style>
.stMainBlockContainer {
    width: 100%;
    max-width: 100%;
    padding-left: 40px;
    padding-right: 40px;
    padding-top: 60px;
    padding-bottom: 120px;
}
</style>
""")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _auto_load():
    if "matching_results_table" in st.session_state:
        return True
    root_dir = st.session_state.root_dir
    query_path = os.path.join(root_dir, "query_dir", "metadata_query.csv")
    if not os.path.exists(query_path):
        return False
    try:
        st.session_state.matching_results_table = load_metadata_file(query_path)
        return True
    except Exception:
        return False


def _save():
    path = os.path.join(st.session_state.root_dir, "query_dir", "metadata_query.csv")
    st.session_state.matching_results_table.to_csv(path, index=False)
    st.session_state.ru_last_saved = datetime.now()


def _get_torso_path(image_path, subdir="original_size", suffix=""):
    ext = image_path.rsplit(".", 1)[-1]
    fname = os.path.basename(image_path)
    return os.path.join(
        st.session_state.processed_img_dir, subdir, fname
    ).replace(f".{ext}", f"_cropped_torso{suffix}.{ext}")


def _load_thumbnail(full_path, size=160):
    zoomed = _get_torso_path(full_path, "zoomed_version", "_zoomed")
    plain = _get_torso_path(full_path)
    for candidate in [zoomed, plain]:
        if os.path.isfile(candidate):
            img = ImageOps.exif_transpose(Image.open(candidate))
            return img.resize((size, size))
    if os.path.isfile(full_path):
        img = ImageOps.exif_transpose(Image.open(full_path))
        w, h = img.size
        if h > 0:
            scale = size / h
            return img.resize((int(w * scale), size))
    return None


def _get_clusters(df):
    """Return list of (individual_id, cluster_df) sorted by cluster size desc."""
    if "assigned_individual_id" not in df.columns:
        return []

    unk_mask = df["assigned_individual_id"].astype(str).str.startswith(NEW_ID_PREFIX, na=False)
    unk_df = df[unk_mask].copy()

    clusters = []
    for ind_id, group in unk_df.groupby("assigned_individual_id"):
        clusters.append((ind_id, group))

    clusters.sort(key=lambda x: -len(x[1]))
    return clusters


def _get_cluster_decision(df, ind_id):
    mask = df["assigned_individual_id"] == ind_id
    if "human_input" not in df.columns:
        return None
    vals = df.loc[mask, "human_input"].dropna().unique()
    return vals[0] if len(vals) > 0 else None


def _set_cluster_decision(ind_id, value):
    df = st.session_state.matching_results_table
    mask = df["assigned_individual_id"] == ind_id
    if "human_input" not in df.columns:
        df["human_input"] = None
    st.session_state.matching_results_table.loc[mask, "human_input"] = value


def _assign_known_id(ind_id, known_id):
    """Reassign all images in the cluster to a known individual ID."""
    df = st.session_state.matching_results_table
    mask = df["assigned_individual_id"] == ind_id
    st.session_state.matching_results_table.loc[mask, "assigned_individual_id"] = known_id
    st.session_state.matching_results_table.loc[mask, "human_input"] = "AcceptId"


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main():
    st.title("Review Unknowns")

    if not _auto_load():
        st.warning("No project found. Go to **Dashboard** to load a project.")
        return

    df = st.session_state.matching_results_table
    if "assigned_individual_id" not in df.columns:
        st.warning("Unknown clustering has not been run yet. Go to **Run Analysis** → Cluster Unknowns.")
        return

    clusters = _get_clusters(df)
    if not clusters:
        st.info("No unknown clusters found.")
        return

    # Progress
    confirmed = sum(
        1 for ind_id, _ in clusters if _get_cluster_decision(df, ind_id) is not None
    )
    n_total = len(clusters)
    last = st.session_state.get("ru_last_saved")

    hc1, hc2, hc3 = st.columns([5, 2, 1])
    with hc1:
        st.progress(
            confirmed / n_total,
            text=f"{confirmed} / {n_total} clusters confirmed — {n_total - confirmed} remaining",
        )
    with hc2:
        if last:
            mins = max(0, int((datetime.now() - last).total_seconds() // 60))
            st.caption(f"Saved {mins}m ago" if mins > 0 else "Just saved")
    with hc3:
        if st.button("Save", use_container_width=True):
            _save()
            st.success("Saved!")

    st.divider()

    root_dir = st.session_state.root_dir

    # Filter controls
    filter_choice = st.radio(
        "Show",
        ["All", "Unconfirmed only", "Confirmed only"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # Render clusters
    IMGS_PER_ROW = 4

    for ind_id, cluster_df in clusters:
        decision = _get_cluster_decision(st.session_state.matching_results_table, ind_id)
        is_confirmed = decision is not None

        if filter_choice == "Unconfirmed only" and is_confirmed:
            continue
        if filter_choice == "Confirmed only" and not is_confirmed:
            continue

        n_photos = len(cluster_df)
        status_icon = "✓" if is_confirmed else "○"
        expander_label = f"{status_icon}  {ind_id} — {n_photos} photo{'s' if n_photos != 1 else ''}"

        with st.expander(expander_label, expanded=not is_confirmed):
            # Thumbnail grid
            paths = [os.path.join(root_dir, p) for p in cluster_df["path_relative_to_root"]]
            image_ids = cluster_df.get("image_id", pd.Series(["?"] * n_photos)).tolist()
            viewpoints = cluster_df.get("viewpoint", pd.Series(["?"] * n_photos)).tolist()
            fused_sims = cluster_df.get("match_fused_sim_1", pd.Series([None] * n_photos)).tolist()

            for row_start in range(0, n_photos, IMGS_PER_ROW):
                row_items = list(zip(
                    paths[row_start:row_start + IMGS_PER_ROW],
                    image_ids[row_start:row_start + IMGS_PER_ROW],
                    viewpoints[row_start:row_start + IMGS_PER_ROW],
                    fused_sims[row_start:row_start + IMGS_PER_ROW],
                ))
                cols = st.columns(IMGS_PER_ROW)
                for col, (img_path, img_id, vp, fused) in zip(cols, row_items):
                    with col:
                        thumb = _load_thumbnail(img_path, size=140)
                        if thumb:
                            cap_parts = [str(img_id)]
                            if vp and vp != "?":
                                cap_parts.append(f"vp: {vp}")
                            if fused is not None:
                                try:
                                    cap_parts.append(f"score: {float(fused):.2f}")
                                except Exception:
                                    pass
                            st.image(thumb, caption="\n".join(cap_parts), use_container_width=False)
                        else:
                            st.caption(f"Missing: {os.path.basename(img_path)}")

            # Action row
            st.divider()
            ac1, ac2 = st.columns([1, 2])
            with ac1:
                confirm_key = f"confirm_{ind_id}"
                if st.button("✓ Confirm as new individual", key=confirm_key, use_container_width=True):
                    _set_cluster_decision(ind_id, "AssignNewId")
                    _save()
                    st.rerun()
            with ac2:
                with st.form(key=f"assign_{ind_id}"):
                    fc1, fc2 = st.columns([3, 1])
                    with fc1:
                        known_id_input = st.text_input(
                            "Assign to known ID",
                            placeholder="e.g. eleph_kunene",
                            label_visibility="collapsed",
                        )
                    with fc2:
                        submitted = st.form_submit_button("Assign", use_container_width=True)
                    if submitted and known_id_input.strip():
                        _assign_known_id(ind_id, known_id_input.strip())
                        _save()
                        st.success(f"Reassigned to {known_id_input.strip()}")
                        st.rerun()

            if decision is not None:
                st.caption(f"Decision: {decision}")


main()
