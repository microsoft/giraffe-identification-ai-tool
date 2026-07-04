# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers_matching import load_data_dirs

from user_authentication import login_ui, authorize_users

if authorize_users() and not st.session_state.get("authenticated", False):
    st.markdown("""<style>[data-testid="stSidebar"] { display: none; }</style>""", unsafe_allow_html=True)
    login_ui()
    st.stop()

if "root_dir" not in st.session_state or not st.session_state.root_dir:
    st.session_state.root_dir, st.session_state.processed_img_dir = load_data_dirs()

st.title("Advanced Tools")

tab_gt, tab_img, tab_settings = st.tabs(["Ground-Truth Evaluation", "Single Image Inspector", "Settings"])

# ---------------------------------------------------------------------------
# Tab 1: Ground-Truth Evaluation (mirrors st_8 functionality)
# ---------------------------------------------------------------------------
with tab_gt:
    st.markdown("Full ground-truth metrics and error inspection are available on the legacy page.")
    if st.button("Open Ground-Truth Evaluation page", use_container_width=False):
        st.switch_page("st_pages/st_8_validate_based_on_ground_truth.py")
    st.caption(
        "Provides: accuracy metrics, false-positive inspection, false-negative inspection, "
        "same-view vs. cross-view breakdown."
    )

# ---------------------------------------------------------------------------
# Tab 2: Single Image Inspector (mirrors st_9 functionality)
# ---------------------------------------------------------------------------
with tab_img:
    st.markdown("Inspect a single image and look up its viewpoint tag from metadata.")
    if st.button("Open Single Image Inspector", use_container_width=False):
        st.switch_page("st_pages/st_9_visualize_single_image.py")

# ---------------------------------------------------------------------------
# Tab 3: Settings
# ---------------------------------------------------------------------------
with tab_settings:
    st.subheader("Project Settings")

    root_dir = st.session_state.root_dir
    st.text_input("Data root dir", value=root_dir, disabled=True)

    query_csv = os.path.join(root_dir, "query_dir", "metadata_query.csv")
    ref_csv = os.path.join(root_dir, "reference_dir", "metadata_reference.csv")
    st.text_input("Query metadata", value=query_csv, disabled=True)
    st.text_input("Reference metadata", value=ref_csv, disabled=True)

    st.divider()
    st.subheader("Reset Review Inputs")
    st.warning(
        "This will erase all human_input values from the query metadata CSV. "
        "All accepted/rejected/skipped decisions will be lost."
    )
    confirm = st.checkbox("I understand — erase all review inputs")
    if confirm and st.button("Reset All Review Inputs", type="secondary"):
        if "matching_results_table" in st.session_state:
            st.session_state.matching_results_table["human_input"] = None
            st.session_state.matching_results_table.to_csv(query_csv, index=False)
            st.success("Review inputs cleared and saved to disk.")
            st.rerun()
        else:
            st.error("No project loaded.")

    st.divider()
    st.subheader("Reload Project from Disk")
    st.caption("Use this if the metadata CSV was modified outside the app.")
    if st.button("Reload metadata from disk"):
        for key in ["matching_results_table", "metadata_table"]:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Project cleared from memory — will reload on next page visit.")
        st.rerun()
