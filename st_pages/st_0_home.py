# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_elephant import EMBEDDINGS_SUBDIR, ACTIVE_DESCRIPTORS, NEW_ID_PREFIX
from utils.helpers_matching import load_data_dirs

from user_authentication import login_ui, authorize_users

if authorize_users() and not st.session_state.get("authenticated", False):
    st.markdown("""<style>[data-testid="stSidebar"] { display: none; }</style>""", unsafe_allow_html=True)
    login_ui()
    st.stop()

# Ensure root_dir is populated (app.py sets it, but guard here for direct page loads)
if "root_dir" not in st.session_state or not st.session_state.root_dir:
    st.session_state.root_dir, st.session_state.processed_img_dir = load_data_dirs()

root_dir = st.session_state.root_dir
processed_img_dir = st.session_state.get("processed_img_dir", os.path.join(root_dir, "processed_images"))


# ---------------------------------------------------------------------------
# Project auto-load
# ---------------------------------------------------------------------------

def _auto_load_project():
    """Load query + reference metadata into session state if not already loaded."""
    if "matching_results_table" in st.session_state:
        return True

    query_path = os.path.join(root_dir, "query_dir", "metadata_query.csv")
    if not os.path.exists(query_path):
        return False

    try:
        st.session_state.matching_results_table = pd.read_csv(query_path)

        ref_path = os.path.join(root_dir, "reference_dir", "metadata_reference.csv")
        if os.path.exists(ref_path):
            st.session_state.metadata_table = pd.read_csv(ref_path)

        # Initialize review navigation state
        if "rm_queue_idx" not in st.session_state:
            st.session_state.rm_queue_idx = 0
            st.session_state.rm_ref_idx = 0
            st.session_state.rm_rank = 1
            st.session_state.rm_decisions_since_save = 0

        return True
    except Exception as exc:
        st.warning(f"Could not load project: {exc}")
        return False


# ---------------------------------------------------------------------------
# Pipeline step detection
# ---------------------------------------------------------------------------

def _check_pipeline_steps(df):
    steps = []

    # Step 1: Detect & Crop — look for crop files under processed_images/
    n_crops = 0
    original_size_dir = os.path.join(processed_img_dir, "original_size")
    if os.path.isdir(original_size_dir):
        n_crops = sum(1 for f in os.listdir(original_size_dir) if "_cropped_torso" in f)
    elif os.path.isdir(processed_img_dir):
        for sub in os.listdir(processed_img_dir):
            subpath = os.path.join(processed_img_dir, sub)
            if os.path.isdir(subpath):
                n_crops += sum(1 for f in os.listdir(subpath) if "_cropped_torso" in f)
    steps.append(("Detect & Crop", n_crops > 0, f"{n_crops} crops found"))

    # Step 2: Extract Features — check for any embedding npy
    emb_dir = os.path.join(root_dir, "query_dir", EMBEDDINGS_SUBDIR)
    has_embeddings = False
    if os.path.isdir(emb_dir):
        for desc in ACTIVE_DESCRIPTORS:
            npy = os.path.join(emb_dir, f"query_{desc}.npy")
            if os.path.isfile(npy):
                has_embeddings = True
                break
    steps.append(("Extract Features", has_embeddings, "embeddings ready" if has_embeddings else "not run"))

    # Step 3: Run Matching — matching_status column present
    has_matching = df is not None and "matching_status" in df.columns
    if has_matching:
        n_matched = (df["matching_status"] == "matched").sum()
        n_nm = (df["matching_status"] == "not_matched").sum()
        detail = f"{n_matched} matched · {n_nm} not-matched"
    else:
        detail = "not run"
    steps.append(("Run Matching", has_matching, detail))

    # Step 4: Cluster Unknowns — any assigned_individual_id starting with NEW_ID_PREFIX
    has_clusters = False
    n_clusters = 0
    if df is not None and "assigned_individual_id" in df.columns:
        mask = df["assigned_individual_id"].astype(str).str.startswith(NEW_ID_PREFIX, na=False)
        n_clusters = df.loc[mask, "assigned_individual_id"].nunique()
        has_clusters = n_clusters > 0
    steps.append(("Cluster Unknowns", has_clusters, f"{n_clusters} clusters" if has_clusters else "not run"))

    # Step 5: Update Catalogue — hard to detect; show as pending
    steps.append(("Update Catalogue", False, "pending review completion"))

    return steps


# ---------------------------------------------------------------------------
# Review stats
# ---------------------------------------------------------------------------

def _get_review_stats(df):
    if df is None or "matching_status" not in df.columns:
        return None

    total_matched = int((df["matching_status"] == "matched").sum())
    total_nm = int((df["matching_status"] == "not_matched").sum())
    total = total_matched + total_nm

    hi = "human_input"
    if hi in df.columns:
        accepted = int((df[hi] == "AcceptId").sum())
        new_id = int((df[hi] == "AssignNewId").sum())
        skipped = int((df[hi] == "SkipImage").sum())
        reviewed = accepted + new_id + skipped
    else:
        accepted = new_id = skipped = reviewed = 0

    return {
        "total": total,
        "total_matched": total_matched,
        "total_not_matched": total_nm,
        "reviewed": reviewed,
        "accepted": accepted,
        "new_id": new_id,
        "skipped": skipped,
        "remaining": max(0, total - reviewed),
    }


def _get_unknowns_stats(df):
    if df is None or "assigned_individual_id" not in df.columns:
        return None

    hi = "human_input"
    unk_mask = df["assigned_individual_id"].astype(str).str.startswith(NEW_ID_PREFIX, na=False)
    unk_df = df[unk_mask]
    total_clusters = unk_df["assigned_individual_id"].nunique()

    confirmed = 0
    if hi in df.columns:
        confirmed_mask = unk_mask & (df[hi].isin(["AssignNewId", "AcceptId"]))
        confirmed = df.loc[confirmed_mask, "assigned_individual_id"].nunique()

    return {"total": total_clusters, "confirmed": confirmed, "remaining": max(0, total_clusters - confirmed)}


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

st.title("Dashboard")

project_loaded = _auto_load_project()
df = st.session_state.get("matching_results_table")

# Project state banner
query_csv = os.path.join(root_dir, "query_dir", "metadata_query.csv")
if os.path.exists(query_csv):
    mtime = os.path.getmtime(query_csv)
    last_saved_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    st.caption(f"Project: `{root_dir}` — last saved: {last_saved_str}")
else:
    st.caption(f"Project: `{root_dir}`")

is_advanced = st.session_state.get("mode", "field") == "advanced"

# --- No query metadata → CTA ---
if not project_loaded or df is None:
    st.divider()
    st.warning("No query metadata found. Create a query table to get started.")
    if is_advanced:
        if st.button("Create Query Table", type="primary"):
            st.switch_page("st_pages/st_1_create_query_table.py")
    else:
        st.info("Switch to **Advanced** mode to create a query table.")
    st.stop()

# --- Pipeline status ---
st.subheader("Pipeline")
steps = _check_pipeline_steps(df)
step_cols = st.columns(len(steps))
for col, (name, done, detail) in zip(step_cols, steps):
    with col:
        icon = "✓" if done else "○"
        color = "green" if done else "gray"
        st.markdown(
            f"<div style='text-align:center'>"
            f"<span style='font-size:1.4em;color:{color}'>{icon}</span><br>"
            f"<strong>{name}</strong><br>"
            f"<small style='color:gray'>{detail}</small>"
            f"</div>",
            unsafe_allow_html=True,
        )

# --- Review Matches progress ---
st.divider()
stats = _get_review_stats(df)

col_matches, col_unknowns = st.columns(2)

with col_matches:
    st.subheader("Review Matches")
    if stats:
        reviewed = stats["reviewed"]
        total = stats["total"]
        pct = reviewed / total if total > 0 else 0
        st.progress(pct, text=f"{reviewed} / {total} reviewed")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accepted", stats["accepted"])
        c2.metric("New ID", stats["new_id"])
        c3.metric("Skipped", stats["skipped"])
        st.caption(f"{stats['total_matched']} matched · {stats['total_not_matched']} not-matched")
    else:
        st.info("Run matching to see review progress.")

    if st.button("Continue Reviewing →", type="primary", use_container_width=True):
        st.switch_page("st_pages/st_review_matches.py")

with col_unknowns:
    st.subheader("Review Unknowns")
    unk_stats = _get_unknowns_stats(df)
    if unk_stats:
        confirmed = unk_stats["confirmed"]
        total_c = unk_stats["total"]
        pct_c = confirmed / total_c if total_c > 0 else 0
        st.progress(pct_c, text=f"{confirmed} / {total_c} clusters confirmed")
        st.caption(f"{unk_stats['remaining']} remaining")
    else:
        st.info("Cluster unknowns to see progress.")

    if st.button("Review Unknowns →", use_container_width=True):
        st.switch_page("st_pages/st_review_unknowns.py")

# --- Advanced-only: reset and catalogue update ---
if is_advanced:
    st.divider()
    st.subheader("Advanced Controls")

    adv_c1, adv_c2 = st.columns(2)

    with adv_c1:
        st.markdown("**Catalogue Update**")
        if stats and stats["remaining"] == 0 and stats["total"] > 0:
            st.success("Review complete — catalogue update available.")
            if st.button("Run Catalogue Update →", use_container_width=True):
                st.switch_page("st_pages/st_pipeline.py")
        else:
            remaining = stats["remaining"] if stats else "?"
            st.info(f"{remaining} images still to review before catalogue update.")

    with adv_c2:
        st.markdown("**Reset Review Inputs**")
        st.warning("This erases all saved human inputs.")
        confirm = st.checkbox("I understand this cannot be undone")
        if confirm and st.button("Reset All Review Inputs", type="secondary", use_container_width=True):
            if "matching_results_table" in st.session_state and "human_input" in st.session_state.matching_results_table.columns:
                st.session_state.matching_results_table["human_input"] = None
                path = os.path.join(root_dir, "query_dir", "metadata_query.csv")
                st.session_state.matching_results_table.to_csv(path, index=False)
                st.success("Review inputs cleared and saved.")
                st.rerun()
