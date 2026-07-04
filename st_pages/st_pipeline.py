# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_matching import pipeline_code_relative_dir
from utils.helpers_matching import load_data_dirs

from user_authentication import login_ui, authorize_users

if authorize_users() and not st.session_state.get("authenticated", False):
    st.markdown("""<style>[data-testid="stSidebar"] { display: none; }</style>""", unsafe_allow_html=True)
    login_ui()
    st.stop()

if "root_dir" not in st.session_state or not st.session_state.root_dir:
    st.session_state.root_dir, st.session_state.processed_img_dir = load_data_dirs()

pipeline_code_dir = os.path.join(str(Path(__file__).resolve().parent.parent), str(pipeline_code_relative_dir))
is_advanced = st.session_state.get("mode", "field") == "advanced"


# ---------------------------------------------------------------------------
# Helpers (adapted from st_2 / st_3 / st_5 / st_7)
# ---------------------------------------------------------------------------

def _run_script(pycode_name, additional_args=None):
    additional_args = additional_args or []
    result = subprocess.run(
        ["bash", "./setup_pipeline.sh", os.path.join(pipeline_code_dir, pycode_name)] + additional_args,
        capture_output=True, text=True,
    )
    return result.stdout, result.stderr


def _check_job_running():
    result = subprocess.run(["pgrep", "-f", "python.*step_"], capture_output=True, text=True)
    return result.returncode == 0


def _stop_job():
    subprocess.run(["pkill", "-f", "python.*step_"], capture_output=True)


def _render_log_section(script_name):
    """Show raw log if available."""
    log_dir = os.path.join(st.session_state.root_dir, "query_dir", "logs")
    if not os.path.isdir(log_dir):
        return
    logs = sorted(
        [f for f in os.listdir(log_dir) if script_name in f and "std_output" in f],
        reverse=True,
    )
    if logs:
        log_path = os.path.join(log_dir, logs[0])
        with st.expander(f"Log: {logs[0]}", expanded=False):
            try:
                with open(log_path) as f:
                    content = f.read()
                st.text(content[-5000:] if len(content) > 5000 else content)
            except Exception as exc:
                st.caption(f"Could not read log: {exc}")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("Run Analysis")

running = _check_job_running()
if running:
    st.warning("A pipeline job is currently running.")
    if is_advanced and st.button("Stop Current Job", type="secondary"):
        _stop_job()
        st.success("Stop signal sent.")
        st.rerun()

st.divider()

STEPS = [
    {
        "label": "Step 1 — Detect & Crop",
        "desc": "Run MegaDetector to detect elephants and crop whole-body patches.",
        "script": "step_1_detect_and_crop.py",
        "partitioned": True,
    },
    {
        "label": "Step 2 — Extract Features",
        "desc": "Extract MiewID, MegaDescriptor, and ear MegaDescriptor embeddings.",
        "script": "step_2_extract_features.py",
        "partitioned": True,
    },
    {
        "label": "Step 3 — Run Matching",
        "desc": "Run WildFusion (global + LightGlue local) matching against the reference catalogue.",
        "script": "step_3_run_initial_matching.py",
        "partitioned": False,
    },
    {
        "label": "Step 4 — Cluster Unknowns",
        "desc": "Partition images that fell below the matching threshold into unknown-individual clusters.",
        "script": "step_4_partition_new_items.py",
        "partitioned": False,
    },
    {
        "label": "Step 5 — Update Catalogue",
        "desc": "Write accepted identities and confirmed new individuals into the reference catalogue.\n"
                "Requires review to be complete.",
        "script": "step_6_update_database.py",
        "partitioned": False,
        "advanced_only": True,
    },
]

for step in STEPS:
    if step.get("advanced_only") and not is_advanced:
        continue

    with st.expander(step["label"], expanded=False):
        st.write(step["desc"])

        if step["partitioned"] and is_advanced:
            partition = st.radio(
                "Partition",
                ["Query Data", "Reference Catalogue"],
                horizontal=True,
                key=f"partition_{step['script']}",
            )
            partition_arg = ["query" if partition == "Query Data" else "reference"]
        else:
            partition_arg = []

        bc1, bc2 = st.columns([1, 3])
        with bc1:
            if st.button(f"Run", key=f"run_{step['script']}", disabled=running, use_container_width=True):
                with st.spinner(f"Running {step['script']} …"):
                    stdout, stderr = _run_script(step["script"], partition_arg)
                st.success("Done.")
                if stdout:
                    st.text_area("Output", stdout[-3000:], height=200)
                if stderr:
                    with st.expander("Errors / warnings"):
                        st.text(stderr[-2000:])

        if is_advanced:
            _render_log_section(step["script"])

if is_advanced:
    st.divider()
    st.caption("Stop button above terminates any running step_*.py process.")
