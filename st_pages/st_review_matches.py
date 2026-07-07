# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import io
import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_elephant import NUM_RECOMMENDED_IDS
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
# State helpers
# ---------------------------------------------------------------------------

def _auto_load():
    if "matching_results_table" in st.session_state:
        return True
    root_dir = st.session_state.root_dir
    for partition in ["query", "reference"]:
        path = os.path.join(root_dir, f"{partition}_dir", f"metadata_{partition}.csv")
        if os.path.exists(path):
            df = load_metadata_file(path)
            if partition == "query":
                st.session_state.matching_results_table = df
            else:
                st.session_state.metadata_table = df
    if "matching_results_table" not in st.session_state:
        return False
    _reset_review_state()
    return True


def _reset_review_state():
    st.session_state.rm_queue_idx = 0
    st.session_state.rm_ref_idx = 0
    st.session_state.rm_rank = 1
    st.session_state.rm_decisions_since_save = 0
    if "rm_last_saved" not in st.session_state:
        st.session_state.rm_last_saved = None


def _save():
    path = os.path.join(st.session_state.root_dir, "query_dir", "metadata_query.csv")
    st.session_state.matching_results_table.to_csv(path, index=False)
    st.session_state.rm_last_saved = datetime.now()
    st.session_state.rm_decisions_since_save = 0


# ---------------------------------------------------------------------------
# Data accessors (adapted from st_4_verify_reidentification.py)
# ---------------------------------------------------------------------------

def _get_torso_path(image_path, subdir="original_size", suffix=""):
    ext = image_path.rsplit(".", 1)[-1]
    fname = os.path.basename(image_path)
    return os.path.join(
        st.session_state.processed_img_dir, subdir, fname
    ).replace(f".{ext}", f"_cropped_torso{suffix}.{ext}")


def _get_ear_crop_path(image_path):
    """Ear crop path — same convention as step_1 / step_2: {stem}_ear_cropped.{ext}."""
    parts = image_path.rsplit(".", 1)
    ext = parts[-1] if len(parts) > 1 else "jpg"
    stem = os.path.basename(image_path).rsplit(".", 1)[0]
    return os.path.join(st.session_state.processed_img_dir, "ear_crops", f"{stem}_ear_cropped.{ext}")


def _build_queue():
    """Return (queue_paths, n_matched) where queue = matched + not_matched.

    Result is cached in session state so the ordering is stable across reruns.
    Call _invalidate_queue() to force a rebuild (e.g. after accepting a below-threshold image).
    """
    if "rm_queue_cache" in st.session_state:
        return st.session_state.rm_queue_cache

    df = st.session_state.matching_results_table
    root = st.session_state.root_dir
    sort_col = "match_local_count_1" if "match_local_count_1" in df.columns else None

    matched_df = df[df["matching_status"] == "matched"].copy()
    if sort_col:
        # kind="stable" keeps equal-inlier-count images in their original CSV order
        matched_df = matched_df.sort_values(sort_col, ascending=False, kind="stable")

    nm_df = df[df["matching_status"] == "not_matched"].copy()

    matched_paths = [os.path.join(root, p) for p in matched_df["path_relative_to_root"]]
    nm_paths = [os.path.join(root, p) for p in nm_df["path_relative_to_root"]]
    result = matched_paths + nm_paths, len(matched_paths)
    st.session_state.rm_queue_cache = result
    return result


def _invalidate_queue():
    """Force queue rebuild on next render (call after matching_status changes)."""
    st.session_state.pop("rm_queue_cache", None)


def _get_matched_label(image_path, rank):
    df = st.session_state.matching_results_table
    name = os.path.basename(image_path)
    row = df[df["path_relative_to_root"].apply(os.path.basename) == name]
    if row.empty:
        return None, None, None, None, None, None, None, None

    def _safe(col, as_float=False):
        if col not in row.columns:
            return None
        val = row[col].iloc[0]
        if isinstance(val, float) and np.isnan(val):
            return None
        if as_float:
            try:
                return float(f"{val:.4f}")
            except Exception:
                return val
        return val

    ind = _safe(f"match_individual_{rank}")
    img_id = _safe(f"match_image_{rank}")
    fused = _safe(f"match_fused_sim_{rank}", as_float=True)
    glob = _safe(f"match_global_sim_{rank}", as_float=True)
    local = _safe(f"match_local_count_{rank}")
    gt = _safe("individual_id")
    q_vp = _safe("viewpoint")
    c_vp = _safe(f"match_viewpoint_{rank}")
    return ind, img_id, fused, glob, local, gt, q_vp, c_vp


def _get_ref_paths(matched_ind, matched_img_id):
    ref_df = st.session_state.get("metadata_table")
    if ref_df is None or matched_ind is None:
        return []
    root = st.session_state.root_dir

    rows = ref_df[ref_df["individual_id"] == matched_ind]
    exact = ref_df[ref_df["image_id"] == matched_img_id] if matched_img_id is not None else ref_df.iloc[:0]

    exact_path = os.path.join(root, exact.iloc[0]["path_relative_to_root"]) if not exact.empty else None
    others_df = rows[rows["image_id"] != matched_img_id] if matched_img_id is not None else rows
    others = [os.path.join(root, p) for p in others_df["path_relative_to_root"]]
    return ([exact_path] if exact_path else []) + others


def _get_human_input(image_path):
    df = st.session_state.matching_results_table
    name = os.path.basename(image_path)
    row = df[df["path_relative_to_root"].apply(os.path.basename) == name]
    if row.empty or "human_input" not in row.columns:
        return None
    val = row["human_input"].iloc[0]
    return None if (isinstance(val, float) and np.isnan(val)) else val


def _set_human_input(image_path, value):
    df = st.session_state.matching_results_table
    name = os.path.basename(image_path)
    mask = df["path_relative_to_root"].apply(os.path.basename) == name
    st.session_state.matching_results_table.loc[mask, "human_input"] = value
    st.session_state.rm_decisions_since_save = st.session_state.get("rm_decisions_since_save", 0) + 1
    if st.session_state.rm_decisions_since_save >= 10:
        _save()


def _overwrite_matching_status(image_path):
    df = st.session_state.matching_results_table
    name = os.path.basename(image_path)
    mask = df["path_relative_to_root"].apply(os.path.basename) == name
    st.session_state.matching_results_table.loc[mask, "matching_status"] = "matched"


def _advance(queue):
    idx = st.session_state.rm_queue_idx
    if idx + 1 < len(queue):
        st.session_state.rm_queue_idx = idx + 1
        st.session_state.rm_ref_idx = 0
        st.session_state.rm_rank = 1


# ---------------------------------------------------------------------------
# Keypoint overlay
# ---------------------------------------------------------------------------

def _render_keypoint_overlay(image_path, rank):
    df = st.session_state.matching_results_table
    name = os.path.basename(image_path)
    row = df[df["path_relative_to_root"].apply(os.path.basename) == name]

    viz_col = f"viz_payload_{rank}"
    if row.empty or viz_col not in row.columns:
        st.caption("Keypoint overlay not available (run matching with local matcher).")
        return

    raw = row[viz_col].iloc[0]
    if not isinstance(raw, str) or not raw.strip():
        st.caption("Keypoint overlay not available.")
        return

    try:
        payload = json.loads(raw)
        q_kpts = payload.get("query_kpts", [])
        r_kpts = payload.get("ref_kpts", [])
        matches = payload.get("matches", [])

        ref_id_col = f"match_image_{rank}"
        ref_id = row[ref_id_col].iloc[0] if ref_id_col in row.columns else None
        ref_df = st.session_state.get("metadata_table")
        if ref_df is None or ref_id is None:
            st.caption("Keypoint overlay: reference metadata not available.")
            return

        ref_row = ref_df[ref_df["image_id"] == ref_id]
        if ref_row.empty:
            st.caption("Keypoint overlay: reference image not found.")
            return

        ref_path = os.path.join(st.session_state.root_dir, ref_row.iloc[0]["path_relative_to_root"])
        q_crop = _get_torso_path(image_path, "zoomed_version", "_zoomed")

        if not os.path.isfile(q_crop) or not os.path.isfile(ref_path):
            st.caption("Keypoint overlay: crop files not on disk.")
            return

        q_img = ImageOps.exif_transpose(Image.open(q_crop)).resize((256, 256))
        r_img = ImageOps.exif_transpose(Image.open(ref_path)).resize((256, 256))

        q_arr = np.array(q_img.convert("RGB"))
        r_arr = np.array(r_img.convert("RGB"))
        canvas = np.zeros((256, 512, 3), dtype=np.uint8)
        canvas[:, :256] = q_arr
        canvas[:, 256:] = r_arr

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(canvas)
        ax.axis("off")
        colors = plt.cm.rainbow(np.linspace(0, 1, max(len(matches), 1)))
        for (qi, ri), color in zip(matches, colors):
            qx, qy = q_kpts[qi]
            rx, ry = r_kpts[ri]
            ax.plot([qx, rx + 256], [qy, ry], color=color, linewidth=0.8, alpha=0.7)
            ax.scatter([qx], [qy], c=[color], s=8, zorder=5)
            ax.scatter([rx + 256], [ry], c=[color], s=8, zorder=5)

        plt.tight_layout(pad=0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        st.image(buf.read(), caption=f"LightGlue keypoints — rank {rank}", use_container_width=True)

    except Exception as exc:
        st.caption(f"Keypoint overlay failed: {exc}")


# ---------------------------------------------------------------------------
# Image loading helper
# ---------------------------------------------------------------------------

@st.cache_data(max_entries=600, show_spinner=False)
def _load_image_bytes(path: str, size: int) -> bytes | None:
    """Load, resize, and return JPEG bytes. Cached so disk is only hit once per path."""
    for candidate in [path]:
        if not os.path.isfile(candidate):
            continue
        try:
            img = ImageOps.exif_transpose(Image.open(candidate)).convert("RGB")
            w, h = img.size
            if h > 0:
                scale = size / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
        except Exception:
            return None
    return None


def _load_display_image(image_path: str, size: int = 300) -> bytes | None:
    """Try zoomed torso → plain torso → full image. Returns JPEG bytes or None."""
    zoomed = _get_torso_path(image_path, "zoomed_version", "_zoomed")
    plain = _get_torso_path(image_path)
    for candidate in [zoomed, plain, image_path]:
        if os.path.isfile(candidate):
            return _load_image_bytes(candidate, size)
    return None


# ---------------------------------------------------------------------------
# Main review UI
# ---------------------------------------------------------------------------

def main():
    if not _auto_load():
        st.warning("No project found. Go to **Dashboard** to create a query table.")
        return

    df = st.session_state.matching_results_table
    if "matching_status" not in df.columns:
        st.warning("Matching has not been run yet. Go to **Run Analysis**.")
        return

    if "rm_queue_idx" not in st.session_state:
        _reset_review_state()

    queue, n_matched = _build_queue()
    if not queue:
        st.info("No images to review.")
        return

    n_total = len(queue)
    # Vectorized count — much faster than per-row lookups
    df = st.session_state.matching_results_table
    queue_mask = df["matching_status"].isin(["matched", "not_matched"])
    if "human_input" in df.columns:
        reviewed = int(df.loc[queue_mask, "human_input"].notna().sum())
    else:
        reviewed = 0

    # --- Header bar ---
    h1, h2, h3 = st.columns([5, 2, 1])
    with h1:
        st.progress(reviewed / n_total, text=f"{reviewed} / {n_total} reviewed — {n_total - reviewed} remaining")
    with h2:
        last = st.session_state.get("rm_last_saved")
        unsaved = st.session_state.get("rm_decisions_since_save", 0)
        if last:
            mins = max(0, int((datetime.now() - last).total_seconds() // 60))
            st.caption(f"{'Saved ' + str(mins) + 'm ago' if mins > 0 else 'Just saved'}"
                       + (" · unsaved changes" if unsaved else ""))
        elif unsaved:
            st.caption("Unsaved changes")
    with h3:
        if st.button("Save", use_container_width=True):
            _save()
            st.success("Saved!")

    # --- Navigation ---
    idx = min(st.session_state.rm_queue_idx, n_total - 1)
    st.session_state.rm_queue_idx = idx

    nc1, nc2, nc3, nc4, nc5 = st.columns([1, 1, 3, 1, 1])
    with nc1:
        if st.button("◀◀ −100"):
            st.session_state.rm_queue_idx = max(0, idx - 100)
            st.session_state.rm_ref_idx = 0
            st.session_state.rm_rank = 1
            st.rerun()
    with nc2:
        if st.button("◀ Prev"):
            st.session_state.rm_queue_idx = max(0, idx - 1)
            st.session_state.rm_ref_idx = 0
            st.session_state.rm_rank = 1
            st.rerun()
    with nc3:
        queue_type = "matched" if idx < n_matched else "not-matched"
        st.markdown(f"<div style='text-align:center;padding-top:8px'>{idx + 1} / {n_total} &nbsp;·&nbsp; <em>{queue_type}</em></div>", unsafe_allow_html=True)
    with nc4:
        if st.button("Next ▶"):
            st.session_state.rm_queue_idx = min(n_total - 1, idx + 1)
            st.session_state.rm_ref_idx = 0
            st.session_state.rm_rank = 1
            st.rerun()
    with nc5:
        if st.button("+100 ▶▶"):
            st.session_state.rm_queue_idx = min(n_total - 1, idx + 100)
            st.session_state.rm_ref_idx = 0
            st.session_state.rm_rank = 1
            st.rerun()

    idx = st.session_state.rm_queue_idx
    current = queue[idx]
    is_not_matched = idx >= n_matched
    rank = st.session_state.get("rm_rank", 1)

    # Candidate info for current rank
    ind, img_id, fused, glob, local, gt, q_vp, c_vp = _get_matched_label(current, rank)
    ref_paths = _get_ref_paths(ind, img_id)
    ref_idx = min(st.session_state.get("rm_ref_idx", 0), max(0, len(ref_paths) - 1))
    st.session_state.rm_ref_idx = ref_idx

    st.divider()

    # --- Two-column card ---
    col_q, col_r = st.columns(2, gap="large")

    with col_q:
        st.markdown("**Query Image**")
        img = _load_display_image(current, size=300)
        if img:
            st.image(img, caption=os.path.basename(current), use_container_width=False)
        else:
            st.warning(f"Image not found: {os.path.basename(current)}")
        if q_vp:
            st.caption(f"Viewpoint: {q_vp}")

    with col_r:
        if ind:
            label = ("NOT MATCHED — AI suggested" if is_not_matched else "TOP MATCH")
            st.markdown(f"**{label}:** {ind}")
            if fused is not None:
                conf_pct = min(int(fused * 100), 100)
                st.progress(conf_pct / 100, text=f"Confidence: {conf_pct}%")
        else:
            st.markdown("**No candidate found**")

        if ref_paths:
            ref_img = _load_display_image(ref_paths[ref_idx], size=300)
            caption = f"Reference {ref_idx + 1}/{len(ref_paths)}: {os.path.basename(ref_paths[ref_idx])}"
            if ref_img:
                st.image(ref_img, caption=caption, use_container_width=False)
            else:
                st.warning("Reference image not found on disk.")

            # Reference gallery navigation
            rc1, rc2, rc3 = st.columns([1, 2, 1])
            with rc1:
                if st.button("◀", key="ref_prev"):
                    st.session_state.rm_ref_idx = max(0, ref_idx - 1)
                    st.rerun()
            with rc2:
                st.caption(f"ref {ref_idx + 1} / {len(ref_paths)}")
            with rc3:
                if st.button("▶", key="ref_next"):
                    st.session_state.rm_ref_idx = min(len(ref_paths) - 1, ref_idx + 1)
                    st.rerun()

            if c_vp:
                st.caption(f"Candidate viewpoint: {c_vp}")
        else:
            st.info("No reference images for this candidate.")

    # --- Candidate rank strip ---
    st.divider()
    n_recomms = st.session_state.get("num_id_recomms", NUM_RECOMMENDED_IDS)
    rank_cols = st.columns(n_recomms)
    for r in range(1, n_recomms + 1):
        r_ind, _, r_fused, _, _, _, _, _ = _get_matched_label(current, r)
        label = f"#{r} {r_ind or 'N/A'}"
        if r_fused is not None:
            label += f" ({int(r_fused * 100)}%)"
        with rank_cols[r - 1]:
            btn_type = "primary" if rank == r else "secondary"
            if st.button(label, key=f"rank_{r}", type=btn_type, use_container_width=True):
                st.session_state.rm_rank = r
                st.session_state.rm_ref_idx = 0
                st.rerun()

    # --- Expandable panels ---
    is_advanced = st.session_state.get("mode", "field") == "advanced"

    with st.expander("Keypoint Overlay", expanded=False):
        _render_keypoint_overlay(current, rank)

    with st.expander("Ear Crops", expanded=False):
        ec1, ec2 = st.columns(2)
        q_ear = _get_ear_crop_path(current)
        with ec1:
            st.caption("Query ear")
            q_ear_bytes = _load_image_bytes(q_ear, 220) if os.path.isfile(q_ear) else None
            if q_ear_bytes:
                st.image(q_ear_bytes, use_container_width=False)
            else:
                st.caption("Not available")
        with ec2:
            st.caption(f"Reference ear — {ind or 'N/A'}")
            ref_ear_shown = False
            if img_id is not None:
                ref_df = st.session_state.get("metadata_table")
                if ref_df is not None:
                    ref_row = ref_df[ref_df["image_id"] == img_id]
                    if not ref_row.empty:
                        ref_full = os.path.join(
                            st.session_state.root_dir,
                            ref_row.iloc[0]["path_relative_to_root"],
                        )
                        r_ear = _get_ear_crop_path(ref_full)
                        r_ear_bytes = _load_image_bytes(r_ear, 220) if os.path.isfile(r_ear) else None
                        if r_ear_bytes:
                            st.image(r_ear_bytes, use_container_width=False)
                            ref_ear_shown = True
            if not ref_ear_shown:
                st.caption("Not available")

    with st.expander("Evidence", expanded=is_advanced):
        ev1, ev2 = st.columns(2)
        with ev1:
            st.write(f"**Matched ID:** {ind or 'N/A'}")
            st.write(f"**Matched Image:** {img_id or 'N/A'}")
            st.write(f"**Fused Score:** {fused if fused is not None else 'N/A'}")
        with ev2:
            st.write(f"**Global Sim:** {glob if glob is not None else 'N/A'}")
            st.write(f"**Local Inliers:** {local if local is not None else 'N/A'}")
            st.write(f"**Ground Truth:** {gt or 'N/A'}")
        current_hi = _get_human_input(current)
        st.write(f"**Current decision:** {current_hi or 'none'}")

    # --- Action buttons ---
    st.divider()

    if is_not_matched:
        accept_label = "✓ Accept Candidate Anyway"
        reject_label = "✗ Assign New ID"
    else:
        accept_label = "✓ Accept Matched ID"
        reject_label = "✗ Assign New ID"

    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        if st.button(accept_label, type="primary", use_container_width=True):
            _set_human_input(current, "AcceptId")
            if is_not_matched:
                _overwrite_matching_status(current)
                _invalidate_queue()  # image moves from not-matched → matched section
            _advance(queue)
            st.rerun()
    with ac2:
        if st.button(reject_label, use_container_width=True):
            _set_human_input(current, "AssignNewId")
            _advance(queue)
            st.rerun()
    with ac3:
        if st.button("→ Skip", use_container_width=True):
            _set_human_input(current, "SkipImage")
            _advance(queue)
            st.rerun()

    # --- Advanced: bulk actions ---
    if is_advanced:
        with st.expander("Bulk Actions (Advanced)", expanded=False):
            matched_paths = queue[:n_matched]
            nm_paths = queue[n_matched:]
            unreviewed_m = [p for p in matched_paths if _get_human_input(p) is None]
            unreviewed_nm = [p for p in nm_paths if _get_human_input(p) is None]

            bc1, bc2 = st.columns(2)
            with bc1:
                if st.button(
                    f"Accept All Matched ({len(unreviewed_m)} unreviewed)",
                    use_container_width=True,
                ):
                    basenames = {os.path.basename(p) for p in unreviewed_m}
                    mask = st.session_state.matching_results_table[
                        "path_relative_to_root"
                    ].apply(os.path.basename).isin(basenames)
                    st.session_state.matching_results_table.loc[mask, "human_input"] = "AcceptId"
                    _save()
                    st.success(f"Accepted {len(unreviewed_m)} matched images.")
                    st.rerun()
            with bc2:
                if st.button(
                    f"Assign New ID to All Not-Matched ({len(unreviewed_nm)} unreviewed)",
                    use_container_width=True,
                ):
                    basenames = {os.path.basename(p) for p in unreviewed_nm}
                    mask = st.session_state.matching_results_table[
                        "path_relative_to_root"
                    ].apply(os.path.basename).isin(basenames)
                    st.session_state.matching_results_table.loc[mask, "human_input"] = "AssignNewId"
                    _save()
                    st.success(f"Assigned new ID to {len(unreviewed_nm)} not-matched images.")
                    st.rerun()


main()
