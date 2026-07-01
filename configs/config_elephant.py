# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
# Species profile for elephant re-identification.
# All pipeline code imports constants from here so the framework stays
# species-agnostic.  Do NOT put Azure credentials or data paths here.
# -------------------------------------------------------------------------

import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Identity columns  (replaces hardcoded 'AID2021' / '#Serial')
# ---------------------------------------------------------------------------
ID_COL          = "individual_id"       # e.g. "eleph_kunene"
IMAGE_ID_COL    = "image_id"            # UUID matching blob filename stem
VIEWPOINT_COL   = "viewpoint"
VIEWPOINT_VALUES = ["left", "right", "frontal", "rear", "unknown"]

# ---------------------------------------------------------------------------
# Global descriptors
# Each entry:  model_id (HuggingFace repo), embedding dim, input image size
# ACTIVE_DESCRIPTORS controls which ones are actually run at embedding time
# and included in the fusion ensemble.
# ---------------------------------------------------------------------------
GLOBAL_DESCRIPTORS = {
    "megadescriptor": {
        "model_id":   "BVRA/MegaDescriptor-L-384",
        "dim":        1536,
        "input_size": 384,
        "loader":     "megadescriptor",
    },
    "miewid": {
        "model_id":   "conservationxlabs/miewid-msv3",
        "dim":        2152,
        "input_size": 440,
        "loader":     "miewid",
    },
    "ear_megadescriptor": {
        "model_id":   "BVRA/MegaDescriptor-L-384",   # same weights, applied to ear crop
        "dim":        1536,
        "input_size": 384,
        "loader":     "megadescriptor",
    },
}
ACTIVE_DESCRIPTORS = ["megadescriptor", "miewid", "ear_megadescriptor"]

# Descriptors that embed the ear crop instead of the whole-animal crop
EAR_DESCRIPTORS = {"ear_megadescriptor"}

# ---------------------------------------------------------------------------
# Matching / fusion
# ---------------------------------------------------------------------------
SHORTLIST_K          = 50          # FAISS candidates passed to local re-ranker
MATCH_ACCEPT_THRESHOLD = 0.65      # fused calibrated score → "matched"
NUM_RECOMMENDED_IDS  = 3           # top-N to surface in UI

# Fusion weights (must sum to 1; set equal for now, tune after ablation)
FUSION_WEIGHTS = {
    "megadescriptor":     0.25,
    "miewid":             0.25,
    "ear_megadescriptor": 0.25,
    "local":              0.25,
}

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
CALIBRATION_METHOD              = "isotonic"   # "isotonic" | "temperature"
MIN_POSITIVE_PAIRS_FOR_ISOTONIC = 200          # fall back to temperature below this
CALIBRATION_DIR                 = "calibration"

# ---------------------------------------------------------------------------
# Local matcher
# ---------------------------------------------------------------------------
LOCAL_MATCHER_BACKEND   = "lightglue"   # "lightglue" | "loftr"
LOCAL_MATCHER_KEYPOINTS = 2048
LOCAL_MATCHER_MIN_INLIERS = 15

# ---------------------------------------------------------------------------
# New individual ID minting
# ---------------------------------------------------------------------------
NEW_ID_PREFIX = "eleph_unk_"    # new unknown individual: eleph_unk_<uuid8>

# ---------------------------------------------------------------------------
# Ground-truth column name (for evaluation; may be absent in production)
# ---------------------------------------------------------------------------
GT_COL = ID_COL     # same column — individual_id is ground truth when known

# ---------------------------------------------------------------------------
# Data artifact paths (relative to partition root dir)
# ---------------------------------------------------------------------------
EMBEDDINGS_SUBDIR       = "embeddings"
FAISS_SUBDIR            = "faiss_index"
LOCAL_FEATURES_SUBDIR   = "local_features"
INDEX_PARQUET_FILENAME  = "index.parquet"      # one parquet per partition
CROP_SUBDIR             = "processed_images"   # whole-animal crops
EAR_CROP_SUBDIR         = "processed_images/ear_crops"  # GroundingDINO ear crops

# ---------------------------------------------------------------------------
# Sharding  (reuse from giraffe; swap ID column)
# ---------------------------------------------------------------------------
MIN_SHARD_SIZE = 7000
MAX_SHARD_SIZE = 15000

# ---------------------------------------------------------------------------
# UI / pipeline bookkeeping
# ---------------------------------------------------------------------------
PIPELINE_CODE_RELATIVE_DIR = "pipeline"
README_UI_FILE             = "../docs/README_UI.md"
