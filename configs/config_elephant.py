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
    "ear_miewid": {
        "model_id":   "conservationxlabs/miewid-msv3",   # same weights as miewid, applied to ear crop
        "dim":        2152,
        "input_size": 440,
        "loader":     "miewid",
    },
}
ACTIVE_DESCRIPTORS = ["megadescriptor", "miewid", "ear_megadescriptor", "ear_miewid"]

# Descriptors that embed the ear crop instead of the whole-animal crop.
# ear_miewid_projected is the projection-adapted variant produced by
# pipeline/transform_miewid_projection.py; it uses ear crops and must be
# treated identically to ear_miewid for crop-kind routing.
EAR_DESCRIPTORS = {"ear_megadescriptor", "ear_miewid", "ear_miewid_projected"}

# ---------------------------------------------------------------------------
# BTEH production-selected channels (frozen after model selection)
# These are separate from ACTIVE_DESCRIPTORS (which governs experimental runs).
# Do NOT add MegaDescriptor here: it received zero OOF weight and provides no
# retrieval signal for BTEH.
# ---------------------------------------------------------------------------
PRODUCTION_SELECTED_CHANNELS: list[str] = ["miewid", "ear_miewid_projected"]

PRODUCTION_FUSION_WEIGHTS: dict[str, float] = {
    "miewid": 0.6,
    "ear_miewid_projected": 0.4,
}

PRODUCTION_CALIBRATION_SUBDIR: str = "calibration_projected"

# ---------------------------------------------------------------------------
# Matching / fusion
# ---------------------------------------------------------------------------
SHORTLIST_K          = 50          # FAISS candidates passed to local re-ranker

# MATCH_ACCEPT_THRESHOLD is intentionally NOT set for BTEH production.
# The open-set threshold has FAR≈27% / FRR≈48% and is unsafe for automatic
# identity acceptance or new-identity creation.  Production must surface ranked
# top candidates for expert human verification only.
# The calibrated threshold value (0.175) is recorded in the production manifest
# for reference but must never be used to auto-accept or auto-create identities.
#
# For legacy giraffe pipeline compatibility this constant is kept but at a
# deliberately conservative value that requires expert sign-off:
MATCH_ACCEPT_THRESHOLD = 0.70      # giraffe pipeline default — NOT used for BTEH auto-matching
NUM_RECOMMENDED_IDS  = 3           # top-N to surface in UI

# Fusion weights (must sum to 1; set equal for now, tune after ablation)
FUSION_WEIGHTS = {
    "megadescriptor":     0.20,
    "miewid":             0.20,
    "ear_megadescriptor": 0.20,
    "ear_miewid":         0.20,
    "local":              0.20,
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

# Strict experimental local matcher settings
# LoFTR must be explicitly approved before use (pilot-only gate).
LOCAL_MATCHER_LOFTR_PILOT_APPROVED: bool = False

# Feature cache: maximum number of entries held in the memory LRU front-cache.
LOCAL_FEATURE_CACHE_MAX_LRU: int = 256

# Identity scorer: maximum number of reference sessions to use.
LOCAL_IDENTITY_SCORER_MAX_SESSIONS: int = 3

# Identity scorer: primary aggregation uses mean of this many top valid pairs.
LOCAL_IDENTITY_SCORER_TOP_K: int = 2

# Local score schema version (bump when schema fields change incompatibly).
LOCAL_SCORE_SCHEMA_VERSION: str = "local-v1"

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
