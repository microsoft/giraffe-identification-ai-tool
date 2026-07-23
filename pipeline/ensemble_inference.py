#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Full-ensemble fusion inference module.

This module is the PRIMARY scorer for fixed-probe inference once the OOF
calibration pipeline has produced frozen artifacts.  It loads the frozen K,
local Platt calibrators, and fusion weights, then applies the EXACT same
candidate and local scoring logic as the OOF calibration pipeline.

Design contracts
----------------
* Loads frozen artifacts from an OOF calibration output directory.
* Uses the same global shortlisting, local scoring (body + ear), and
  4-channel fusion as run_oof_calibration().
* No probe execution is performed in this module — the run_inference CLI
  stub validates artifacts and reports their provenance without scoring.
* The score_query_candidates() function is the single entry point for
  both OOF evaluation (via run_oof_calibration) and fixed-probe inference
  (future), ensuring exact function parity.
* selected-v1 production calibrators/weights are read-only; never mutated.

Usage (check / validate artifacts)
-----------------------------------
    python pipeline/ensemble_inference.py check \\
        --artifacts-dir PATH

Usage (future inference — not executed here)
--------------------------------------------
    python pipeline/ensemble_inference.py score \\
        --artifacts-dir PATH --query-image PATH \\
        --gallery-dir PATH
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_elephant import (
    LOCAL_IDENTITY_SCORER_MAX_SESSIONS,
    LOCAL_IDENTITY_SCORER_TOP_K,
    LOCAL_SCORE_SCHEMA_VERSION,
    PRODUCTION_FUSION_WEIGHTS,
    PRODUCTION_SELECTED_CHANNELS,
)
from models.calibration import Calibrator
from models.identity_scorer import LocalIdentityScorer, QueryCrop, ReferenceImage
from models.identity_fusion import IdentityScore, QueryResult, _cosine_identity_max
from models.local_matcher import REGION_BODY, REGION_EAR
from pipeline.local_oof_calibration import (
    ALL_CHANNELS,
    CHANNEL_BODY_LOCAL,
    CHANNEL_EAR_LOCAL,
    CHANNEL_EAR,
    CHANNEL_MIEWID,
    GLOBAL_CHANNELS,
    OOF_CONFIG_JSON,
    OOF_FINGERPRINT_JSON,
    OOF_METRICS_JSON,
    WEIGHTS_JSON,
    ProbePollutionError,
    _assert_no_probe_ids,
    _build_query_crops,
    _filter_gallery_only,
    _shortlist_fingerprint,
    build_shortlist_registration,
    compute_global_oof_rankings,
    load_oof_artifacts,
    score_local_for_candidate,
    select_globally_strongest_refs,
    select_k_threshold,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frozen artifact container
# ---------------------------------------------------------------------------

@dataclass
class EnsembleArtifacts:
    """
    All frozen artifacts required for full-ensemble inference.

    Loaded from an OOF calibration output directory.  All fields should be
    treated as read-only once loaded.
    """
    frozen_k: int
    fusion_weights: Dict[str, float]
    calibrators_global: Dict[str, Calibrator]      # miewid, ear_miewid_projected
    calibrator_body: Optional[Calibrator]          # body_local Platt
    calibrator_ear: Optional[Calibrator]           # ear_local Platt
    oof_metrics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    fingerprint: Dict[str, Any] = field(default_factory=dict)
    all_channels: List[str] = field(default_factory=lambda: list(ALL_CHANNELS))


def load_ensemble_artifacts(
    artifacts_dir: Path,
    calibrators_global: Optional[Dict[str, Calibrator]] = None,
) -> EnsembleArtifacts:
    """
    Load frozen OOF artifacts from *artifacts_dir*.

    Parameters
    ----------
    artifacts_dir : path to the OOF calibration output directory.
    calibrators_global : pre-loaded selected-v1 global calibrators.
        If None, placeholder (unfitted) Calibrators are created; callers
        must supply real calibrators for actual scoring.

    Returns
    -------
    EnsembleArtifacts
    """
    raw = load_oof_artifacts(artifacts_dir)

    weights = raw["fusion_weights"]
    frozen_k = raw["oof_metrics"].get("frozen_k", 50)
    config_dict = raw["config"]
    all_channels = config_dict.get("all_channels", list(ALL_CHANNELS))

    return EnsembleArtifacts(
        frozen_k=frozen_k,
        fusion_weights=weights,
        calibrators_global=calibrators_global or {},
        calibrator_body=raw["calibrator_body"],
        calibrator_ear=raw["calibrator_ear"],
        oof_metrics=raw["oof_metrics"],
        config=config_dict,
        fingerprint=raw["fingerprint"],
        all_channels=all_channels,
    )


# ---------------------------------------------------------------------------
# Core query scoring (exact parity with OOF calibration path)
# ---------------------------------------------------------------------------

def score_query_candidates(
    query_image_id: str,
    query_session_id: str,
    candidate_individual_ids: List[str],
    gallery_df: pd.DataFrame,
    crop_df: pd.DataFrame,
    embedding_matrices: Dict[str, np.ndarray],
    descriptor_mappings: Dict[str, pd.DataFrame],
    artifacts: EnsembleArtifacts,
    local_scorer_body: Optional[LocalIdentityScorer],
    local_scorer_ear: Optional[LocalIdentityScorer],
    *,
    query_individual_id: Optional[str] = None,
    source_fingerprint: str = "",
    split_fingerprint: str = "",
) -> List[IdentityScore]:
    """
    Score *candidate_individual_ids* for *query_image_id*, fusing all
    4 channels with the frozen calibrators and weights.

    This function is the SINGLE entry point for both:
      - OOF evaluation (called from run_oof_calibration via build_oof_table)
      - Fixed-probe inference (called from this module's score command)

    Parameters
    ----------
    query_image_id : image to score
    query_session_id : session of the query (excluded from reference sets)
    candidate_individual_ids : identities to rank
    gallery_df : gallery-only image DataFrame
    crop_df : crop manifest DataFrame
    embedding_matrices : {channel: np.ndarray} L2-normalised
    descriptor_mappings : {channel: DataFrame} descriptor mapping
    artifacts : frozen EnsembleArtifacts
    local_scorer_body, local_scorer_ear : canonical LocalIdentityScorer instances
    query_individual_id : ground-truth identity (optional, for evaluation)

    Returns
    -------
    List of IdentityScore sorted by fused_score descending.
    """
    emb_miewid = embedding_matrices.get(CHANNEL_MIEWID)
    dm_miewid = descriptor_mappings.get(CHANNEL_MIEWID)

    all_channels = artifacts.all_channels
    weights = artifacts.fusion_weights

    identity_scores: List[IdentityScore] = []

    for cand_id in candidate_individual_ids:
        ch_raw: Dict[str, float] = {}
        ch_cal: Dict[str, float] = {}

        # --- Global channel scores ---
        for ch_name, col_raw, col_cal in [
            (CHANNEL_MIEWID, "global_miewid_raw", "global_miewid_calibrated"),
            (CHANNEL_EAR, "global_ear_raw", "global_ear_calibrated"),
        ]:
            emb_mat = embedding_matrices.get(ch_name)
            dm = descriptor_mappings.get(ch_name)
            if emb_mat is None or dm is None:
                continue

            dm = dm.copy()
            dm["image_id"] = dm["image_id"].astype(str)
            gal_ids_set = set(gallery_df["image_id"].astype(str)) - set(
                gallery_df.loc[
                    gallery_df["session_id"].astype(str) == str(query_session_id),
                    "image_id"
                ].astype(str)
            )
            image_to_indiv = dict(
                zip(gallery_df["image_id"].astype(str), gallery_df["individual_id"].astype(str))
            )
            gm = dm[dm["image_id"].isin(gal_ids_set)]
            if gm.empty:
                continue

            # Query embedding
            q_dm = dm[dm["image_id"] == str(query_image_id)]
            if q_dm.empty:
                continue
            q_rows = q_dm["embedding_row"].astype(int).to_numpy()
            q_emb = emb_mat[q_rows]

            # Reference embeddings for this candidate
            cand_imgs = gallery_df[
                (gallery_df["individual_id"].astype(str) == str(cand_id))
                & (gallery_df["session_id"].astype(str) != str(query_session_id))
            ]
            if cand_imgs.empty:
                continue
            cand_img_ids = set(cand_imgs["image_id"].astype(str))
            r_dm = gm[gm["image_id"].isin(cand_img_ids)]
            if r_dm.empty:
                continue
            r_rows = r_dm["embedding_row"].astype(int).to_numpy()
            r_emb = emb_mat[r_rows]

            raw_score = float(np.max(q_emb @ r_emb.T))
            ch_raw[ch_name] = raw_score

            cal = artifacts.calibrators_global.get(ch_name)
            if cal is not None:
                ch_cal[ch_name] = float(cal.transform(np.array([raw_score]))[0])
            else:
                ch_cal[ch_name] = raw_score

        # --- Local scores ---
        local_result = score_local_for_candidate(
            query_image_id=query_image_id,
            fold_session_id=query_session_id,
            candidate_individual_id=cand_id,
            gallery_df=gallery_df,
            crop_df=crop_df,
            emb_matrix_miewid=emb_miewid,
            desc_mapping_miewid=dm_miewid,
            local_scorer_body=local_scorer_body,
            local_scorer_ear=local_scorer_ear,
            source_fingerprint=source_fingerprint,
            split_fingerprint=split_fingerprint,
        )

        if (
            local_result.get("body_local_available", False)
            and pd.notna(local_result.get("body_local_score"))
            and artifacts.calibrator_body is not None
        ):
            raw_b = float(local_result["body_local_score"])
            ch_raw[CHANNEL_BODY_LOCAL] = raw_b
            ch_cal[CHANNEL_BODY_LOCAL] = float(
                artifacts.calibrator_body.transform(np.array([raw_b]))[0]
            )

        if (
            local_result.get("ear_local_available", False)
            and pd.notna(local_result.get("ear_local_score"))
            and artifacts.calibrator_ear is not None
        ):
            raw_e = float(local_result["ear_local_score"])
            ch_raw[CHANNEL_EAR_LOCAL] = raw_e
            ch_cal[CHANNEL_EAR_LOCAL] = float(
                artifacts.calibrator_ear.transform(np.array([raw_e]))[0]
            )

        available = [ch for ch in all_channels if ch in ch_cal]
        total_w = sum(weights.get(ch, 0.0) for ch in available)
        if total_w > 0:
            fused = sum(weights.get(ch, 0.0) * ch_cal[ch] / total_w for ch in available)
        else:
            fused = 0.0

        identity_scores.append(
            IdentityScore(
                individual_id=str(cand_id),
                channel_raw=ch_raw,
                channel_calibrated=ch_cal,
                channels_available=available,
                fused_score=float(fused),
            )
        )

    identity_scores.sort(key=lambda x: x.fused_score, reverse=True)
    return identity_scores


# ---------------------------------------------------------------------------
# EnsembleScorer (unified interface)
# ---------------------------------------------------------------------------

class EnsembleScorer:
    """
    Full-ensemble identity scorer using frozen OOF artifacts.

    Provides the same scoring interface for both OOF evaluation and
    fixed-probe inference.  All scoring logic is delegated to
    score_query_candidates() to ensure exact function parity.

    Parameters
    ----------
    artifacts : EnsembleArtifacts loaded from OOF calibration output.
    gallery_df : gallery-only image DataFrame.
    crop_df : crop manifest DataFrame.
    embedding_matrices : {channel: np.ndarray}.
    descriptor_mappings : {channel: DataFrame}.
    local_scorer_body : canonical LocalIdentityScorer for body crops.
    local_scorer_ear : canonical LocalIdentityScorer for ear crops.
    """

    def __init__(
        self,
        artifacts: EnsembleArtifacts,
        gallery_df: pd.DataFrame,
        crop_df: pd.DataFrame,
        embedding_matrices: Dict[str, np.ndarray],
        descriptor_mappings: Dict[str, pd.DataFrame],
        local_scorer_body: Optional[LocalIdentityScorer],
        local_scorer_ear: Optional[LocalIdentityScorer],
    ):
        _assert_no_probe_ids(gallery_df, context="EnsembleScorer gallery_df")
        self.artifacts = artifacts
        self.gallery_df = gallery_df.copy()
        self.gallery_df["image_id"] = self.gallery_df["image_id"].astype(str)
        self.gallery_df["individual_id"] = self.gallery_df["individual_id"].astype(str)
        self.gallery_df["session_id"] = self.gallery_df["session_id"].astype(str)
        self.crop_df = crop_df
        self.embedding_matrices = embedding_matrices
        self.descriptor_mappings = descriptor_mappings
        self.local_scorer_body = local_scorer_body
        self.local_scorer_ear = local_scorer_ear

        self._all_gallery_ids: List[str] = sorted(
            self.gallery_df["individual_id"].unique()
        )

    def score(
        self,
        query_image_id: str,
        query_session_id: str,
        *,
        candidate_ids: Optional[List[str]] = None,
        query_individual_id: Optional[str] = None,
        source_fingerprint: str = "",
        split_fingerprint: str = "",
    ) -> List[IdentityScore]:
        """
        Score *query_image_id* against gallery identities.

        If *candidate_ids* is None, all gallery identities are scored;
        otherwise, only the provided candidates are scored.  The frozen K
        is not applied here — callers should pre-compute the shortlist via
        the global scorer if needed.

        Returns list of IdentityScore sorted by fused_score descending.
        """
        cands = candidate_ids if candidate_ids is not None else self._all_gallery_ids
        return score_query_candidates(
            query_image_id=query_image_id,
            query_session_id=query_session_id,
            candidate_individual_ids=cands,
            gallery_df=self.gallery_df,
            crop_df=self.crop_df,
            embedding_matrices=self.embedding_matrices,
            descriptor_mappings=self.descriptor_mappings,
            artifacts=self.artifacts,
            local_scorer_body=self.local_scorer_body,
            local_scorer_ear=self.local_scorer_ear,
            query_individual_id=query_individual_id,
            source_fingerprint=source_fingerprint,
            split_fingerprint=split_fingerprint,
        )

    @property
    def frozen_k(self) -> int:
        return self.artifacts.frozen_k

    @property
    def fusion_weights(self) -> Dict[str, float]:
        return dict(self.artifacts.fusion_weights)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Full-ensemble fusion inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # check: validate artifacts, no probe execution
    check = sub.add_parser(
        "check",
        help="Validate frozen OOF artifacts (no probe execution).",
    )
    check.add_argument(
        "--artifacts-dir",
        required=True,
        help="Path to the OOF calibration output directory.",
    )

    # score: stub for future fixed-probe inference
    score = sub.add_parser(
        "score",
        help="[NOT IMPLEMENTED] Fixed-probe inference (future).",
    )
    score.add_argument("--artifacts-dir", required=True)
    score.add_argument("--query-image", required=True)
    score.add_argument("--gallery-dir", required=True)

    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        artifacts_dir = Path(args.artifacts_dir)
        try:
            raw = load_oof_artifacts(artifacts_dir)
        except FileNotFoundError as exc:
            logger.error("Artifact validation failed: %s", exc)
            return 1

        print(json.dumps({
            "status": "ok",
            "artifacts_dir": str(artifacts_dir),
            "frozen_k": raw["oof_metrics"].get("frozen_k"),
            "fusion_weights": raw["fusion_weights"],
            "calibrator_body_fitted": raw["calibrator_body"] is not None,
            "calibrator_ear_fitted": raw["calibrator_ear"] is not None,
            "pipeline_fingerprint": raw["fingerprint"].get("config_fingerprint"),
            "schema_version": raw["fingerprint"].get("schema_version"),
            "saved_at": raw["fingerprint"].get("saved_at"),
        }, indent=2))
        return 0

    if args.command == "score":
        logger.warning(
            "Fixed-probe inference is not implemented in this release. "
            "Load EnsembleArtifacts via load_ensemble_artifacts() and call "
            "EnsembleScorer.score() from application code."
        )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
