# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

import os
import sys
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
import faiss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_elephant import (
    FUSION_WEIGHTS,
    SHORTLIST_K,
    NUM_RECOMMENDED_IDS,
    ACTIVE_DESCRIPTORS,
)
from models.embedder import GlobalEmbedder
from models.local_matcher import LocalMatcher
from models.calibration import Calibrator

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    individual_id: str
    image_id: str
    crop_path: str
    viewpoint: str
    fused_sim: float
    global_sims: dict        # {"megadescriptor": 0.82, "miewid": 0.79}
    local_inliers: int
    local_sim: float         # calibrated
    viz_payload: dict        # from LocalMatcher.score


class WildFusionMatcher:
    """
    Hybrid re-ID matcher: global FAISS shortlist → local re-rank → calibrated fusion.
    """

    def __init__(
        self,
        embedders: dict,          # {desc_name: GlobalEmbedder}
        faiss_indexes: dict,      # {desc_name: faiss.Index}
        ref_meta: dict,           # {desc_name: list[(individual_id, image_id, crop_path, viewpoint)]}
        local_matcher: LocalMatcher,
        calibrators: dict,        # {matcher_name: Calibrator}; may be empty pre-Phase-4
        weights: dict = None,
        shortlist_k: int = SHORTLIST_K,
    ):
        self.embedders     = embedders
        self.faiss_indexes = faiss_indexes
        self.ref_meta      = ref_meta
        self.local_matcher = local_matcher
        self.calibrators   = calibrators if calibrators else {}
        self.weights       = weights if weights is not None else dict(FUSION_WEIGHTS)
        self.shortlist_k   = shortlist_k

        if not self.calibrators:
            logger.warning(
                "No calibrators loaded. WildFusionMatcher will use raw global sims only "
                "(local_sim=0). Train calibrators in Phase 4 to enable full fusion."
            )

    # ------------------------------------------------------------------
    # Step 1 – Global shortlist via FAISS
    # ------------------------------------------------------------------

    def shortlist(self, query_embedding_per_desc: dict) -> list:
        """
        Returns list of dicts:
          {individual_id, image_id, crop_path, viewpoint, global_sims, faiss_score}
        Deduplication is by image_id; highest per-descriptor score is kept.
        """
        # {image_id: candidate_dict}
        candidates: dict[str, dict] = {}

        for desc in ACTIVE_DESCRIPTORS:
            if desc not in self.faiss_indexes or desc not in query_embedding_per_desc:
                continue

            q_vec = query_embedding_per_desc[desc].astype(np.float32).reshape(1, -1)
            distances, indices = self.faiss_indexes[desc].search(q_vec, self.shortlist_k)

            meta_list = self.ref_meta[desc]
            for rank, (faiss_row, sim) in enumerate(zip(indices[0], distances[0])):
                if faiss_row < 0 or faiss_row >= len(meta_list):
                    continue
                ind_id, img_id, crop_path, viewpoint = meta_list[faiss_row]
                if img_id not in candidates:
                    candidates[img_id] = {
                        "individual_id": ind_id,
                        "image_id":      img_id,
                        "crop_path":     crop_path,
                        "viewpoint":     viewpoint,
                        "global_sims":   {},
                        "faiss_score":   float(sim),
                    }
                cand = candidates[img_id]
                # keep best score for this descriptor
                prev = cand["global_sims"].get(desc, -1.0)
                if float(sim) > prev:
                    cand["global_sims"][desc] = float(sim)
                # update aggregated faiss_score
                if float(sim) > cand["faiss_score"]:
                    cand["faiss_score"] = float(sim)

        return list(candidates.values())

    # ------------------------------------------------------------------
    # Step 2 – Local re-rank
    # ------------------------------------------------------------------

    def rerank(self, query_crop_bgr: np.ndarray, candidates: list) -> list:
        """
        Adds 'local_inliers' and 'viz_payload' keys to each candidate dict.
        Reads crop images from disk; skips gracefully if a file is missing.
        """
        for cand in candidates:
            crop_path = cand.get("crop_path", "")
            if not crop_path or not os.path.isfile(crop_path):
                cand["local_inliers"] = 0
                cand["viz_payload"]   = {}
                continue

            ref_bgr = cv2.imread(crop_path)
            if ref_bgr is None:
                cand["local_inliers"] = 0
                cand["viz_payload"]   = {}
                continue

            n_inliers, viz = self.local_matcher.score(query_crop_bgr, ref_bgr)
            cand["local_inliers"] = n_inliers
            cand["viz_payload"]   = viz

        return candidates

    # ------------------------------------------------------------------
    # Step 3 – Score fusion
    # ------------------------------------------------------------------

    def fuse(self, global_sims: dict, local_inliers: int) -> float:
        """
        Calibrate each score and compute weighted sum.
        Falls back to raw global mean when calibrators are absent.
        """
        if not self.calibrators:
            # Pre-Phase-4 baseline: mean of available global sims
            vals = list(global_sims.values())
            return float(np.mean(vals)) if vals else 0.0

        total_weight = 0.0
        fused = 0.0

        for desc, raw_sim in global_sims.items():
            w = self.weights.get(desc, 0.0)
            if w == 0.0:
                continue
            if desc in self.calibrators:
                cal_sim = float(self.calibrators[desc].transform(np.array([raw_sim]))[0])
            else:
                cal_sim = float(raw_sim)
            fused        += w * cal_sim
            total_weight += w

        # Local matcher contribution
        w_local = self.weights.get("local", 0.0)
        if w_local > 0.0:
            # Normalise inlier count to [0, 1] using a soft sigmoid-like mapping.
            # 50 inliers → ~0.77; this keeps the score bounded without a hard cap.
            local_raw = float(local_inliers) / (float(local_inliers) + 20.0)
            if "local" in self.calibrators:
                local_cal = float(self.calibrators["local"].transform(np.array([local_raw]))[0])
            else:
                local_cal = local_raw
            fused        += w_local * local_cal
            total_weight += w_local

        if total_weight > 0:
            fused /= total_weight

        return float(np.clip(fused, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def identify(
        self,
        query_embedding_per_desc: dict,
        query_crop_bgr: np.ndarray,
    ) -> list:
        """
        Returns top-NUM_RECOMMENDED_IDS Recommendations sorted by fused_sim descending.
        """
        candidates = self.shortlist(query_embedding_per_desc)

        if not candidates:
            return []

        candidates = self.rerank(query_crop_bgr, candidates)

        recommendations = []
        for cand in candidates:
            local_inliers = cand.get("local_inliers", 0)
            global_sims   = cand.get("global_sims", {})
            fused_sim     = self.fuse(global_sims, local_inliers)

            # local_sim: calibrated inlier-count score (same path as fuse uses)
            if self.calibrators and "local" in self.calibrators:
                local_raw = float(local_inliers) / (float(local_inliers) + 20.0)
                local_sim = float(self.calibrators["local"].transform(np.array([local_raw]))[0])
            else:
                local_sim = float(local_inliers) / (float(local_inliers) + 20.0) if local_inliers > 0 else 0.0

            recommendations.append(
                Recommendation(
                    individual_id = cand["individual_id"],
                    image_id      = cand["image_id"],
                    crop_path     = cand["crop_path"],
                    viewpoint     = cand.get("viewpoint", "unknown"),
                    fused_sim     = fused_sim,
                    global_sims   = global_sims,
                    local_inliers = local_inliers,
                    local_sim     = local_sim,
                    viz_payload   = cand.get("viz_payload", {}),
                )
            )

        recommendations.sort(key=lambda r: r.fused_sim, reverse=True)
        return recommendations[:NUM_RECOMMENDED_IDS]
