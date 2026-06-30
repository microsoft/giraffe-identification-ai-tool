# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

import os
import sys
import logging
import numpy as np
import cv2
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_elephant import LOCAL_MATCHER_BACKEND, LOCAL_MATCHER_KEYPOINTS, LOCAL_MATCHER_MIN_INLIERS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LightGlue import strategy: prefer the standalone lightglue package,
# fall back to kornia.feature which ships its own implementation.
# ---------------------------------------------------------------------------
try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    _LIGHTGLUE_NATIVE = True
except ImportError:
    _LIGHTGLUE_NATIVE = False
    logger.info("lightglue package not found; will use kornia.feature.LightGlue.")


def _grayscale(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def _to_tensor_gray(image_gray: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W) uint8 → (1, 1, H, W) float32 in [0, 1]."""
    t = torch.from_numpy(image_gray).float() / 255.0
    return t.unsqueeze(0).unsqueeze(0).to(device)


class LocalMatcher:
    """
    Compute local feature match count between two crops.
    Supports 'lightglue' and 'loftr' backends.
    """

    def __init__(
        self,
        backend: str = LOCAL_MATCHER_BACKEND,
        max_keypoints: int = LOCAL_MATCHER_KEYPOINTS,
        min_inliers: int = LOCAL_MATCHER_MIN_INLIERS,
    ):
        self.backend = backend
        self.max_keypoints = max_keypoints
        self.min_inliers = min_inliers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if backend == "lightglue":
            self._init_lightglue()
        elif backend == "loftr":
            self._init_loftr()
        else:
            raise ValueError(f"Unknown local matcher backend '{backend}'.")

        logger.info("LocalMatcher('%s') ready on %s.", backend, self.device)

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_lightglue(self):
        if _LIGHTGLUE_NATIVE:
            self._extractor = SuperPoint(max_num_keypoints=self.max_keypoints).eval().to(self.device)
            self._matcher   = LightGlue(features="superpoint").eval().to(self.device)
            self._match_fn  = self._match_lightglue_native
        else:
            import kornia.feature as KF
            self._extractor = KF.KeyNetAffNetHardNet(
                num_features=self.max_keypoints, upright=True
            ).eval().to(self.device)
            self._matcher = KF.LightGlue("keynet").eval().to(self.device)
            self._match_fn = self._match_lightglue_kornia

    def _init_loftr(self):
        import kornia.feature as KF
        self._matcher  = KF.LoFTR(pretrained="outdoor").eval().to(self.device)
        self._match_fn = self._match_loftr

    # ------------------------------------------------------------------
    # Matching implementations
    # ------------------------------------------------------------------

    def _match_lightglue_native(self, query_bgr: np.ndarray, ref_bgr: np.ndarray):
        qg = _grayscale(query_bgr)
        rg = _grayscale(ref_bgr)
        qt = _to_tensor_gray(qg, self.device)
        rt = _to_tensor_gray(rg, self.device)

        with torch.no_grad():
            q_feats = self._extractor.extract(qt)
            r_feats = self._extractor.extract(rt)
            result  = self._matcher({"image0": q_feats, "image1": r_feats})

        # rbd strips the batch dimension
        q_feats, r_feats, result = [rbd(x) for x in (q_feats, r_feats, result)]

        matches   = result["matches"]           # (M, 2)
        q_kpts    = q_feats["keypoints"]        # (N, 2)
        r_kpts    = r_feats["keypoints"]        # (K, 2)

        if matches.shape[0] < 4:
            return 0, {}, q_kpts.cpu().numpy(), r_kpts.cpu().numpy(), matches.cpu().numpy()

        q_matched = q_kpts[matches[:, 0]].cpu().numpy()
        r_matched = r_kpts[matches[:, 1]].cpu().numpy()
        return (
            len(matches),
            {},
            q_kpts.cpu().numpy(),
            r_kpts.cpu().numpy(),
            matches.cpu().numpy(),
            q_matched,
            r_matched,
        )

    def _match_lightglue_kornia(self, query_bgr: np.ndarray, ref_bgr: np.ndarray):
        qg = _grayscale(query_bgr)
        rg = _grayscale(ref_bgr)
        qt = _to_tensor_gray(qg, self.device)
        rt = _to_tensor_gray(rg, self.device)

        with torch.no_grad():
            q_kps, q_descs, q_lafs = self._extractor(qt)
            r_kps, r_descs, r_lafs = self._extractor(rt)
            dists, match_ids = self._matcher(
                q_descs, r_descs, q_lafs, r_lafs
            )

        q_matched = q_kps[0, match_ids[0, :, 0]].cpu().numpy()
        r_matched = r_kps[0, match_ids[0, :, 1]].cpu().numpy()
        matches_np = match_ids[0].cpu().numpy()
        return (
            len(matches_np),
            {},
            q_kps[0].cpu().numpy(),
            r_kps[0].cpu().numpy(),
            matches_np,
            q_matched,
            r_matched,
        )

    def _match_loftr(self, query_bgr: np.ndarray, ref_bgr: np.ndarray):
        qg = _grayscale(query_bgr)
        rg = _grayscale(ref_bgr)
        qt = _to_tensor_gray(qg, self.device)
        rt = _to_tensor_gray(rg, self.device)

        with torch.no_grad():
            correspondences = self._matcher({"image0": qt, "image1": rt})

        q_matched = correspondences["keypoints0"][0].cpu().numpy()
        r_matched = correspondences["keypoints1"][0].cpu().numpy()
        conf      = correspondences["confidence"][0].cpu().numpy()

        matches_np = np.stack(
            [np.arange(len(q_matched)), np.arange(len(r_matched))], axis=1
        )
        return (
            len(q_matched),
            {"confidence": conf},
            q_matched,
            r_matched,
            matches_np,
            q_matched,
            r_matched,
        )

    # ------------------------------------------------------------------
    # RANSAC geometric verification
    # ------------------------------------------------------------------

    def _ransac_inliers(self, q_matched: np.ndarray, r_matched: np.ndarray) -> int:
        """Returns the number of RANSAC inliers (homography)."""
        if len(q_matched) < 4:
            return 0
        _, mask = cv2.findHomography(
            q_matched.astype(np.float32),
            r_matched.astype(np.float32),
            cv2.RANSAC,
            ransacReprojThreshold=8.0,
        )
        if mask is None:
            return 0
        return int(mask.sum())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, query_bgr: np.ndarray, ref_bgr: np.ndarray) -> tuple:
        """
        Returns (n_inliers, viz_payload).
        viz_payload keys: query_kpts, ref_kpts, matches.
        Returns (0, {}) if matching fails entirely.
        """
        try:
            result = self._match_fn(query_bgr, ref_bgr)
        except Exception as exc:
            logger.warning("Local matching failed: %s", exc)
            return 0, {}

        if len(result) == 5:
            # loftr / kornia path where matched pts are the full point sets
            raw_count, extra, q_kpts, r_kpts, matches_np = result[:5]
            q_matched = q_kpts
            r_matched = r_kpts
        else:
            raw_count, extra, q_kpts, r_kpts, matches_np, q_matched, r_matched = result

        if raw_count < 4:
            return 0, {}

        n_inliers = self._ransac_inliers(q_matched, r_matched)

        viz_payload = {
            "query_kpts": q_kpts,
            "ref_kpts":   r_kpts,
            "matches":    matches_np,
        }
        viz_payload.update(extra)
        return n_inliers, viz_payload

    def score_against(self, query_bgr: np.ndarray, ref_bgrs: list) -> list:
        """Returns list of (n_inliers, viz_payload) for each ref image."""
        return [self.score(query_bgr, ref_bgr) for ref_bgr in ref_bgrs]
