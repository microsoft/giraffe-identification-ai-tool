# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_elephant import (
    LOCAL_MATCHER_BACKEND,
    LOCAL_MATCHER_KEYPOINTS,
    LOCAL_MATCHER_MIN_INLIERS,
    LOCAL_MATCHER_LOFTR_PILOT_APPROVED,
)

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

# ---------------------------------------------------------------------------
# Region constants
# ---------------------------------------------------------------------------
REGION_EAR = "ear"
REGION_BODY = "body"
REGION_HEAD = "head"  # retained for contract compatibility; disabled in quality gate

# Regions supported by the strict path (head disabled by quality gate).
STRICT_SUPPORTED_REGIONS: frozenset[str] = frozenset({REGION_EAR, REGION_BODY})

# Geometric verification model choices
GEOM_HOMOGRAPHY = "homography"
GEOM_PARTIAL_AFFINE = "partial_affine"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LocalMatcherError(RuntimeError):
    """Raised by StrictLocalMatcher on any unrecoverable condition."""


class LocalMatcherModelError(LocalMatcherError):
    """Raised when the underlying model cannot be initialised or loaded."""


class LocalMatcherInferenceError(LocalMatcherError):
    """Raised when feature extraction or matching fails during inference."""


class LocalMatcherFileError(LocalMatcherError):
    """Raised when a required input file is missing or unreadable."""


class LocalMatcherRegionError(LocalMatcherError):
    """Raised when region constraints are violated."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FeatureBundle:
    """
    Compact container for extracted SuperPoint features.
    orientation: 'original' | 'flipped'
    """
    keypoints: np.ndarray         # (N, 2) float32
    descriptors: np.ndarray       # (N, D) float32
    scores: np.ndarray            # (N,)   float32
    image_shape: tuple[int, int]  # (H, W)
    orientation: str              # 'original' | 'flipped'
    backend: str
    model_fingerprint: str

    def is_valid(self) -> bool:
        return (
            self.keypoints.ndim == 2
            and self.keypoints.shape[1] == 2
            and len(self.keypoints) > 0
        )


@dataclass
class GeomVerification:
    """Result of geometric verification (RANSAC)."""
    model_used: str               # GEOM_HOMOGRAPHY | GEOM_PARTIAL_AFFINE
    n_raw_matches: int
    n_inliers: int
    inlier_ratio: float
    geometric_spread: float       # stddev of inlier spatial positions, pixels
    H: Optional[np.ndarray]       # transform matrix (may be None)
    inlier_mask: Optional[np.ndarray]  # boolean mask over raw matches


@dataclass
class MatchResult:
    """
    Full result of a strict pair match including geometry.

    orientation: which reference orientation was selected ('original' | 'flipped')
    """
    region: str
    orientation: str
    raw_matches: np.ndarray       # (M, 2) int indices into query/ref keypoints
    query_keypoints: np.ndarray   # (N, 2)
    ref_keypoints: np.ndarray     # (K, 2)
    geom: GeomVerification
    backend: str
    model_fingerprint: str
    # Score exposed to callers — inlier count after geometric verification
    n_inliers: int
    # Convenience
    viz_payload: dict = field(default_factory=dict)

    @property
    def inlier_ratio(self) -> float:
        return self.geom.inlier_ratio

    @property
    def geometric_spread(self) -> float:
        return self.geom.geometric_spread


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _grayscale(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def _to_tensor_gray(image_gray: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W) uint8 → (1, 1, H, W) float32 in [0, 1]."""
    t = torch.from_numpy(image_gray).float() / 255.0
    return t.unsqueeze(0).unsqueeze(0).to(device)


def _flip_horizontal(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.flip(image_bgr, 1)


def _geometric_spread(q_pts: np.ndarray) -> float:
    """Spatial spread of inlier points as mean standard deviation across axes."""
    if len(q_pts) < 2:
        return 0.0
    return float(np.sqrt(np.var(q_pts[:, 0]) + np.var(q_pts[:, 1])))


def _ransac_homography(
    q_pts: np.ndarray,
    r_pts: np.ndarray,
    reproj_threshold: float = 8.0,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if len(q_pts) < 4:
        return None, None
    H, mask = cv2.findHomography(
        q_pts.astype(np.float32),
        r_pts.astype(np.float32),
        cv2.RANSAC,
        ransacReprojThreshold=reproj_threshold,
    )
    return H, mask


def _ransac_partial_affine(
    q_pts: np.ndarray,
    r_pts: np.ndarray,
    reproj_threshold: float = 8.0,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if len(q_pts) < 3:
        return None, None
    M, mask = cv2.estimateAffinePartial2D(
        q_pts.astype(np.float32),
        r_pts.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj_threshold,
    )
    return M, mask


def _run_geom_verification(
    q_matched: np.ndarray,
    r_matched: np.ndarray,
    raw_matches: np.ndarray,
    region: str,
    geom_model: Optional[str] = None,
) -> GeomVerification:
    """
    Run RANSAC geometric verification for the given region.

    Region policy:
      ear  → homography (planar surface assumption)
      body → partial_affine (flexible deformation) with homography fallback
      head → homography (retained for contract compatibility, disabled at caller)
    """
    n_raw = len(q_matched)

    if geom_model is None:
        if region == REGION_EAR or region == REGION_HEAD:
            geom_model = GEOM_HOMOGRAPHY
        else:
            geom_model = GEOM_PARTIAL_AFFINE

    H = None
    mask = None

    if geom_model == GEOM_HOMOGRAPHY:
        H, mask = _ransac_homography(q_matched, r_matched)
    elif geom_model == GEOM_PARTIAL_AFFINE:
        H, mask = _ransac_partial_affine(q_matched, r_matched)
        if mask is None and n_raw >= 4:
            # fallback to homography for body when partial-affine degenerates
            H, mask = _ransac_homography(q_matched, r_matched)
            geom_model = GEOM_HOMOGRAPHY
    else:
        raise LocalMatcherRegionError(f"Unknown geom_model: {geom_model!r}")

    if mask is None:
        return GeomVerification(
            model_used=geom_model,
            n_raw_matches=n_raw,
            n_inliers=0,
            inlier_ratio=0.0,
            geometric_spread=0.0,
            H=None,
            inlier_mask=None,
        )

    mask_bool = mask.ravel().astype(bool)
    n_inliers = int(mask_bool.sum())
    inlier_ratio = n_inliers / n_raw if n_raw > 0 else 0.0
    inlier_q = q_matched[mask_bool]
    spread = _geometric_spread(inlier_q)

    return GeomVerification(
        model_used=geom_model,
        n_raw_matches=n_raw,
        n_inliers=n_inliers,
        inlier_ratio=inlier_ratio,
        geometric_spread=spread,
        H=H,
        inlier_mask=mask_bool,
    )


def _model_fingerprint(
    backend: str,
    max_keypoints: int,
    preprocess_tag: str = "",
) -> str:
    """Deterministic fingerprint for the model configuration."""
    raw = f"{backend}:{max_keypoints}:{_LIGHTGLUE_NATIVE}:{preprocess_tag}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Legacy public API (unchanged behaviour, callers must not be broken)
# ---------------------------------------------------------------------------

class LocalMatcher:
    """
    Compute local feature match count between two crops.
    Supports 'lightglue' and 'loftr' backends.

    This is the legacy API.  New code should use StrictLocalMatcher.
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
        # cuDNN drops CC 7.0 support in newer builds; fall back to generic CUDA ops.
        if self.device.type == "cuda":
            cc = torch.cuda.get_device_capability(self.device)
            if cc < (7, 5):
                torch.backends.cudnn.enabled = False
                logger.info("Disabled cuDNN (CC %d.%d < 7.5); using generic CUDA ops.", *cc)

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


# ---------------------------------------------------------------------------
# Strict experimental API
# ---------------------------------------------------------------------------

class StrictLocalMatcher:
    """
    Experimental strict local matcher for body↔body and ear↔ear regions.

    Key differences from LocalMatcher:
    - Explicit ``device`` and optional ``disable_cudnn`` at construction.
    - Fails loudly (raises LocalMatcherError subclasses) rather than returning
      zeros on model, init, inference, or file errors.
    - Feature extraction (SuperPoint) and matching (LightGlue) are separated so
      features can be cached externally.
    - Mirror search: both original and horizontally-flipped reference orientations
      are tried; the orientation yielding more inliers is returned.
    - Region-specific geometry: homography for ear; partial-affine (with homography
      fallback) for body; head geometry retained for contract compatibility but the
      head region is disabled at the caller level by the quality gate.
    - LoFTR backend requires explicit pilot approval (LOCAL_MATCHER_LOFTR_PILOT_APPROVED).
    - All result fields (raw matches, inliers, inlier ratio, geometric spread,
      model used) are stored on the returned MatchResult dataclass.

    Supported regions: ear, body.  Head is structurally supported but disabled.
    """

    SUPPORTED_BACKENDS: frozenset[str] = frozenset({"lightglue", "loftr"})

    def __init__(
        self,
        backend: str = LOCAL_MATCHER_BACKEND,
        max_keypoints: int = LOCAL_MATCHER_KEYPOINTS,
        min_inliers: int = LOCAL_MATCHER_MIN_INLIERS,
        *,
        device: Optional[torch.device] = None,
        disable_cudnn: bool = False,
        allow_loftr_pilot: bool = False,
        loftr_max_edge: int = 640,
    ):
        if backend not in self.SUPPORTED_BACKENDS:
            raise LocalMatcherModelError(
                f"Unknown backend {backend!r}; supported: {sorted(self.SUPPORTED_BACKENDS)}"
            )
        if (
            backend == "loftr"
            and not LOCAL_MATCHER_LOFTR_PILOT_APPROVED
            and not allow_loftr_pilot
        ):
            raise LocalMatcherModelError(
                "LoFTR backend requires LOCAL_MATCHER_LOFTR_PILOT_APPROVED=True in config."
            )

        self.backend = backend
        self.max_keypoints = max_keypoints
        self.min_inliers = min_inliers
        self.disable_cudnn = disable_cudnn
        self.loftr_max_edge = loftr_max_edge

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self.device = device

        if disable_cudnn and device.type == "cuda":
            torch.backends.cudnn.enabled = False
            if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
                torch.backends.cuda.enable_cudnn_sdp(False)
            logger.info("StrictLocalMatcher: cuDNN explicitly disabled.")

        preprocess_tag = (
            f"max_edge={loftr_max_edge}" if backend == "loftr" else ""
        )
        self.model_fingerprint = _model_fingerprint(
            backend, max_keypoints, preprocess_tag
        )

        try:
            if backend == "lightglue":
                self._init_lightglue_strict()
            elif backend == "loftr":
                self._init_loftr_strict()
        except LocalMatcherModelError:
            raise
        except Exception as exc:
            raise LocalMatcherModelError(
                f"Failed to initialise backend {backend!r}: {exc}"
            ) from exc

        logger.info(
            "StrictLocalMatcher('%s') ready on %s (fingerprint=%s).",
            backend, self.device, self.model_fingerprint,
        )

    # ------------------------------------------------------------------
    # Backend initialisation (strict)
    # ------------------------------------------------------------------

    def _init_lightglue_strict(self) -> None:
        if self.device.type == "cuda" and not self.disable_cudnn:
            cc = torch.cuda.get_device_capability(self.device)
            if cc < (7, 5):
                torch.backends.cudnn.enabled = False
                logger.info(
                    "StrictLocalMatcher: Disabled cuDNN (CC %d.%d < 7.5).", *cc
                )

        if _LIGHTGLUE_NATIVE:
            self._extractor = SuperPoint(
                max_num_keypoints=self.max_keypoints
            ).eval().to(self.device)
            self._matcher = LightGlue(features="superpoint").eval().to(self.device)
        else:
            import kornia.feature as KF
            self._extractor = KF.KeyNetAffNetHardNet(
                num_features=self.max_keypoints, upright=True
            ).eval().to(self.device)
            self._matcher = KF.LightGlue("keynet").eval().to(self.device)

    def _init_loftr_strict(self) -> None:
        import kornia.feature as KF
        self._loftr = KF.LoFTR(pretrained="outdoor").eval().to(self.device)

    # ------------------------------------------------------------------
    # Feature extraction (separated from matching for caching)
    # ------------------------------------------------------------------

    def extract_features(
        self,
        image_bgr: np.ndarray,
        orientation: str = "original",
    ) -> FeatureBundle:
        """
        Extract SuperPoint features from a BGR image.

        Parameters
        ----------
        image_bgr:
            Input image as (H, W, 3) uint8 BGR numpy array.
        orientation:
            'original' or 'flipped' (horizontal flip).

        Returns
        -------
        FeatureBundle with keypoints, descriptors, scores, and metadata.

        Raises
        ------
        LocalMatcherModelError  if called on a LoFTR matcher (LoFTR is pairwise).
        LocalMatcherInferenceError  on any inference failure.
        """
        if self.backend == "loftr":
            raise LocalMatcherModelError(
                "LoFTR is pairwise; use score_pair_strict() directly."
            )

        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise LocalMatcherInferenceError(
                f"Expected (H, W, 3) BGR image, got shape {image_bgr.shape}"
            )

        img = image_bgr
        if orientation == "flipped":
            img = _flip_horizontal(img)
        elif orientation != "original":
            raise LocalMatcherInferenceError(
                f"Unknown orientation {orientation!r}; expected 'original' or 'flipped'"
            )

        gray = _grayscale(img)
        tensor = _to_tensor_gray(gray, self.device)

        try:
            with torch.no_grad():
                if _LIGHTGLUE_NATIVE:
                    feats = self._extractor.extract(tensor)
                    feats = rbd(feats)
                    kpts = feats["keypoints"].cpu().numpy()
                    descs = feats["descriptors"].cpu().numpy()
                    scores = feats.get("keypoint_scores", feats.get("scores", torch.zeros(len(kpts)))).cpu().numpy()
                else:
                    kps, descs_t, lafs = self._extractor(tensor)
                    kpts = kps[0].cpu().numpy()
                    descs = descs_t[0].cpu().numpy()
                    scores = np.ones(len(kpts), dtype=np.float32)
        except Exception as exc:
            raise LocalMatcherInferenceError(
                f"Feature extraction failed: {exc}"
            ) from exc

        return FeatureBundle(
            keypoints=kpts.astype(np.float32),
            descriptors=descs.astype(np.float32),
            scores=scores.astype(np.float32),
            image_shape=(gray.shape[0], gray.shape[1]),
            orientation=orientation,
            backend=self.backend,
            model_fingerprint=self.model_fingerprint,
        )

    # ------------------------------------------------------------------
    # Matching from feature bundles
    # ------------------------------------------------------------------

    def match_features(
        self,
        query_bundle: FeatureBundle,
        ref_bundle: FeatureBundle,
        region: str,
        geom_model: Optional[str] = None,
    ) -> MatchResult:
        """
        Run LightGlue matching between two pre-extracted FeatureBundles and
        apply region-appropriate geometric verification.

        Raises
        ------
        LocalMatcherRegionError  if region is not in STRICT_SUPPORTED_REGIONS.
        LocalMatcherInferenceError  on any matching failure.
        """
        if region not in STRICT_SUPPORTED_REGIONS:
            raise LocalMatcherRegionError(
                f"Region {region!r} is not supported by StrictLocalMatcher. "
                f"Supported: {sorted(STRICT_SUPPORTED_REGIONS)}"
            )
        if self.backend == "loftr":
            raise LocalMatcherModelError(
                "LoFTR requires pairwise images; use score_pair_strict()."
            )
        if query_bundle.model_fingerprint != self.model_fingerprint:
            raise LocalMatcherInferenceError(
                f"Query bundle fingerprint {query_bundle.model_fingerprint!r} "
                f"does not match matcher {self.model_fingerprint!r}"
            )
        if ref_bundle.model_fingerprint != self.model_fingerprint:
            raise LocalMatcherInferenceError(
                f"Ref bundle fingerprint {ref_bundle.model_fingerprint!r} "
                f"does not match matcher {self.model_fingerprint!r}"
            )

        try:
            raw_matches, q_kpts_all, r_kpts_all = self._run_lightglue_match(
                query_bundle, ref_bundle
            )
        except LocalMatcherInferenceError:
            raise
        except Exception as exc:
            raise LocalMatcherInferenceError(
                f"LightGlue matching failed: {exc}"
            ) from exc

        if len(raw_matches) < 4:
            geom = GeomVerification(
                model_used=geom_model or (
                    GEOM_HOMOGRAPHY if region in (REGION_EAR, REGION_HEAD)
                    else GEOM_PARTIAL_AFFINE
                ),
                n_raw_matches=len(raw_matches),
                n_inliers=0,
                inlier_ratio=0.0,
                geometric_spread=0.0,
                H=None,
                inlier_mask=None,
            )
            return MatchResult(
                region=region,
                orientation=ref_bundle.orientation,
                raw_matches=raw_matches,
                query_keypoints=q_kpts_all,
                ref_keypoints=r_kpts_all,
                geom=geom,
                backend=self.backend,
                model_fingerprint=self.model_fingerprint,
                n_inliers=0,
            )

        if len(raw_matches) >= 2 and raw_matches.ndim == 2 and raw_matches.shape[1] == 2:
            q_matched = q_kpts_all[raw_matches[:, 0]]
            r_matched = r_kpts_all[raw_matches[:, 1]]
        else:
            q_matched = q_kpts_all[: len(raw_matches)]
            r_matched = r_kpts_all[: len(raw_matches)]

        geom = _run_geom_verification(q_matched, r_matched, raw_matches, region, geom_model)

        viz = {
            "query_kpts": q_kpts_all,
            "ref_kpts": r_kpts_all,
            "matches": raw_matches,
        }

        return MatchResult(
            region=region,
            orientation=ref_bundle.orientation,
            raw_matches=raw_matches,
            query_keypoints=q_kpts_all,
            ref_keypoints=r_kpts_all,
            geom=geom,
            backend=self.backend,
            model_fingerprint=self.model_fingerprint,
            n_inliers=geom.n_inliers,
            viz_payload=viz,
        )

    def _run_lightglue_match(
        self,
        query_bundle: FeatureBundle,
        ref_bundle: FeatureBundle,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (raw_matches (M,2), q_kpts (N,2), r_kpts (K,2))."""
        if _LIGHTGLUE_NATIVE:
            def _bundle_to_dict(b: FeatureBundle) -> dict:
                kpts = torch.from_numpy(b.keypoints).unsqueeze(0).to(self.device)
                descs = torch.from_numpy(b.descriptors).unsqueeze(0).to(self.device)
                image_size = torch.tensor(
                    [[b.image_shape[1], b.image_shape[0]]],
                    device=self.device,
                )
                return {
                    "keypoints": kpts,
                    "descriptors": descs,
                    "image_size": image_size,
                }

            q_dict = _bundle_to_dict(query_bundle)
            r_dict = _bundle_to_dict(ref_bundle)

            with torch.no_grad():
                result = self._matcher({"image0": q_dict, "image1": r_dict})
            result = rbd(result)
            matches = result["matches"].cpu().numpy()
        else:
            import kornia.feature as KF
            q_descs = torch.from_numpy(query_bundle.descriptors).unsqueeze(0).to(self.device)
            r_descs = torch.from_numpy(ref_bundle.descriptors).unsqueeze(0).to(self.device)
            q_kps_t = torch.from_numpy(query_bundle.keypoints).unsqueeze(0).to(self.device)
            r_kps_t = torch.from_numpy(ref_bundle.keypoints).unsqueeze(0).to(self.device)
            q_lafs = KF.laf_from_center_scale_ori(q_kps_t.unsqueeze(2), torch.ones(1, len(query_bundle.keypoints), 1, 1, device=self.device))
            r_lafs = KF.laf_from_center_scale_ori(r_kps_t.unsqueeze(2), torch.ones(1, len(ref_bundle.keypoints), 1, 1, device=self.device))
            with torch.no_grad():
                _, match_ids = self._matcher(q_descs, r_descs, q_lafs, r_lafs)
            matches = match_ids[0].cpu().numpy()

        return (
            matches,
            query_bundle.keypoints,
            ref_bundle.keypoints,
        )

    # ------------------------------------------------------------------
    # LoFTR pairwise scoring
    # ------------------------------------------------------------------

    def _score_loftr_pair(
        self,
        query_bgr: np.ndarray,
        ref_bgr: np.ndarray,
        region: str,
        geom_model: Optional[str] = None,
    ) -> MatchResult:
        """Direct LoFTR pairwise scoring; also applies geometric verification."""
        try:
            def _resize(image: np.ndarray) -> np.ndarray:
                height, width = image.shape[:2]
                scale = min(1.0, self.loftr_max_edge / max(height, width))
                new_height = max(8, int(height * scale) // 8 * 8)
                new_width = max(8, int(width * scale) // 8 * 8)
                if (new_height, new_width) == (height, width):
                    return image
                return cv2.resize(
                    image,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA,
                )

            qg = _grayscale(_resize(query_bgr))
            rg = _grayscale(_resize(ref_bgr))
            qt = _to_tensor_gray(qg, self.device)
            rt = _to_tensor_gray(rg, self.device)
            with torch.no_grad():
                corr = self._loftr({"image0": qt, "image1": rt})
            q_points = corr["keypoints0"]
            r_points = corr["keypoints1"]
            if q_points.ndim == 3:
                q_points = q_points[0]
                r_points = r_points[0]
            q_matched = q_points.cpu().numpy()
            r_matched = r_points.cpu().numpy()
        except Exception as exc:
            raise LocalMatcherInferenceError(f"LoFTR inference failed: {exc}") from exc

        raw_matches = np.stack(
            [np.arange(len(q_matched)), np.arange(len(r_matched))], axis=1
        )
        geom = _run_geom_verification(q_matched, r_matched, raw_matches, region, geom_model)

        return MatchResult(
            region=region,
            orientation="original",
            raw_matches=raw_matches,
            query_keypoints=q_matched,
            ref_keypoints=r_matched,
            geom=geom,
            backend=self.backend,
            model_fingerprint=self.model_fingerprint,
            n_inliers=geom.n_inliers,
            viz_payload={"query_kpts": q_matched, "ref_kpts": r_matched, "matches": raw_matches},
        )

    # ------------------------------------------------------------------
    # Mirror search
    # ------------------------------------------------------------------

    def _mirror_search(
        self,
        query_bundle: FeatureBundle,
        ref_bgr: np.ndarray,
        region: str,
        geom_model: Optional[str] = None,
    ) -> MatchResult:
        """
        Score against original and horizontally-flipped reference; return best.

        'Best' is determined by n_inliers after geometric verification.
        """
        ref_bundle_orig = self.extract_features(ref_bgr, orientation="original")
        ref_bundle_flip = self.extract_features(ref_bgr, orientation="flipped")

        result_orig = self.match_features(query_bundle, ref_bundle_orig, region, geom_model)
        result_flip = self.match_features(query_bundle, ref_bundle_flip, region, geom_model)

        if result_flip.n_inliers > result_orig.n_inliers:
            return result_flip
        return result_orig

    def _mirror_search_loftr(
        self,
        query_bgr: np.ndarray,
        ref_bgr: np.ndarray,
        region: str,
        geom_model: Optional[str] = None,
    ) -> MatchResult:
        """LoFTR variant of mirror search."""
        result_orig = self._score_loftr_pair(query_bgr, ref_bgr, region, geom_model)
        ref_flip = _flip_horizontal(ref_bgr)
        result_flip = self._score_loftr_pair(query_bgr, ref_flip, region, geom_model)
        if result_flip.n_inliers > result_orig.n_inliers:
            r = result_flip
            r.orientation = "flipped"
            return r
        return result_orig

    # ------------------------------------------------------------------
    # High-level strict pair scoring
    # ------------------------------------------------------------------

    def score_pair_strict(
        self,
        query_bgr: np.ndarray,
        ref_bgr: np.ndarray,
        region: str,
        *,
        geom_model: Optional[str] = None,
        mirror_search: bool = True,
    ) -> MatchResult:
        """
        Score a query↔reference pair for the given region.

        Raises LocalMatcherError subclasses on any failure.
        Does NOT catch exceptions and return zeros — all errors propagate.

        Parameters
        ----------
        query_bgr, ref_bgr:
            (H, W, 3) uint8 BGR images.
        region:
            'ear' or 'body'.  'head' raises LocalMatcherRegionError.
        geom_model:
            Override RANSAC model ('homography' or 'partial_affine').
            If None, region policy is applied.
        mirror_search:
            If True (default), score both original and flipped reference;
            return the orientation yielding more inliers.
        """
        if region not in STRICT_SUPPORTED_REGIONS:
            raise LocalMatcherRegionError(
                f"Region {region!r} not in STRICT_SUPPORTED_REGIONS "
                f"({sorted(STRICT_SUPPORTED_REGIONS)}). "
                "Head was dropped by the quality gate."
            )

        for arr, name in ((query_bgr, "query"), (ref_bgr, "ref")):
            if not isinstance(arr, np.ndarray):
                raise LocalMatcherInferenceError(f"{name} is not a numpy array")
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise LocalMatcherInferenceError(
                    f"{name} must be (H, W, 3) BGR; got {arr.shape}"
                )

        if self.backend == "loftr":
            if mirror_search:
                return self._mirror_search_loftr(query_bgr, ref_bgr, region, geom_model)
            return self._score_loftr_pair(query_bgr, ref_bgr, region, geom_model)

        query_bundle = self.extract_features(query_bgr, orientation="original")

        if mirror_search:
            return self._mirror_search(query_bundle, ref_bgr, region, geom_model)

        ref_bundle = self.extract_features(ref_bgr, orientation="original")
        return self.match_features(query_bundle, ref_bundle, region, geom_model)

    def score_pair_from_paths(
        self,
        query_path: str,
        ref_path: str,
        region: str,
        **kwargs,
    ) -> MatchResult:
        """
        Load images from disk and call score_pair_strict.

        Raises LocalMatcherFileError if either path is missing or unreadable.
        """
        query_bgr = self._load_image(query_path)
        ref_bgr = self._load_image(ref_path)
        return self.score_pair_strict(query_bgr, ref_bgr, region, **kwargs)

    def _load_image(self, path: str) -> np.ndarray:
        if not os.path.isfile(path):
            raise LocalMatcherFileError(f"Image file not found: {path!r}")
        img = cv2.imread(path)
        if img is None:
            raise LocalMatcherFileError(f"cv2.imread returned None for {path!r}")
        return img
