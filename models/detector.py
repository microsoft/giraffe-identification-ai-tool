# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ElephantDetector:
    def __init__(self, backend: str = "megadetector", conf: float = 0.5, device: str = "cuda"):
        if backend not in {"megadetector", "passthrough"}:
            raise ValueError(f"backend must be 'megadetector' or 'passthrough', got '{backend}'")
        self.conf = conf
        self.device = device
        self._model = None

        if backend == "megadetector":
            try:
                from PytorchWildlife.models import detection as pw_detection
                self._model = pw_detection.MegaDetectorV5(device=device, pretrained=True)
                self.backend = "megadetector"
                logger.info("MegaDetector v5 loaded on %s.", device)
            except Exception as exc:
                logger.warning(
                    "PytorchWildlife / MegaDetector unavailable (%s); falling back to passthrough.", exc
                )
                self.backend = "passthrough"
        else:
            self.backend = "passthrough"

    def crop(self, image_bgr: np.ndarray) -> tuple[np.ndarray | None, str]:
        if self.backend == "passthrough" or self._model is None:
            return image_bgr, "unknown"

        try:
            # MegaDetector expects RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = self._model.single_image_detection(image_rgb, conf_thres=self.conf)
            detections = results.get("detections", None)

            if detections is None or len(detections.xyxy) == 0:
                return None, "unknown"

            # Pick the highest-confidence animal detection (category 1 in MD v5)
            best_idx = None
            best_conf = -1.0
            for i, (conf_val, cls_id) in enumerate(
                zip(detections.confidence.tolist(), detections.class_id.tolist())
            ):
                # MD v5 class 1 = animal
                if int(cls_id) == 1 and float(conf_val) >= self.conf and float(conf_val) > best_conf:
                    best_conf = float(conf_val)
                    best_idx = i

            if best_idx is None:
                return None, "unknown"

            x1, y1, x2, y2 = [int(v) for v in detections.xyxy[best_idx]]
            viewpoint = self._infer_viewpoint((x1, y1, x2, y2), image_bgr.shape)
            crop = self._square_pad_crop(image_bgr, x1, y1, x2, y2)
            return crop, viewpoint

        except Exception as exc:
            logger.warning("MegaDetector inference failed: %s — returning None.", exc)
            return None, "unknown"

    def _infer_viewpoint(self, bbox: tuple[int, int, int, int], image_shape: tuple) -> str:
        x1, y1, x2, y2 = bbox
        img_h, img_w = image_shape[:2]

        w = x2 - x1
        h = y2 - y1
        if h == 0:
            return "unknown"

        ratio = w / h
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        if ratio < 0.6:
            # Narrow bbox → lateral view; animal faces toward center of frame
            # Convention: if bbox is in the left half of the image the animal
            # is likely facing right, so its visible side is the left flank.
            return "right" if center_x < img_w / 2 else "left"
        elif ratio > 1.5:
            # Wide, short bbox → frontal or rear
            return "frontal" if center_y < img_h / 2 else "rear"
        else:
            return "unknown"

    def _square_pad_crop(self, image_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        img_h, img_w = image_bgr.shape[:2]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half = max(x2 - x1, y2 - y1) // 2

        sx = max(cx - half, 0)
        sy = max(cy - half, 0)
        ex = min(cx + half, img_w)
        ey = min(cy + half, img_h)

        return image_bgr[sy:ey, sx:ex].copy()
