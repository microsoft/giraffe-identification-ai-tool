# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import os
import sys

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_EAR_PROMPT = "elephant ear."   # GroundingDINO period-terminated phrase


class EarDetector:
    """
    Zero-shot elephant ear detector using GroundingDINO (transformers >= 4.38).
    Call detect_ear(whole_body_crop_bgr) → ear_crop_bgr | None.
    """

    def __init__(self, device: str = "cuda"):
        import torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        model_id = "IDEA-Research/grounding-dino-base"
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
            self.model.eval()
            self._available = True
            logger.info("EarDetector (GroundingDINO) loaded on %s.", self.device)
        except Exception as exc:
            logger.warning("EarDetector unavailable (%s). Ear crops will be skipped.", exc)
            self._available = False

    def detect_ear(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float = 0.25,
        min_area_frac: float = 0.01,
    ) -> "np.ndarray | None":
        """
        Returns the highest-confidence ear bounding-box crop (BGR) or None.
        image_bgr should be the whole-animal crop from MegaDetector.
        """
        if not self._available:
            return None

        import torch
        from PIL import Image

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        h, w = image_bgr.shape[:2]

        inputs = self.processor(
            images=pil_img, text=_EAR_PROMPT, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=conf_threshold,
            text_threshold=0.20,
            target_sizes=[(h, w)],
        )

        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        if len(boxes) == 0:
            return None

        best = int(scores.argmax())
        x1, y1, x2, y2 = [int(v) for v in boxes[best].cpu().tolist()]

        # Filter out implausibly small detections
        area_frac = (x2 - x1) * (y2 - y1) / max(h * w, 1)
        if area_frac < min_area_frac:
            logger.debug("EarDetector: best box too small (%.3f); skipping.", area_frac)
            return None

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        return image_bgr[y1:y2, x1:x2].copy()


class ElephantDetector:
    def __init__(self, backend: str = "megadetector", conf: float = 0.5, device: str = "cuda"):
        if backend not in {"megadetector", "passthrough"}:
            raise ValueError(f"backend must be 'megadetector' or 'passthrough', got '{backend}'")
        self.conf = conf
        self.device = device
        self._model = None

        if backend == "megadetector":
            try:
                # PytorchWildlife imports 'models.yolo' internally, which collides
                # with our project's models/ package.  Temporarily hide both our
                # sys.path entry and the cached sys.modules['models'] so that
                # PytorchWildlife resolves its own internal modules correctly.
                _proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                _removed = [p for p in sys.path if os.path.normcase(os.path.abspath(p)) == os.path.normcase(_proj_root)]
                for p in _removed:
                    sys.path.remove(p)
                _saved_models = sys.modules.pop("models", None)
                try:
                    from PytorchWildlife.models import detection as pw_detection
                    self._model = pw_detection.MegaDetectorV5(device=device, pretrained=True)
                finally:
                    if _saved_models is not None:
                        sys.modules["models"] = _saved_models
                    sys.path.extend(_removed)
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
            # MegaDetector expects RGB; cap the long edge to 1280px for inference
            h, w = image_bgr.shape[:2]
            scale = min(1.0, 1280.0 / max(h, w))
            if scale < 1.0:
                infer_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
            else:
                infer_bgr = image_bgr
            image_rgb = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB)
            results = self._model.single_image_detection(image_rgb, det_conf_thres=self.conf)
            detections = results.get("detections", None)

            if detections is None or len(detections.xyxy) == 0:
                return None, "unknown"

            # Pick the highest-confidence animal detection (class 0 in MD v5)
            best_idx = None
            best_conf = -1.0
            for i, (conf_val, cls_id) in enumerate(
                zip(detections.confidence.tolist(), detections.class_id.tolist())
            ):
                if int(cls_id) == 0 and float(conf_val) >= self.conf and float(conf_val) > best_conf:
                    best_conf = float(conf_val)
                    best_idx = i

            if best_idx is None:
                return None, "unknown"

            # Scale bbox back to original image dimensions
            x1, y1, x2, y2 = [int(v / scale) for v in detections.xyxy[best_idx]]
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
