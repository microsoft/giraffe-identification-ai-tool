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

_EAR_PROMPT = "elephant ear."    # GroundingDINO period-terminated phrase
_HEAD_PROMPT = "elephant head."  # GroundingDINO period-terminated phrase


def _iou(box_a: list[float], box_b: list[float]) -> float:
    """Return intersection-over-union for two ``[x1, y1, x2, y2]`` boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def _nms(
    boxes: list[list[float]], scores: list[float], iou_threshold: float
) -> list[int]:
    """Non-maximum suppression; return kept indices sorted by score descending."""
    if len(boxes) != len(scores):
        raise ValueError(
            f"boxes and scores length mismatch: {len(boxes)} != {len(scores)}"
        )
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    kept: list[int] = []
    for index in order:
        if all(_iou(boxes[index], boxes[kept_index]) <= iou_threshold for kept_index in kept):
            kept.append(index)
    return kept


def _square_pad_crop_from_box(
    image_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    pad_frac: float = 0.15,
) -> np.ndarray:
    """Square-pad a box by ``pad_frac`` and crop it without stretching."""
    image_h, image_w = image_bgr.shape[:2]
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    if box_w == 0 or box_h == 0:
        return image_bgr[0:0, 0:0].copy()

    padded_w = box_w * (1.0 + 2.0 * pad_frac)
    padded_h = box_h * (1.0 + 2.0 * pad_frac)
    side = max(padded_w, padded_h)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(1, int(np.ceil(side)))
    raw_sx = int(np.floor(center_x - side / 2.0))
    raw_sy = int(np.floor(center_y - side / 2.0))
    raw_ex = raw_sx + side
    raw_ey = raw_sy + side
    sx, sy = max(0, raw_sx), max(0, raw_sy)
    ex, ey = min(image_w, raw_ex), min(image_h, raw_ey)
    crop = image_bgr[sy:ey, sx:ex].copy()
    return cv2.copyMakeBorder(
        crop,
        max(0, -raw_sy),
        max(0, raw_ey - image_h),
        max(0, -raw_sx),
        max(0, raw_ex - image_w),
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )


def _filter_valid_boxes(
    box_values: list[list[float]],
    score_values: list[float],
    h: int,
    w: int,
    conf_threshold: float,
    min_area_frac: float,
    max_area_frac: float,
    min_aspect: float,
    max_aspect: float,
) -> tuple[list[list[float]], list[float]]:
    """Filter raw GroundingDINO detections by score, area, and aspect ratio."""
    valid_boxes: list[list[float]] = []
    valid_scores: list[float] = []
    for box, score in zip(box_values, score_values):
        score = float(score)
        if score < conf_threshold:
            continue
        x1 = max(0.0, min(float(w), float(box[0])))
        y1 = max(0.0, min(float(h), float(box[1])))
        x2 = max(0.0, min(float(w), float(box[2])))
        y2 = max(0.0, min(float(h), float(box[3])))
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 0 or box_h <= 0:
            continue
        area_frac = box_w * box_h / max(h * w, 1)
        aspect = box_w / box_h
        if not min_area_frac <= area_frac <= max_area_frac:
            continue
        if not min_aspect <= aspect <= max_aspect:
            continue
        valid_boxes.append([x1, y1, x2, y2])
        valid_scores.append(score)
    return valid_boxes, valid_scores


class _GroundingDINOBackend:
    """Shared GroundingDINO processor + model — load once, inject into detectors.

    Create one instance and pass it to both ``EarDetector(backend=...)`` and
    ``HeadDetector(backend=...)`` to avoid loading the 700 MB model twice in
    the same process.
    """

    def __init__(self, device: str = "cuda"):
        import torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._available = False
        self.processor = None
        self.model = None
        _model_id = "IDEA-Research/grounding-dino-base"
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self.processor = AutoProcessor.from_pretrained(_model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                _model_id
            ).to(self.device)
            self.model.eval()
            self._available = True
            logger.info("GroundingDINO backend loaded on %s.", self.device)
        except Exception as exc:
            logger.warning("GroundingDINO backend unavailable (%s).", exc)

    def run_prompt(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        conf_threshold: float,
        text_threshold: float = 0.20,
    ) -> tuple[list[list[float]], list[float]]:
        """Run inference and return ``(box_list, score_list)`` in pixel coords."""
        import torch
        from PIL import Image

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        h, w = image_bgr.shape[:2]

        inputs = self.processor(
            images=pil_img, text=prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=conf_threshold,
            text_threshold=text_threshold,
            target_sizes=[(h, w)],
        )

        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        if len(boxes) == 0:
            return [], []

        box_list = (
            boxes.detach().cpu().tolist()
            if hasattr(boxes, "detach")
            else np.asarray(boxes).tolist()
        )
        score_list = (
            scores.detach().cpu().tolist()
            if hasattr(scores, "detach")
            else np.asarray(scores).tolist()
        )
        return box_list, score_list


class EarDetector:
    """
    Zero-shot elephant ear detector using GroundingDINO (transformers >= 4.38).
    Call detect_ear(whole_body_crop_bgr) → ear_crop_bgr | None.

    Pass ``backend=`` to share a pre-loaded :class:`_GroundingDINOBackend`
    instance with a :class:`HeadDetector` so the model is loaded only once.
    """

    def __init__(
        self,
        device: str = "cuda",
        backend: "_GroundingDINOBackend | None" = None,
    ):
        if backend is not None:
            # Share the caller-supplied backend.
            self._backend = backend
            self.device = backend.device
            self._available = backend._available
            self.processor = backend.processor
            self.model = backend.model
        else:
            _backend = _GroundingDINOBackend(device=device)
            self._backend = _backend
            self.device = _backend.device
            self._available = _backend._available
            self.processor = _backend.processor
            self.model = _backend.model
            if self._available:
                logger.info("EarDetector (GroundingDINO) loaded on %s.", self.device)
            else:
                logger.warning(
                    "EarDetector unavailable. Ear crops will be skipped."
                )

    def detect_ear(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float = 0.35,
        min_area_frac: float = 0.01,
        max_area_frac: float = 0.50,
    ) -> "np.ndarray | None":
        """
        Returns the highest-confidence ear bounding-box crop (BGR) or None.
        image_bgr should be the whole-animal crop from MegaDetector.
        """
        ears = self.detect_ears(
            image_bgr,
            conf_threshold=conf_threshold,
            min_area_frac=min_area_frac,
            max_area_frac=max_area_frac,
        )
        return ears[0]["crop"] if ears else None

    def detect_ears(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float = 0.35,
        min_area_frac: float = 0.01,
        max_area_frac: float = 0.50,
        min_aspect: float = 0.3,
        max_aspect: float = 3.5,
        iou_threshold: float = 0.5,
        require_available: bool = False,
    ) -> list[dict]:
        """Return up to two valid ears in deterministic left-to-right order."""
        if not self._available:
            if require_available:
                raise RuntimeError("EarDetector model is unavailable")
            return []
        if image_bgr is None or image_bgr.size == 0:
            return []

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
            return []

        box_values = boxes.detach().cpu().tolist() if hasattr(boxes, "detach") else np.asarray(boxes).tolist()
        score_values = scores.detach().cpu().tolist() if hasattr(scores, "detach") else np.asarray(scores).tolist()

        valid_boxes, valid_scores = _filter_valid_boxes(
            box_values, score_values, h, w,
            conf_threshold, min_area_frac, max_area_frac, min_aspect, max_aspect,
        )

        kept = _nms(valid_boxes, valid_scores, iou_threshold)[:2]
        detections = []
        for index in kept:
            x1, y1, x2, y2 = [
                int(round(value)) for value in valid_boxes[index]
            ]
            crop = _square_pad_crop_from_box(image_bgr, x1, y1, x2, y2)
            if crop.size == 0:
                continue
            detections.append(
                {
                    "box": [x1, y1, x2, y2],
                    "score": valid_scores[index],
                    "crop": crop,
                }
            )

        detections.sort(
            key=lambda detection: (
                (detection["box"][0] + detection["box"][2]) / 2.0,
                detection["box"][1],
                -detection["score"],
            )
        )
        for ordinal, detection in enumerate(detections):
            detection["ordinal"] = ordinal
        return detections


class HeadDetector:
    """Zero-shot elephant head detector using GroundingDINO (transformers >= 4.38).

    Returns at most **one** deterministic accepted result per call.  The single
    best-scoring candidate after NMS is returned; ties are broken by index
    (lower raw-box index wins) for full determinism.

    Pass ``backend=`` to share a pre-loaded :class:`_GroundingDINOBackend`
    with an :class:`EarDetector` so the model is loaded only once.
    """

    # Default detection constraints — intentionally independent from ear defaults.
    DEFAULT_CONF_THRESHOLD: float = 0.30
    DEFAULT_MIN_AREA_FRAC: float = 0.02
    DEFAULT_MAX_AREA_FRAC: float = 0.70
    DEFAULT_MIN_ASPECT: float = 0.40
    DEFAULT_MAX_ASPECT: float = 2.50
    DEFAULT_IOU_THRESHOLD: float = 0.50
    DEFAULT_PAD_FRAC: float = 0.10

    def __init__(
        self,
        device: str = "cuda",
        backend: "_GroundingDINOBackend | None" = None,
        prompt: str = _HEAD_PROMPT,
    ):
        self.prompt = prompt
        if backend is not None:
            self._backend = backend
            self.device = backend.device
            self._available = backend._available
            self.processor = backend.processor
            self.model = backend.model
        else:
            _backend = _GroundingDINOBackend(device=device)
            self._backend = _backend
            self.device = _backend.device
            self._available = _backend._available
            self.processor = _backend.processor
            self.model = _backend.model
            if self._available:
                logger.info("HeadDetector (GroundingDINO) loaded on %s.", self.device)
            else:
                logger.warning(
                    "HeadDetector unavailable. Head crops will be skipped."
                )

    def detect_head(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
        max_area_frac: float = DEFAULT_MAX_AREA_FRAC,
        min_aspect: float = DEFAULT_MIN_ASPECT,
        max_aspect: float = DEFAULT_MAX_ASPECT,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        pad_frac: float = DEFAULT_PAD_FRAC,
        require_available: bool = False,
    ) -> "dict | None":
        """Return the single best accepted head detection dict, or ``None``.

        The returned dict contains:
        ``box`` ([x1, y1, x2, y2] in pixels), ``score`` (float),
        ``crop`` (BGR ndarray), ``ordinal`` (always 0),
        ``source`` (always ``"grounding_dino"``).
        """
        if not self._available:
            if require_available:
                raise RuntimeError("HeadDetector model is unavailable")
            return None
        if image_bgr is None or image_bgr.size == 0:
            return None

        import torch
        from PIL import Image

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        h, w = image_bgr.shape[:2]

        inputs = self.processor(
            images=pil_img,
            text=getattr(self, "prompt", _HEAD_PROMPT),
            return_tensors="pt",
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

        box_values = (
            boxes.detach().cpu().tolist()
            if hasattr(boxes, "detach")
            else np.asarray(boxes).tolist()
        )
        score_values = (
            scores.detach().cpu().tolist()
            if hasattr(scores, "detach")
            else np.asarray(scores).tolist()
        )

        valid_boxes, valid_scores = _filter_valid_boxes(
            box_values, score_values, h, w,
            conf_threshold, min_area_frac, max_area_frac, min_aspect, max_aspect,
        )
        if not valid_boxes:
            return None

        # NMS; then take the highest-scoring survivor (index 0 = best score).
        kept = _nms(valid_boxes, valid_scores, iou_threshold)
        if not kept:
            return None

        best = kept[0]
        x1, y1, x2, y2 = [int(round(v)) for v in valid_boxes[best]]
        crop = _square_pad_crop_from_box(image_bgr, x1, y1, x2, y2, pad_frac=pad_frac)
        if crop.size == 0:
            return None

        return {
            "box": [x1, y1, x2, y2],
            "score": valid_scores[best],
            "crop": crop,
            "ordinal": 0,
            "source": "grounding_dino",
        }


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
            raise RuntimeError(f"MegaDetector inference failed: {exc}") from exc

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
