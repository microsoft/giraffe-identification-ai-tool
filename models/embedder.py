# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

import os
import sys
import logging
import numpy as np
import torch
import torch.nn.functional as F
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_elephant import GLOBAL_DESCRIPTORS

logger = logging.getLogger(__name__)

# Attempt to import wildlife_tools; graceful fallback to timm for megadescriptor
try:
    from wildlife_tools.features import DeepFeatures as _WildlifeDeepFeatures
    _WILDLIFE_TOOLS_AVAILABLE = True
except ImportError:
    _WILDLIFE_TOOLS_AVAILABLE = False
    logger.info("wildlife_tools not found; megadescriptor will use timm as backend.")

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_DEFAULT_BATCH_SIZE = 32


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (x / norms).astype(np.float32)


def _preprocess_bgr(image_bgr: np.ndarray, input_size: int) -> np.ndarray:
    """BGR → RGB, resize, float32 [0,1], ImageNet normalize. Returns (C, H, W)."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    image_rgb = image_rgb.astype(np.float32) / 255.0
    image_rgb = (image_rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return image_rgb.transpose(2, 0, 1)  # (C, H, W)


class GlobalEmbedder:
    """
    Produces L2-normalized global deep descriptors for a single crop image.
    Supports 'megadescriptor' and 'miewid' backends.
    """

    def __init__(self, backend: str, device: str = "cuda"):
        if backend not in GLOBAL_DESCRIPTORS:
            raise ValueError(f"Unknown backend '{backend}'. Choose from {list(GLOBAL_DESCRIPTORS)}.")

        self.backend = backend
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        cfg = GLOBAL_DESCRIPTORS[backend]
        self.model_id = cfg["model_id"]
        self.dim = cfg["dim"]
        self.input_size = cfg["input_size"]

        self._model = self._load_model()
        self._model.eval()
        logger.info("GlobalEmbedder('%s') loaded on %s", backend, self.device)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> torch.nn.Module:
        if self.backend == "megadescriptor":
            return self._load_megadescriptor()
        else:
            return self._load_miewid()

    def _load_megadescriptor(self) -> torch.nn.Module:
        import timm
        # timm requires "hf_hub:" prefix for HuggingFace-hosted models
        hf_name = f"hf_hub:{self.model_id}" if not self.model_id.startswith("hf_hub:") else self.model_id
        model = timm.create_model(hf_name, pretrained=True, num_classes=0)
        model = model.to(self.device)
        logger.info("megadescriptor loaded via timm (hf_hub).")
        return model

    def _load_miewid(self) -> torch.nn.Module:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        model = model.to(self.device)
        logger.info("miewid loaded via transformers.AutoModel.")
        return model

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _tensor_from_array(self, chw: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(chw).unsqueeze(0).to(self.device)

    def _run_batch(self, batch_tensor: torch.Tensor) -> np.ndarray:
        """Forward pass on a (B, C, H, W) tensor; returns (B, D) numpy float32."""
        with torch.no_grad():
            out = self._model(batch_tensor)
        if isinstance(out, torch.Tensor):
            feats = out
        elif hasattr(out, "last_hidden_state"):
            # HuggingFace BaseModelOutput — mean-pool token dim
            feats = out.last_hidden_state.mean(dim=1)
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output
        else:
            raise RuntimeError(f"Unrecognised model output type: {type(out)}")

        if feats.dim() > 2:
            # Global average pool any spatial dims
            feats = feats.flatten(2).mean(dim=-1)

        return feats.cpu().float().numpy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        """Returns float32 (D,), L2-normalized."""
        chw = _preprocess_bgr(image_bgr, self.input_size)
        tensor = self._tensor_from_array(chw)
        feats = self._run_batch(tensor)          # (1, D)
        return _l2_normalize(feats)[0]           # (D,)

    def embed_batch(self, images: list, batch_size: int = _DEFAULT_BATCH_SIZE) -> np.ndarray:
        """Returns float32 (B, D), each row L2-normalized."""
        all_feats = []
        for start in range(0, len(images), batch_size):
            batch_imgs = images[start: start + batch_size]
            batch_chw = [_preprocess_bgr(img, self.input_size) for img in batch_imgs]
            batch_tensor = torch.from_numpy(np.stack(batch_chw, axis=0)).to(self.device)
            feats = self._run_batch(batch_tensor)   # (b, D)
            all_feats.append(feats)
        stacked = np.concatenate(all_feats, axis=0)   # (B, D)
        return _l2_normalize(stacked)
