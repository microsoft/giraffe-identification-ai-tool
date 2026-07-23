# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Persistent disk feature cache for StrictLocalMatcher.

Cache key: (crop_id, content_sha256, orientation, backend_model_fingerprint)

On-disk format: .npz file (numpy compressed archive) with:
    - keypoints, descriptors, scores, image_shape
    - metadata stored as .npz attributes (orientation, backend, fingerprint)
    - a sha256 integrity header over the serialised keypoints+descriptors

Architecture:
    FeatureCache wraps a configurable-size in-memory LRU dict (front cache)
    backed by a directory tree on disk.  Disk writes are atomic (write to a
    temp file in the same directory, then os.replace).  Corruption is
    detected by re-checking the content SHA-256 stored inside the file.
    Stale entries (fingerprint mismatch) are never returned.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Subdirectory per backend fingerprint so different model configs never collide.
_CACHE_VERSION = "fc-v1"


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureCacheKey:
    """Immutable cache key for a single extracted feature set."""
    crop_id: str
    content_sha256: str   # SHA-256 hex of the raw image bytes
    orientation: str      # 'original' | 'flipped'
    model_fingerprint: str  # from StrictLocalMatcher.model_fingerprint

    def cache_path(self, root: Path) -> Path:
        """
        Deterministic path under *root*.
        Sharded by first 4 hex chars of content_sha256 to avoid huge flat dirs.
        """
        shard = self.content_sha256[:4]
        stem = (
            f"{self.crop_id}__{self.content_sha256[:16]}"
            f"__{self.orientation}__{self.model_fingerprint}"
        )
        return root / _CACHE_VERSION / shard / f"{stem}.npz"

    @staticmethod
    def from_image_bytes(
        image_bytes: bytes,
        crop_id: str,
        orientation: str,
        model_fingerprint: str,
    ) -> "FeatureCacheKey":
        sha = hashlib.sha256(image_bytes).hexdigest()
        return FeatureCacheKey(
            crop_id=crop_id,
            content_sha256=sha,
            orientation=orientation,
            model_fingerprint=model_fingerprint,
        )

    @staticmethod
    def from_image_array(
        image_bgr,  # np.ndarray
        crop_id: str,
        orientation: str,
        model_fingerprint: str,
    ) -> "FeatureCacheKey":
        """Compute SHA-256 directly from the ndarray byte buffer (C-contiguous)."""
        arr = np.ascontiguousarray(image_bgr)
        sha = hashlib.sha256(arr.tobytes()).hexdigest()
        return FeatureCacheKey(
            crop_id=crop_id,
            content_sha256=sha,
            orientation=orientation,
            model_fingerprint=model_fingerprint,
        )


# ---------------------------------------------------------------------------
# Bundle serialisation helpers
# ---------------------------------------------------------------------------

def _integrity_hash(kpts: np.ndarray, descs: np.ndarray) -> str:
    """SHA-256 over concatenation of the two arrays' raw bytes."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(kpts).tobytes())
    h.update(np.ascontiguousarray(descs).tobytes())
    return h.hexdigest()


def _bundle_to_npz(bundle) -> bytes:
    """Serialise a FeatureBundle to compressed npz bytes."""
    from models.local_matcher import FeatureBundle  # local import to avoid circular
    buf = io.BytesIO()
    integrity = _integrity_hash(bundle.keypoints, bundle.descriptors)
    np.savez_compressed(
        buf,
        keypoints=bundle.keypoints,
        descriptors=bundle.descriptors,
        scores=bundle.scores,
        image_shape=np.array(bundle.image_shape, dtype=np.int64),
        # Metadata stored as 0-d arrays (npz only supports array types)
        orientation=np.array(bundle.orientation),
        backend=np.array(bundle.backend),
        model_fingerprint=np.array(bundle.model_fingerprint),
        integrity_hash=np.array(integrity),
    )
    return buf.getvalue()


def _npz_to_bundle(data: bytes) -> Optional[object]:
    """
    Deserialise npz bytes to a FeatureBundle.
    Returns None if the integrity check fails (corruption detected).
    """
    from models.local_matcher import FeatureBundle  # local import
    try:
        buf = io.BytesIO(data)
        npz = np.load(buf, allow_pickle=False)
        kpts = npz["keypoints"]
        descs = npz["descriptors"]
        scores = npz["scores"]
        image_shape = tuple(int(x) for x in npz["image_shape"].tolist())
        orientation = str(npz["orientation"])
        backend = str(npz["backend"])
        model_fingerprint = str(npz["model_fingerprint"])
        stored_hash = str(npz["integrity_hash"])
    except Exception as exc:
        logger.warning("Feature cache: npz deserialisation failed: %s", exc)
        return None

    expected = _integrity_hash(kpts, descs)
    if expected != stored_hash:
        logger.warning(
            "Feature cache: integrity mismatch (stored=%s, computed=%s) — discarding.",
            stored_hash[:12], expected[:12],
        )
        return None

    return FeatureBundle(
        keypoints=kpts,
        descriptors=descs,
        scores=scores,
        image_shape=image_shape,
        orientation=orientation,
        backend=backend,
        model_fingerprint=model_fingerprint,
    )


# ---------------------------------------------------------------------------
# LRU memory front-cache
# ---------------------------------------------------------------------------

class _LRUCache:
    """Simple thread-unsafe ordered-dict LRU cache."""

    def __init__(self, maxsize: int):
        if maxsize < 1:
            raise ValueError(f"LRU maxsize must be >= 1, got {maxsize}")
        self.maxsize = maxsize
        self._store: OrderedDict = OrderedDict()

    def get(self, key: FeatureCacheKey):
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: FeatureCacheKey, value) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: FeatureCacheKey) -> bool:
        return key in self._store

    def evict(self, key: FeatureCacheKey) -> bool:
        """Remove a key if present; return True if evicted."""
        if key in self._store:
            del self._store[key]
            return True
        return False


# ---------------------------------------------------------------------------
# Persistent feature cache
# ---------------------------------------------------------------------------

class FeatureCache:
    """
    Two-level cache for SuperPoint FeatureBundle objects.

    Level 1 (front): in-memory LRU dict (fast, bounded size).
    Level 2 (back):  compressed .npz files on disk (persistent, content-addressed).

    Parameters
    ----------
    cache_dir:
        Root directory for on-disk cache files.  Will be created if absent.
    model_fingerprint:
        Fingerprint of the model whose features are cached.  Entries from a
        different fingerprint are never returned (stale detection).
    max_lru_entries:
        Number of bundles held in the memory LRU front cache.

    Thread safety
    -------------
    Not thread-safe.  Use a single cache per process or protect externally.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        model_fingerprint: str,
        max_lru_entries: int = 256,
    ):
        self.cache_dir = Path(cache_dir)
        self.model_fingerprint = model_fingerprint
        self._lru = _LRUCache(maxsize=max_lru_entries)
        self._hits_lru = 0
        self._hits_disk = 0
        self._misses = 0
        self._writes = 0
        self._corruption_evictions = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, key: FeatureCacheKey):
        """
        Return the cached FeatureBundle for *key*, or None on miss.

        Stale entries (fingerprint mismatch) are treated as misses and
        removed from the LRU front cache.  On-disk corrupt entries are
        discarded (and the corrupt file is removed) and treated as misses.
        """
        if key.model_fingerprint != self.model_fingerprint:
            logger.debug("Cache miss: fingerprint mismatch for %s", key.crop_id)
            self._misses += 1
            return None

        # Level 1: memory LRU
        bundle = self._lru.get(key)
        if bundle is not None:
            self._hits_lru += 1
            return bundle

        # Level 2: disk
        path = key.cache_path(self.cache_dir)
        if path.is_file():
            try:
                data = path.read_bytes()
            except OSError as exc:
                logger.warning("Feature cache: disk read failed for %s: %s", path, exc)
                self._misses += 1
                return None

            bundle = _npz_to_bundle(data)
            if bundle is None:
                # Corruption detected — remove the bad file
                logger.warning("Feature cache: removing corrupt file %s", path)
                self._corruption_evictions += 1
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
                self._misses += 1
                return None

            # Stale fingerprint check (should not happen if key is correct,
            # but guard against manual file manipulation)
            if bundle.model_fingerprint != self.model_fingerprint:
                logger.warning(
                    "Feature cache: stale fingerprint in %s — discarding.", path
                )
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
                self._misses += 1
                return None

            self._lru.put(key, bundle)
            self._hits_disk += 1
            return bundle

        self._misses += 1
        return None

    def put(self, key: FeatureCacheKey, bundle) -> None:
        """
        Store a FeatureBundle under *key*.

        Writes atomically to disk (temp file + os.replace).
        Also inserts into the memory LRU.

        Raises ValueError if the bundle's model_fingerprint does not match
        the cache's model_fingerprint.
        """
        if bundle.model_fingerprint != self.model_fingerprint:
            raise ValueError(
                f"Cannot cache bundle with fingerprint {bundle.model_fingerprint!r} "
                f"in a cache configured for {self.model_fingerprint!r}"
            )

        # Memory LRU first
        self._lru.put(key, bundle)

        # Disk write (atomic)
        path = key.cache_path(self.cache_dir)
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = path.parent / f".tmp_{uuid.uuid4().hex}_{path.name}"
        try:
            data = _bundle_to_npz(bundle)
            tmp_path.write_bytes(data)
            os.replace(str(tmp_path), str(path))
            self._writes += 1
        except Exception as exc:
            logger.error("Feature cache: disk write failed for %s: %s", path, exc)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def invalidate(self, key: FeatureCacheKey) -> bool:
        """
        Evict *key* from both memory and disk.
        Returns True if anything was removed.
        """
        removed_lru = self._lru.evict(key)
        path = key.cache_path(self.cache_dir)
        removed_disk = False
        if path.is_file():
            try:
                path.unlink()
                removed_disk = True
            except OSError as exc:
                logger.warning("Feature cache: could not remove %s: %s", path, exc)
        return removed_lru or removed_disk

    def stats(self) -> dict:
        return {
            "hits_lru": self._hits_lru,
            "hits_disk": self._hits_disk,
            "misses": self._misses,
            "writes": self._writes,
            "corruption_evictions": self._corruption_evictions,
            "lru_size": len(self._lru),
        }

    # ------------------------------------------------------------------
    # Convenience: extract-or-cache
    # ------------------------------------------------------------------

    def get_or_extract(
        self,
        matcher,  # StrictLocalMatcher
        image_bgr,  # np.ndarray
        crop_id: str,
        orientation: str = "original",
    ):
        """
        Return cached FeatureBundle if available; otherwise extract, cache, and return.

        Parameters
        ----------
        matcher:
            A StrictLocalMatcher instance (backend must not be 'loftr').
        image_bgr:
            (H, W, 3) uint8 BGR numpy array.
        crop_id:
            Identifier for this crop (used in the cache key).
        orientation:
            'original' or 'flipped'.
        """
        key = FeatureCacheKey.from_image_array(
            image_bgr, crop_id, orientation, self.model_fingerprint
        )
        bundle = self.get(key)
        if bundle is not None:
            return bundle, key

        bundle = matcher.extract_features(image_bgr, orientation=orientation)
        self.put(key, bundle)
        return bundle, key


# ---------------------------------------------------------------------------
# LoFTR pair result cache (separate, keyed by query+ref content hashes)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairCacheKey:
    """Cache key for a LoFTR pairwise result."""
    query_sha256: str
    ref_sha256: str
    orientation: str       # reference orientation
    model_fingerprint: str

    def cache_path(self, root: Path) -> Path:
        combined = hashlib.sha256(
            (self.query_sha256 + self.ref_sha256 + self.orientation).encode()
        ).hexdigest()
        shard = combined[:4]
        stem = f"{combined}__{self.model_fingerprint}"
        return root / "pairs" / _CACHE_VERSION / shard / f"{stem}.npz"


class PairResultCache:
    """
    Disk+LRU cache for LoFTR pairwise MatchResult objects.

    Stores a compact subset of MatchResult fields (not the full viz payload).
    Used only when LoFTR pilot is approved; the cache is separate from the
    SuperPoint feature cache.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        model_fingerprint: str,
        max_lru_entries: int = 128,
    ):
        self.cache_dir = Path(cache_dir)
        self.model_fingerprint = model_fingerprint
        self._lru: OrderedDict = OrderedDict()
        self._maxsize = max(1, max_lru_entries)

    @staticmethod
    def key_from_arrays(query_bgr, ref_bgr, orientation: str, model_fingerprint: str) -> PairCacheKey:
        import numpy as np
        q_sha = hashlib.sha256(np.ascontiguousarray(query_bgr).tobytes()).hexdigest()
        r_sha = hashlib.sha256(np.ascontiguousarray(ref_bgr).tobytes()).hexdigest()
        return PairCacheKey(q_sha, r_sha, orientation, model_fingerprint)

    def _lru_get(self, key: PairCacheKey):
        if key not in self._lru:
            return None
        self._lru.move_to_end(key)
        return self._lru[key]

    def _lru_put(self, key: PairCacheKey, value) -> None:
        self._lru[key] = value
        self._lru.move_to_end(key)
        while len(self._lru) > self._maxsize:
            self._lru.popitem(last=False)

    def get(self, key: PairCacheKey):
        if key.model_fingerprint != self.model_fingerprint:
            return None
        val = self._lru_get(key)
        if val is not None:
            return val
        path = key.cache_path(self.cache_dir)
        if not path.is_file():
            return None
        try:
            data = path.read_bytes()
            buf = io.BytesIO(data)
            npz = np.load(buf, allow_pickle=False)
            result = {k: npz[k] for k in npz.files}
            self._lru_put(key, result)
            return result
        except Exception as exc:
            logger.warning("PairResultCache: read failed for %s: %s", path, exc)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def put(self, key: PairCacheKey, n_inliers: int, inlier_ratio: float, geom_spread: float, model_used: str) -> None:
        if key.model_fingerprint != self.model_fingerprint:
            raise ValueError("Fingerprint mismatch in PairResultCache.put")
        payload = {
            "n_inliers": np.array(n_inliers, dtype=np.int32),
            "inlier_ratio": np.array(inlier_ratio, dtype=np.float32),
            "geom_spread": np.array(geom_spread, dtype=np.float32),
            "model_used": np.array(model_used),
        }
        self._lru_put(key, payload)
        path = key.cache_path(self.cache_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".tmp_{uuid.uuid4().hex}_{path.name}"
        try:
            buf = io.BytesIO()
            np.savez_compressed(buf, **payload)
            tmp.write_bytes(buf.getvalue())
            os.replace(str(tmp), str(path))
        except Exception as exc:
            logger.error("PairResultCache: write failed: %s", exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
