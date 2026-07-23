#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Cache precomputation and pair scoring CLI.

Experiment namespace only — do NOT use on production selected-v1 artifacts.
Writes all outputs under <experiment_namespace>/ inside <output_dir>.

Usage examples
--------------
Precompute feature cache from a crop manifest:

    python scripts/cache_precompute.py cache \
        --manifest crops.parquet \
        --output-dir experiments/local_features \
        --namespace my_exp \
        --backend lightglue \
        --device cpu

Score pairs from a pair table (CSV with query_crop_path, ref_crop_path, region):

    python scripts/cache_precompute.py pairs \
        --pairs pairs.csv \
        --output-dir experiments/pair_scores \
        --namespace my_exp \
        --backend lightglue \
        --device cpu

Resume behaviour
----------------
Both commands are content-addressed and resumable:
- Cache precompute: skips crops already present in the feature cache.
- Pair scoring: reads an existing scores.parquet and skips pairs whose
  (query_crop_id, ref_crop_id) are already present.

The output is always written to:
    <output_dir>/<namespace>/feature_cache/   (for cache precompute)
    <output_dir>/<namespace>/pair_scores.parquet  (for pair scoring)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

# Make repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_elephant import (
    LOCAL_MATCHER_BACKEND,
    LOCAL_MATCHER_KEYPOINTS,
    LOCAL_MATCHER_MIN_INLIERS,
    LOCAL_FEATURE_CACHE_MAX_LRU,
    LOCAL_SCORE_SCHEMA_VERSION,
)
from models.local_matcher import (
    StrictLocalMatcher,
    LocalMatcherError,
    STRICT_SUPPORTED_REGIONS,
)
from models.feature_cache import FeatureCache, FeatureCacheKey
from models.local_score_schema import (
    LocalPairScore,
    SCHEMA_VERSION,
    make_scoring_fingerprint,
    assert_pair_score_integrity,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("cache_precompute")

# Experiment guard: any output path must be under a namespace directory.
_EXPERIMENT_GUARD_MARKER = "experiments"


def _guard_namespace(output_dir: Path, namespace: str) -> Path:
    """Return the namespaced output path and verify it looks experimental."""
    out = output_dir / namespace
    out.mkdir(parents=True, exist_ok=True)
    # Warn but do not block if parent is not under an 'experiments' tree.
    parts = [p.lower() for p in out.parts]
    if _EXPERIMENT_GUARD_MARKER not in parts:
        logger.warning(
            "Output path %s is not under an 'experiments/' subtree. "
            "This tool is intended for experiment namespaces only.",
            out,
        )
    return out


# ---------------------------------------------------------------------------
# Cache precompute command
# ---------------------------------------------------------------------------

def cmd_cache(args: argparse.Namespace) -> int:
    """Precompute and persist SuperPoint features for all crops in a manifest."""
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        logger.error("Manifest file not found: %s", manifest_path)
        return 1

    logger.info("Loading manifest: %s", manifest_path)
    df = _load_manifest(manifest_path)

    required_cols = {"crop_id", "crop_path", "crop_kind"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error("Manifest missing required columns: %s", missing)
        return 1

    # Filter to supported regions only
    df = df[df["crop_kind"].isin(STRICT_SUPPORTED_REGIONS)].copy()
    if df.empty:
        logger.warning("No rows with supported crop_kind in manifest — nothing to cache.")
        return 0

    device = torch.device(args.device)
    logger.info("Initialising StrictLocalMatcher (backend=%s, device=%s)", args.backend, device)
    try:
        matcher = StrictLocalMatcher(
            backend=args.backend,
            max_keypoints=args.max_keypoints,
            device=device,
        )
    except LocalMatcherError as exc:
        logger.error("Failed to initialise matcher: %s", exc)
        return 1

    namespaced_dir = _guard_namespace(Path(args.output_dir), args.namespace)
    cache_dir = namespaced_dir / "feature_cache"
    cache = FeatureCache(
        cache_dir=cache_dir,
        model_fingerprint=matcher.model_fingerprint,
        max_lru_entries=LOCAL_FEATURE_CACHE_MAX_LRU,
    )

    orientations = ["original", "flipped"] if args.mirror else ["original"]
    total = len(df)
    processed = 0
    skipped = 0
    errors = 0

    logger.info("Processing %d crops (orientations: %s)...", total, orientations)
    t_start = time.perf_counter()

    for _, row in df.iterrows():
        crop_id = str(row["crop_id"])
        crop_path = str(row["crop_path"])

        if not os.path.isfile(crop_path):
            logger.warning("Missing crop file: %s (crop_id=%s)", crop_path, crop_id)
            errors += 1
            continue

        img = cv2.imread(crop_path)
        if img is None:
            logger.warning("Cannot read image: %s", crop_path)
            errors += 1
            continue

        for orientation in orientations:
            key = FeatureCacheKey.from_image_array(
                img, crop_id, orientation, matcher.model_fingerprint
            )
            if cache.get(key) is not None:
                skipped += 1
                continue
            try:
                bundle = matcher.extract_features(img, orientation=orientation)
                cache.put(key, bundle)
                processed += 1
            except LocalMatcherError as exc:
                logger.warning("Extraction failed for %s [%s]: %s", crop_id, orientation, exc)
                errors += 1

    elapsed = time.perf_counter() - t_start
    stats = cache.stats()
    logger.info(
        "Done. processed=%d skipped=%d errors=%d elapsed=%.1fs",
        processed, skipped, errors, elapsed,
    )
    logger.info("Cache stats: %s", stats)
    return 0 if errors == 0 else 2


# ---------------------------------------------------------------------------
# Pair scoring command
# ---------------------------------------------------------------------------

def cmd_pairs(args: argparse.Namespace) -> int:
    """Score a table of pairs and write results to parquet."""
    pairs_path = Path(args.pairs)
    if not pairs_path.is_file():
        logger.error("Pairs file not found: %s", pairs_path)
        return 1

    logger.info("Loading pairs table: %s", pairs_path)
    pairs_df = _load_manifest(pairs_path)

    required_cols = {"query_crop_id", "ref_crop_id", "query_crop_path", "ref_crop_path", "region"}
    missing = required_cols - set(pairs_df.columns)
    if missing:
        logger.error("Pairs table missing required columns: %s", missing)
        return 1

    # Filter unsupported regions
    pairs_df = pairs_df[pairs_df["region"].isin(STRICT_SUPPORTED_REGIONS)].copy()
    if pairs_df.empty:
        logger.warning("No rows with supported region in pairs table.")
        return 0

    device = torch.device(args.device)
    logger.info("Initialising StrictLocalMatcher (backend=%s, device=%s)", args.backend, device)
    try:
        matcher = StrictLocalMatcher(
            backend=args.backend,
            max_keypoints=args.max_keypoints,
            device=device,
        )
    except LocalMatcherError as exc:
        logger.error("Failed to initialise matcher: %s", exc)
        return 1

    namespaced_dir = _guard_namespace(Path(args.output_dir), args.namespace)

    # Optional feature cache
    cache = None
    if args.use_cache:
        cache_dir = namespaced_dir / "feature_cache"
        cache = FeatureCache(
            cache_dir=cache_dir,
            model_fingerprint=matcher.model_fingerprint,
            max_lru_entries=LOCAL_FEATURE_CACHE_MAX_LRU,
        )

    output_parquet = namespaced_dir / "pair_scores.parquet"

    # Resume: load existing results
    done_pairs: set[tuple[str, str]] = set()
    existing_rows: list[dict] = []
    if output_parquet.is_file():
        try:
            existing = pd.read_parquet(output_parquet)
            for _, row in existing.iterrows():
                done_pairs.add((str(row["query_crop_id"]), str(row["ref_crop_id"])))
                existing_rows.append(row.to_dict())
            logger.info("Resuming: %d pairs already scored.", len(done_pairs))
        except Exception as exc:
            logger.warning("Could not load existing results for resume: %s", exc)

    scoring_fp = make_scoring_fingerprint(
        backend=matcher.backend,
        model_fingerprint=matcher.model_fingerprint,
        schema_version=SCHEMA_VERSION,
        geom_model="region_default",
        mirror_search=True,
    )

    results: list[dict] = list(existing_rows)
    total = len(pairs_df)
    processed = 0
    skipped = 0
    errors = 0

    t_start = time.perf_counter()
    logger.info("Scoring %d pairs (resume: %d already done)...", total, len(done_pairs))

    for _, row in pairs_df.iterrows():
        q_id = str(row["query_crop_id"])
        r_id = str(row["ref_crop_id"])
        region = str(row["region"])

        if (q_id, r_id) in done_pairs:
            skipped += 1
            continue

        ps_dict = {
            "schema_version": SCHEMA_VERSION,
            "backend": matcher.backend,
            "model_fingerprint": matcher.model_fingerprint,
            "scoring_fingerprint": scoring_fp,
            "query_crop_id": q_id,
            "ref_crop_id": r_id,
            "region": region,
            "orientation": "original",
            "geom_model_used": "homography" if region == "ear" else "partial_affine",
            "n_raw_matches": 0,
            "n_inliers": 0,
            "inlier_ratio": 0.0,
            "geometric_spread": 0.0,
            "score": 0.0,
            "missing_file": False,
            "latency_ms": 0.0,
            "source_fingerprint": str(row.get("source_fingerprint", "")),
            "split_fingerprint": str(row.get("split_fingerprint", "")),
            "query_crop_kind": region,
            "ref_crop_kind": region,
        }

        q_path = str(row["query_crop_path"])
        r_path = str(row["ref_crop_path"])

        for path, label in ((q_path, "query"), (r_path, "ref")):
            if not os.path.isfile(path):
                logger.warning("Missing %s file: %s", label, path)
                ps_dict["missing_file"] = True
                break
        else:
            t0 = time.perf_counter()
            try:
                q_img = cv2.imread(q_path)
                r_img = cv2.imread(r_path)
                if q_img is None or r_img is None:
                    raise LocalMatcherError("cv2.imread returned None")

                if cache is not None and matcher.backend != "loftr":
                    from models.identity_scorer import LocalIdentityScorer
                    q_bundle, _ = cache.get_or_extract(matcher, q_img, q_id, "original")
                    r_orig, _ = cache.get_or_extract(matcher, r_img, r_id, "original")
                    r_flip, _ = cache.get_or_extract(matcher, r_img, r_id, "flipped")
                    res_orig = matcher.match_features(q_bundle, r_orig, region)
                    res_flip = matcher.match_features(q_bundle, r_flip, region)
                    result = res_flip if res_flip.n_inliers > res_orig.n_inliers else res_orig
                else:
                    result = matcher.score_pair_strict(q_img, r_img, region, mirror_search=True)

                ps_dict["orientation"] = result.orientation
                ps_dict["geom_model_used"] = result.geom.model_used
                ps_dict["n_raw_matches"] = result.geom.n_raw_matches
                ps_dict["n_inliers"] = result.n_inliers
                ps_dict["inlier_ratio"] = result.geom.inlier_ratio
                ps_dict["geometric_spread"] = result.geom.geometric_spread
                ps_dict["score"] = float(result.n_inliers)
            except LocalMatcherError as exc:
                logger.warning("Pair %s↔%s failed: %s", q_id, r_id, exc)
                errors += 1
            ps_dict["latency_ms"] = (time.perf_counter() - t0) * 1000

        results.append(ps_dict)
        processed += 1

        # Checkpoint every 100 pairs
        if processed % 100 == 0:
            _write_parquet(results, output_parquet)
            logger.info("Checkpoint: %d / %d pairs done.", processed + len(done_pairs), total)

    _write_parquet(results, output_parquet)
    elapsed = time.perf_counter() - t_start
    logger.info(
        "Done. processed=%d skipped=%d errors=%d elapsed=%.1fs output=%s",
        processed, skipped, errors, elapsed, output_parquet,
    )
    return 0 if errors == 0 else 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_manifest(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported manifest format: {suffix!r}; expected .parquet or .csv")


def _write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp_{uuid.uuid4().hex}_{path.name}"
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(str(tmp), str(path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Local feature cache precomputation and pair scoring (experiment only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # Shared arguments
    def _add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--output-dir", required=True, help="Root output directory.")
        sp.add_argument("--namespace", required=True, help="Experiment namespace subdirectory.")
        sp.add_argument("--backend", default=LOCAL_MATCHER_BACKEND, choices=["lightglue", "loftr"])
        sp.add_argument("--max-keypoints", type=int, default=LOCAL_MATCHER_KEYPOINTS)
        sp.add_argument("--device", default="cpu", help="PyTorch device string (e.g. cpu, cuda:0).")

    # cache subcommand
    sp_cache = sub.add_parser("cache", help="Precompute feature cache from a crop manifest.")
    _add_common(sp_cache)
    sp_cache.add_argument("--manifest", required=True, help="Crop manifest (.parquet or .csv).")
    sp_cache.add_argument(
        "--mirror", action="store_true", default=True,
        help="Also cache horizontally-flipped features (default: True).",
    )
    sp_cache.add_argument("--no-mirror", dest="mirror", action="store_false")

    # pairs subcommand
    sp_pairs = sub.add_parser("pairs", help="Score pairs from a pair table.")
    _add_common(sp_pairs)
    sp_pairs.add_argument(
        "--pairs", required=True,
        help="Pair table (.parquet or .csv) with columns: "
             "query_crop_id, ref_crop_id, query_crop_path, ref_crop_path, region.",
    )
    sp_pairs.add_argument(
        "--use-cache", action="store_true", default=False,
        help="Use feature cache for extraction (lightglue only).",
    )

    return p


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "cache":
        return cmd_cache(args)
    if args.command == "pairs":
        return cmd_pairs(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
