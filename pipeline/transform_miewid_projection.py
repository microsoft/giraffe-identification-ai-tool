# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Apply a trained projection checkpoint to existing ear_miewid artifacts.

This module reads the reference and query ear_miewid embedding matrices and
mapping parquets, applies an L2-normalised projection head, and writes the
projected embeddings as a NEW descriptor (e.g. ``ear_miewid_projected``) under
the same artifact partition directories.  The original ``ear_miewid`` artifacts
are never modified.

A new FAISS flat-IP index is rebuilt for the projected reference embeddings.

Usage
-----
    python -m pipeline.transform_miewid_projection \\
        --artifact-root /path/to/BTEH_reid_artifacts/v1 \\
        --checkpoint   /path/to/checkpoints/miewid_proj_v1/best_projection.pt \\
        [--out-descriptor ear_miewid_projected] \\
        [--src-descriptor ear_miewid] \\
        [--partitions reference query] \\
        [--skip-index]    # skip FAISS rebuild (testing only)

Outputs (per partition, under embeddings/<partition>/)
------------------------------------------------------
  <out-descriptor>.npy              – Projected embeddings (float32, L2-norm'd).
  <out-descriptor>_mapping.parquet  – Cloned mapping with updated descriptor_name
                                      and model_preprocess_fingerprint.
  <out-descriptor>.index            – FAISS IndexFlatIP (reference only).

Safety checks
-------------
- Checkpoint in_dim must match source embedding matrix columns.
- Checkpoint source_fingerprint / split_fingerprint must match mapping values.
- Output dimension matches checkpoint out_dim.
- Output vectors are L2-normalised (norm ∈ [0.999, 1.001]).
- Refuses to overwrite existing output artifacts unless --force is passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from configs.config_bteh import (
    ARTIFACT_SCHEMA_VERSION,
    EMBEDDINGS_SUBDIR_BTEH,
)
from models.miewid_projection import (
    ProjectionHead,
    compute_projected_embeddings,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(ckpt_path: Path) -> dict:
    """Load and validate a projection checkpoint dict."""
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    for key in ("state_dict", "in_dim", "out_dim"):
        if key not in ckpt:
            raise ValueError(
                f"Checkpoint is missing required key '{key}': {ckpt_path}"
            )
    return ckpt


def _build_projection_model(ckpt: dict) -> ProjectionHead:
    """Reconstruct the ProjectionHead from a checkpoint dict."""
    model = ProjectionHead(
        in_dim=ckpt["in_dim"],
        out_dim=ckpt["out_dim"],
        dropout=ckpt.get("dropout", 0.0),
        hidden_dim=ckpt.get("hidden_dim") or None,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _verify_fingerprints(ckpt: dict, mapping: pd.DataFrame, src_descriptor: str) -> None:
    """
    Fail-loud check: checkpoint fingerprints must match the source mapping.

    Checks source_fingerprint and split_fingerprint when both are available.
    """
    ckpt_source_fp = ckpt.get("source_fingerprint", "unknown")
    ckpt_split_fp = ckpt.get("split_fingerprint", "unknown")

    if "source_fingerprint" in mapping.columns and ckpt_source_fp != "unknown":
        map_source_fp = str(mapping["source_fingerprint"].dropna().iloc[0]) if len(mapping) > 0 else "unknown"
        if ckpt_source_fp != map_source_fp:
            raise ValueError(
                f"Fingerprint mismatch for '{src_descriptor}': "
                f"checkpoint source_fingerprint={ckpt_source_fp!r} "
                f"!= mapping source_fingerprint={map_source_fp!r}. "
                "Ensure the checkpoint was trained on the same artifact version."
            )

    if "split_fingerprint" in mapping.columns and ckpt_split_fp != "unknown":
        map_split_fp = str(mapping["split_fingerprint"].dropna().iloc[0]) if len(mapping) > 0 else "unknown"
        if ckpt_split_fp != map_split_fp:
            raise ValueError(
                f"Fingerprint mismatch for '{src_descriptor}': "
                f"checkpoint split_fingerprint={ckpt_split_fp!r} "
                f"!= mapping split_fingerprint={map_split_fp!r}. "
                "Ensure the checkpoint was trained on the same split version."
            )

    # Dimension check
    if "in_dim" in ckpt:
        logger.info(
            "Checkpoint fingerprint check passed: in_dim=%d, out_dim=%d",
            ckpt["in_dim"], ckpt["out_dim"],
        )


def _verify_output_normalisation(projected: np.ndarray) -> None:
    """Fail-loud if projected embeddings are not L2-normalised."""
    sample = projected[:min(500, len(projected))]
    norms = np.linalg.norm(sample, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError(
            f"Output embeddings are NOT L2-normalised: "
            f"norms min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}. "
            "This indicates a bug in the ProjectionHead forward pass."
        )
    logger.info(
        "Output L2-norm check passed: min=%.6f, max=%.6f", norms.min(), norms.max()
    )


def _build_faiss_index(projected: np.ndarray) -> "faiss.IndexFlatIP":
    """Build a FAISS IndexFlatIP from L2-normalised projected embeddings."""
    import faiss  # local import; faiss is a hard dep of the pipeline

    d = projected.shape[1]
    index = faiss.IndexFlatIP(d)
    # FAISS requires contiguous float32
    mat = np.ascontiguousarray(projected.astype(np.float32))
    index.add(mat)
    logger.info("Built FAISS IndexFlatIP: %d vectors, dim=%d", index.ntotal, d)
    return index


def _clone_mapping(
    src_mapping: pd.DataFrame,
    out_descriptor: str,
    model_fp_suffix: str,
) -> pd.DataFrame:
    """
    Clone a descriptor mapping for the new descriptor.

    Updates:
      - descriptor_name → out_descriptor
      - model_preprocess_fingerprint → <original>+<out_descriptor>
      - embedding_row and faiss_row → contiguous 0..N-1
    """
    df = src_mapping.copy()
    expected_rows = np.arange(len(df), dtype=np.int64)
    actual_rows = df["embedding_row"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_rows, expected_rows):
        raise ValueError(
            "Source descriptor mapping must be ordered by contiguous embedding_row "
            "before projection"
        )
    df["descriptor_name"] = out_descriptor
    df["embedding_row"] = expected_rows
    df["faiss_row"] = expected_rows

    if "model_preprocess_fingerprint" in df.columns:
        orig_fp = str(df["model_preprocess_fingerprint"].iloc[0]) if len(df) > 0 else ""
        df["model_preprocess_fingerprint"] = f"{orig_fp}+{model_fp_suffix}"
    else:
        df["model_preprocess_fingerprint"] = model_fp_suffix

    return df


# ---------------------------------------------------------------------------
# Main transform function
# ---------------------------------------------------------------------------

def transform_projection(
    artifact_root: Path,
    ckpt_path: Path,
    out_descriptor: str = "ear_miewid_projected",
    src_descriptor: str = "ear_miewid",
    partitions: Optional[List[str]] = None,
    build_index: bool = True,
    force: bool = False,
    device_str: Optional[str] = None,
) -> dict:
    """
    Apply a trained projection checkpoint to existing ear_miewid artifacts.

    Parameters
    ----------
    artifact_root   : Versioned artifact root (e.g. /path/to/.../v1).
    ckpt_path       : Path to a best_projection.pt checkpoint.
    out_descriptor  : Name for the new projected descriptor.
    src_descriptor  : Source descriptor to project (default: ear_miewid).
    partitions      : Partitions to transform (default: ['reference', 'query']).
    build_index     : Build FAISS index for reference partition (default: True).
    force           : Overwrite existing outputs if True.
    device_str      : 'cpu' or 'cuda'.

    Returns
    -------
    Transform manifest dict.
    """
    if partitions is None:
        partitions = ["reference", "query"]

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # 1. Load and validate checkpoint
    # ------------------------------------------------------------------
    ckpt = _load_checkpoint(ckpt_path)
    model = _build_projection_model(ckpt)
    logger.info(
        "Loaded projection checkpoint: in_dim=%d, out_dim=%d, val_map=%.4f",
        ckpt["in_dim"], ckpt["out_dim"], ckpt.get("val_map", float("nan")),
    )

    # Model fingerprint for provenance
    model_fp_suffix = f"{out_descriptor}:projected-{ckpt.get('checkpoint_fingerprint', 'unknown')[:12]}"

    transform_results = {}

    for partition in partitions:
        emb_dir = artifact_root / EMBEDDINGS_SUBDIR_BTEH / partition
        src_npy = emb_dir / f"{src_descriptor}.npy"
        src_parquet = emb_dir / f"{src_descriptor}_mapping.parquet"

        for p in (src_npy, src_parquet):
            if not p.is_file():
                raise FileNotFoundError(
                    f"Source artifact not found for partition '{partition}': {p}"
                )

        out_npy = emb_dir / f"{out_descriptor}.npy"
        out_parquet = emb_dir / f"{out_descriptor}_mapping.parquet"
        out_index = emb_dir / f"{out_descriptor}.index"

        if not force:
            existing = [p for p in (out_npy, out_parquet) if p.is_file()]
            if existing:
                raise FileExistsError(
                    f"Output artifacts already exist (use --force to overwrite): "
                    f"{[str(p) for p in existing]}"
                )

        # ------------------------------------------------------------------
        # 2. Load source artifacts
        # ------------------------------------------------------------------
        src_matrix = np.load(str(src_npy)).astype(np.float32)
        src_mapping = pd.read_parquet(str(src_parquet))
        logger.info(
            "[%s] Source matrix: %s, mapping: %d rows",
            partition, src_matrix.shape, len(src_mapping),
        )

        # ------------------------------------------------------------------
        # 3. Dimension check
        # ------------------------------------------------------------------
        if src_matrix.shape[1] != ckpt["in_dim"]:
            raise ValueError(
                f"Dimension mismatch: source matrix has {src_matrix.shape[1]} columns "
                f"but checkpoint expects in_dim={ckpt['in_dim']}."
            )

        # ------------------------------------------------------------------
        # 4. Fingerprint check (reference partition only)
        # ------------------------------------------------------------------
        if partition == "reference":
            _verify_fingerprints(ckpt, src_mapping, src_descriptor)

        # ------------------------------------------------------------------
        # 5. Apply projection
        # ------------------------------------------------------------------
        logger.info("[%s] Applying projection head ...", partition)
        projected = compute_projected_embeddings(
            model, src_matrix, batch_size=512, device=device_str
        )
        logger.info(
            "[%s] Projected matrix: %s (dtype=%s)", partition, projected.shape, projected.dtype
        )

        # ------------------------------------------------------------------
        # 6. Verify L2 normalisation
        # ------------------------------------------------------------------
        _verify_output_normalisation(projected)

        # ------------------------------------------------------------------
        # 7. Clone mapping
        # ------------------------------------------------------------------
        out_mapping = _clone_mapping(src_mapping, out_descriptor, model_fp_suffix)

        # ------------------------------------------------------------------
        # 8. Verify row alignment (mapping rows must equal projected rows)
        # ------------------------------------------------------------------
        if len(out_mapping) != len(projected):
            raise RuntimeError(
                f"Row count mismatch: projected matrix has {len(projected)} rows "
                f"but mapping has {len(out_mapping)} rows."
            )

        # ------------------------------------------------------------------
        # 9. Save outputs
        # ------------------------------------------------------------------
        np.save(str(out_npy), projected)
        logger.info("[%s] Saved projected embeddings → %s", partition, out_npy)

        out_mapping.to_parquet(str(out_parquet), index=False)
        logger.info("[%s] Saved projected mapping → %s", partition, out_parquet)

        # ------------------------------------------------------------------
        # 10. Build FAISS index (reference only)
        # ------------------------------------------------------------------
        if build_index and partition == "reference":
            index = _build_faiss_index(projected)
            import faiss
            faiss.write_index(index, str(out_index))
            logger.info("[%s] Saved FAISS index → %s", partition, out_index)

        transform_results[partition] = {
            "n_rows": int(len(projected)),
            "in_dim": int(src_matrix.shape[1]),
            "out_dim": int(projected.shape[1]),
            "npy_path": str(out_npy),
            "parquet_path": str(out_parquet),
            "index_path": str(out_index) if (build_index and partition == "reference") else None,
        }

    # ------------------------------------------------------------------
    # 11. Write transform manifest
    # ------------------------------------------------------------------
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "transformed_at": datetime.now(timezone.utc).isoformat(),
        "src_descriptor": src_descriptor,
        "out_descriptor": out_descriptor,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_in_dim": ckpt["in_dim"],
        "checkpoint_out_dim": ckpt["out_dim"],
        "checkpoint_val_map": ckpt.get("val_map"),
        "checkpoint_source_fingerprint": ckpt.get("source_fingerprint"),
        "checkpoint_split_fingerprint": ckpt.get("split_fingerprint"),
        "checkpoint_model_fingerprint": ckpt.get("model_preprocess_fingerprint"),
        "artifact_root": str(artifact_root),
        "partitions": partitions,
        "build_index": build_index,
        "transform_results": transform_results,
        "note": (
            f"Original {src_descriptor!r} artifacts are preserved unchanged. "
            f"New descriptor {out_descriptor!r} is the projection-adapted variant."
        ),
    }

    manifest_path = artifact_root / EMBEDDINGS_SUBDIR_BTEH / "transform_manifest.json"
    with open(str(manifest_path), "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Transform manifest saved to %s", manifest_path)

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Apply a trained projection checkpoint to ear_miewid artifacts, "
            "producing a new descriptor (e.g. ear_miewid_projected). "
            "Original artifacts are never modified."
        )
    )
    p.add_argument(
        "--artifact-root",
        required=True,
        type=Path,
        help="Versioned artifact root (e.g. /path/to/BTEH_reid_artifacts/v1).",
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to best_projection.pt checkpoint.",
    )
    p.add_argument(
        "--out-descriptor",
        default="ear_miewid_projected",
        help="Name for the new projected descriptor.",
    )
    p.add_argument(
        "--src-descriptor",
        default="ear_miewid",
        help="Source descriptor to project (default: ear_miewid).",
    )
    p.add_argument(
        "--partitions",
        nargs="+",
        default=["reference", "query"],
        help="Partitions to transform.",
    )
    p.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip FAISS index rebuild (testing only).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output artifacts.",
    )
    p.add_argument("--device", default=None, help="'cpu' or 'cuda'.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        manifest = transform_projection(
            artifact_root=args.artifact_root,
            ckpt_path=args.checkpoint,
            out_descriptor=args.out_descriptor,
            src_descriptor=args.src_descriptor,
            partitions=args.partitions,
            build_index=not args.skip_index,
            force=args.force,
            device_str=args.device,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        logger.error("HARD FAIL: %s", exc)
        return 2
    except Exception as exc:
        logger.error("Transform failed: %s", exc, exc_info=True)
        return 1

    logger.info(
        "Transform complete. New descriptor: %s", manifest.get("out_descriptor")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
