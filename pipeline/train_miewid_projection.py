# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Train a projection-head metric adaptation on precomputed ear_miewid embeddings.

This module performs PROJECTION-HEAD ONLY training – the MiewID backbone is
frozen and never loaded.  All learning is applied to a small Linear projection
head that maps 2152-D (or configured) normalised embeddings to a lower-
dimensional space, then re-normalises.

Usage
-----
    python -m pipeline.train_miewid_projection \\
        --artifact-root /path/to/BTEH_reid_artifacts/v1 \\
        --out-dir /path/to/BTEH_reid_artifacts/v1/checkpoints/miewid_proj_v1 \\
        [--out-dim 512] \\
        [--dropout 0.1] \\
        [--hidden-dim 0] \\
        [--loss triplet|supcon|both] \\
        [--arcface] \\
        [--epochs 50] \\
        [--P 16] [--K 4] \\
        [--lr 1e-3] \\
        [--seed 42] \\
        [--min-map-delta 0.005] \\
        [--device cpu|cuda]

Outputs (under --out-dir)
-------------------------
  best_projection.pt          – Best checkpoint (by inner-val mAP).
  training_manifest.json      – Full provenance, hyperparams, curves, gate.
  training_curves.json        – Per-epoch loss and metric curves.
  experiment_diagnostics.json – Always written, even for rejected experiments.

Safety
------
- Hard-fails if any probe/held-out image_id enters training.
- Hard-fails if any held-out individual_id is in the training identity pool.
- Projection-only: backbone weights are never loaded or modified.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from configs.config_bteh import (
    ARTIFACT_SCHEMA_VERSION,
    EMBEDDINGS_SUBDIR_BTEH,
    SPLITS_FILENAME,
    SPLITS_SUBDIR,
)
from models.miewid_projection import (
    ArcFaceHead,
    AdoptionGateResult,
    EmbeddingRecord,
    InnerSplitResult,
    ProjectionHead,
    PxKDataset,
    PxKSampler,
    adoption_gate,
    batch_hard_triplet_loss,
    build_inner_split,
    build_px_k_dataset,
    compute_projected_embeddings,
    get_labels_for_image_ids,
    retrieval_map_top1,
    supervised_contrastive_loss,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Artifact loading helpers
# ---------------------------------------------------------------------------

def _load_ear_miewid_artifacts(
    artifact_root: Path,
    partition: str = "reference",
    descriptor: str = "ear_miewid",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load embedding matrix and mapping parquet for a descriptor."""
    emb_dir = artifact_root / EMBEDDINGS_SUBDIR_BTEH / partition
    npy_path = emb_dir / f"{descriptor}.npy"
    parquet_path = emb_dir / f"{descriptor}_mapping.parquet"

    for p in (npy_path, parquet_path):
        if not p.is_file():
            raise FileNotFoundError(f"Expected artifact not found: {p}")

    matrix = np.load(str(npy_path)).astype(np.float32)
    mapping = pd.read_parquet(str(parquet_path))
    logger.info(
        "Loaded %s: matrix=%s, mapping=%d rows",
        descriptor, matrix.shape, len(mapping),
    )
    return matrix, mapping


def _load_splits(artifact_root: Path) -> pd.DataFrame:
    """Load the BTEH splits parquet."""
    splits_path = artifact_root / SPLITS_SUBDIR / SPLITS_FILENAME
    if not splits_path.is_file():
        raise FileNotFoundError(f"Splits parquet not found: {splits_path}")
    splits_df = pd.read_parquet(str(splits_path))
    logger.info("Loaded splits: %d rows, split counts: %s",
                len(splits_df),
                splits_df["split"].value_counts().to_dict())
    return splits_df


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _one_epoch(
    model: ProjectionHead,
    arcface_head: Optional[ArcFaceHead],
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_mode: str,
    device: torch.device,
    triplet_margin: float = 0.3,
    supcon_temperature: float = 0.07,
    arcface_weight: float = 0.3,
    epoch: int = 0,
) -> Dict[str, float]:
    """Run one training epoch, return loss components."""
    model.train()
    if arcface_head is not None:
        arcface_head.train()

    total_loss = 0.0
    total_metric_loss = 0.0
    total_arcface_loss = 0.0
    n_batches = 0

    ce_loss_fn = nn.CrossEntropyLoss()

    for batch in dataloader:
        embs_in, labels, _ = batch
        embs_in = embs_in.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        projected = model(embs_in)

        # Metric loss
        if loss_mode == "triplet":
            metric_loss = batch_hard_triplet_loss(projected, labels, margin=triplet_margin)
        elif loss_mode == "supcon":
            metric_loss = supervised_contrastive_loss(projected, labels, temperature=supcon_temperature)
        elif loss_mode == "both":
            t_loss = batch_hard_triplet_loss(projected, labels, margin=triplet_margin)
            s_loss = supervised_contrastive_loss(projected, labels, temperature=supcon_temperature)
            metric_loss = 0.5 * t_loss + 0.5 * s_loss
        else:
            raise ValueError(f"Unknown loss mode: {loss_mode}")

        loss = metric_loss

        # Optional ArcFace auxiliary loss
        if arcface_head is not None:
            arc_logits = arcface_head(projected, labels)
            arc_loss = ce_loss_fn(arc_logits, labels)
            loss = (1.0 - arcface_weight) * metric_loss + arcface_weight * arc_loss
            total_arcface_loss += arc_loss.item()

        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if arcface_head is not None:
            nn.utils.clip_grad_norm_(arcface_head.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_metric_loss += metric_loss.item()
        n_batches += 1

    if n_batches == 0:
        return {"total": 0.0, "metric": 0.0, "arcface": 0.0}

    return {
        "total": total_loss / n_batches,
        "metric": total_metric_loss / n_batches,
        "arcface": total_arcface_loss / n_batches if arcface_head is not None else 0.0,
    }


@torch.no_grad()
def _validate(
    model: ProjectionHead,
    matrix: np.ndarray,
    train_row_indices: np.ndarray,
    train_labels: np.ndarray,
    val_row_indices: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
) -> Tuple[float, float]:
    """Compute retrieval mAP and top-1 on inner validation set."""
    model.eval()
    model_device = next(model.parameters()).device

    # Project train embeddings (reference for retrieval)
    train_embs = compute_projected_embeddings(
        model, matrix[train_row_indices], device=str(device)
    )
    # Project val embeddings (queries)
    val_embs = compute_projected_embeddings(
        model, matrix[val_row_indices], device=str(device)
    )

    return retrieval_map_top1(val_embs, val_labels, train_embs, train_labels)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_projection(
    artifact_root: Path,
    out_dir: Path,
    descriptor: str = "ear_miewid",
    out_dim: int = 512,
    dropout: float = 0.0,
    hidden_dim: Optional[int] = None,
    loss_mode: str = "both",
    use_arcface: bool = False,
    arcface_s: float = 32.0,
    arcface_m: float = 0.50,
    arcface_weight: float = 0.3,
    epochs: int = 50,
    P: int = 16,
    K: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    min_map_delta: float = 0.005,
    min_top1_delta: float = 0.0,
    instability_threshold: float = -0.05,
    triplet_margin: float = 0.3,
    supcon_temperature: float = 0.07,
    early_stop_patience: int = 10,
    device_str: Optional[str] = None,
    partition: str = "reference",
) -> Dict:
    """
    Train a projection head and return a full experiment manifest.

    This is PROJECTION-ONLY training. No backbone is loaded or trained.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random as _random
    _random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info("Training device: %s", device)

    # ------------------------------------------------------------------
    # 1. Load artifacts
    # ------------------------------------------------------------------
    matrix, mapping = _load_ear_miewid_artifacts(artifact_root, partition, descriptor)
    splits_df = _load_splits(artifact_root)

    in_dim = matrix.shape[1]
    logger.info("Input embedding dimension: %d, out_dim: %d", in_dim, out_dim)

    # Verify L2 normalisation (sample check)
    sample_norms = np.linalg.norm(matrix[:min(100, len(matrix))], axis=1)
    if not np.allclose(sample_norms, 1.0, atol=1e-3):
        logger.warning(
            "Input embeddings are not L2-normalised (sample norms: mean=%.4f, "
            "std=%.4f). Proceeding anyway – outputs will still be normalised.",
            sample_norms.mean(), sample_norms.std(),
        )

    # Collect fingerprints for manifest
    source_fp = str(mapping["source_fingerprint"].iloc[0]) if "source_fingerprint" in mapping.columns else "unknown"
    split_fp = str(mapping["split_fingerprint"].iloc[0]) if "split_fingerprint" in mapping.columns else "unknown"
    model_fp = str(mapping["model_preprocess_fingerprint"].iloc[0]) if "model_preprocess_fingerprint" in mapping.columns else "unknown"

    # ------------------------------------------------------------------
    # 2. Build inner split
    # ------------------------------------------------------------------
    inner_split = build_inner_split(mapping, splits_df, seed=seed)

    logger.info(
        "Inner split: %d train images, %d val images",
        len(inner_split.train_image_ids),
        len(inner_split.val_image_ids),
    )

    # ------------------------------------------------------------------
    # 3. Build datasets
    # ------------------------------------------------------------------
    train_dataset, train_label_to_indices, identity_to_label = build_px_k_dataset(
        matrix, mapping, inner_split.train_image_ids
    )
    n_identities = len(identity_to_label)
    logger.info(
        "Training set: %d crops, %d identities", len(train_dataset), n_identities
    )

    # Validate P and K bounds
    P_actual = min(P, n_identities)
    if P_actual < P:
        logger.warning(
            "Requested P=%d but only %d identities available; using P=%d.",
            P, n_identities, P_actual,
        )

    sampler = PxKSampler(
        label_to_indices=train_label_to_indices,
        P=P_actual,
        K=K,
        seed=seed,
    )
    dataloader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Validation row indices and labels
    val_row_indices, val_labels = get_labels_for_image_ids(
        inner_split.val_image_ids, mapping, identity_to_label
    )
    train_row_indices, train_labels = get_labels_for_image_ids(
        inner_split.train_image_ids, mapping, identity_to_label
    )
    logger.info(
        "Val set: %d crops, %d unique identities with val fold",
        len(val_row_indices),
        len(np.unique(val_labels)),
    )

    # ------------------------------------------------------------------
    # 4. Build model
    # ------------------------------------------------------------------
    model = ProjectionHead(
        in_dim=in_dim,
        out_dim=out_dim,
        dropout=dropout,
        hidden_dim=hidden_dim if hidden_dim and hidden_dim > 0 else None,
    ).to(device)

    arcface_head: Optional[ArcFaceHead] = None
    if use_arcface:
        arcface_head = ArcFaceHead(
            in_dim=out_dim,
            n_classes=n_identities,
            s=arcface_s,
            m=arcface_m,
        ).to(device)

    params = list(model.parameters())
    if arcface_head is not None:
        params += list(arcface_head.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # ------------------------------------------------------------------
    # 5. Baseline (pre-training) validation metrics
    # ------------------------------------------------------------------
    logger.info("Computing baseline (pre-training) validation metrics ...")
    with torch.no_grad():
        # Baseline: use raw (un-projected) embeddings
        raw_train_embs = matrix[train_row_indices].astype(np.float32)
        raw_val_embs = matrix[val_row_indices].astype(np.float32)
    baseline_map, baseline_top1 = retrieval_map_top1(
        raw_val_embs, val_labels, raw_train_embs, train_labels
    )
    logger.info(
        "Baseline (raw embeddings): mAP=%.4f, top1=%.4f",
        baseline_map, baseline_top1,
    )

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    best_val_map = -1.0
    best_val_top1 = 0.0
    best_epoch = -1
    best_state_dict = None
    patience_counter = 0
    curves: List[Dict] = []

    logger.info(
        "Starting training: epochs=%d, P=%d, K=%d, loss=%s, arcface=%s",
        epochs, P_actual, K, loss_mode, use_arcface,
    )

    for epoch in range(epochs):
        # Re-seed sampler per epoch for shuffling identities
        sampler.seed = seed + epoch

        epoch_start = time.time()
        losses = _one_epoch(
            model=model,
            arcface_head=arcface_head,
            dataloader=dataloader,
            optimizer=optimizer,
            loss_mode=loss_mode,
            device=device,
            triplet_margin=triplet_margin,
            supcon_temperature=supcon_temperature,
            arcface_weight=arcface_weight,
            epoch=epoch,
        )

        # Validate
        val_map, val_top1 = _validate(
            model=model,
            matrix=matrix,
            train_row_indices=train_row_indices,
            train_labels=train_labels,
            val_row_indices=val_row_indices,
            val_labels=val_labels,
            device=device,
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start

        curve_entry = {
            "epoch": epoch + 1,
            "loss_total": round(losses["total"], 6),
            "loss_metric": round(losses["metric"], 6),
            "loss_arcface": round(losses["arcface"], 6),
            "val_map": round(val_map, 6),
            "val_top1": round(val_top1, 6),
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_s": round(epoch_time, 2),
        }
        curves.append(curve_entry)

        logger.info(
            "[Epoch %3d/%d] loss=%.4f metric=%.4f | val_mAP=%.4f val_top1=%.4f | %.1fs",
            epoch + 1, epochs,
            losses["total"], losses["metric"],
            val_map, val_top1,
            epoch_time,
        )

        # Best checkpoint tracking
        if val_map > best_val_map:
            best_val_map = val_map
            best_val_top1 = val_top1
            best_epoch = epoch + 1
            best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            logger.info(
                "Early stopping at epoch %d (patience=%d, best_epoch=%d).",
                epoch + 1, early_stop_patience, best_epoch,
            )
            break

    # ------------------------------------------------------------------
    # 7. Save best checkpoint
    # ------------------------------------------------------------------
    if best_state_dict is None:
        best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state_dict)
    ckpt_fingerprint = model.parameter_fingerprint()

    ckpt_path = out_dir / "best_projection.pt"
    torch.save(
        {
            "state_dict": best_state_dict,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "dropout": dropout,
            "hidden_dim": hidden_dim,
            "descriptor": descriptor,
            "seed": seed,
            "best_epoch": best_epoch,
            "val_map": best_val_map,
            "val_top1": best_val_top1,
            "source_fingerprint": source_fp,
            "split_fingerprint": split_fp,
            "model_preprocess_fingerprint": model_fp,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "training_mode": "projection_only",
            "checkpoint_fingerprint": ckpt_fingerprint,
        },
        str(ckpt_path),
    )
    logger.info("Best checkpoint saved to %s (epoch=%d)", ckpt_path, best_epoch)

    # ------------------------------------------------------------------
    # 8. Adoption gate
    # ------------------------------------------------------------------
    gate_result = adoption_gate(
        baseline_map=baseline_map,
        projected_map=best_val_map,
        baseline_top1=baseline_top1,
        projected_top1=best_val_top1,
        min_map_delta=min_map_delta,
        min_top1_delta=min_top1_delta,
        instability_threshold=instability_threshold,
    )
    logger.info("Adoption gate: %s", gate_result.reason)

    # ------------------------------------------------------------------
    # 9. Build and save manifests
    # ------------------------------------------------------------------
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "training_mode": "projection_only",
        "backbone_note": (
            "MiewID backbone is FROZEN. Only a small Linear projection head "
            "is trained. No backbone weights are loaded or modified."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "descriptor": descriptor,
        "artifact_root": str(artifact_root),
        "out_dir": str(out_dir),
        "partition": partition,
        "source_fingerprint": source_fp,
        "split_fingerprint": split_fp,
        "base_model_fingerprint": model_fp,
        "checkpoint_fingerprint": ckpt_fingerprint,
        "seed": seed,
        "hyperparameters": {
            "in_dim": in_dim,
            "out_dim": out_dim,
            "dropout": dropout,
            "hidden_dim": hidden_dim,
            "loss_mode": loss_mode,
            "use_arcface": use_arcface,
            "arcface_s": arcface_s,
            "arcface_m": arcface_m,
            "arcface_weight": arcface_weight,
            "epochs_requested": epochs,
            "P": P_actual,
            "K": K,
            "lr": lr,
            "weight_decay": weight_decay,
            "triplet_margin": triplet_margin,
            "supcon_temperature": supcon_temperature,
            "early_stop_patience": early_stop_patience,
        },
        "training_identities": sorted(identity_to_label.keys()),
        "n_training_identities": n_identities,
        "train_only_identities": inner_split.train_only_identities,
        "n_train_images": len(inner_split.train_image_ids),
        "n_val_images": len(inner_split.val_image_ids),
        "n_val_identities": len(inner_split.val_sessions_by_identity) - len(inner_split.train_only_identities),
        "val_sessions_by_identity": {
            k: v for k, v in inner_split.val_sessions_by_identity.items() if v is not None
        },
        "best_epoch": best_epoch,
        "baseline_map": round(baseline_map, 6),
        "baseline_top1": round(baseline_top1, 6),
        "best_val_map": round(best_val_map, 6),
        "best_val_top1": round(best_val_top1, 6),
        "gate": {
            "adopted": gate_result.adopted,
            "reason": gate_result.reason,
            "map_delta": round(gate_result.map_delta, 6),
            "top1_delta": round(gate_result.top1_delta, 6),
            "min_map_delta": min_map_delta,
            "min_top1_delta": min_top1_delta,
            "instability_threshold": instability_threshold,
        },
        "safety": {
            "forbidden_image_ids_checked": True,
            "forbidden_individual_ids_checked": True,
            "session_disjoint_verified": True,
            "probe_never_used_for_training": True,
            "heldout_never_used_for_training": True,
        },
    }

    manifest_path = out_dir / "training_manifest.json"
    with open(str(manifest_path), "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Training manifest saved to %s", manifest_path)

    curves_path = out_dir / "training_curves.json"
    with open(str(curves_path), "w") as fh:
        json.dump({"curves": curves}, fh, indent=2)
    logger.info("Training curves saved to %s", curves_path)

    diag_path = out_dir / "experiment_diagnostics.json"
    diagnostics = {
        "adopted": gate_result.adopted,
        "gate_reason": gate_result.reason,
        "baseline_map": round(baseline_map, 6),
        "projected_map": round(best_val_map, 6),
        "baseline_top1": round(baseline_top1, 6),
        "projected_top1": round(best_val_top1, 6),
        "best_epoch": best_epoch,
        "epochs_run": len(curves),
        "early_stopped": len(curves) < epochs,
        "train_only_identities": inner_split.train_only_identities,
        "n_train_images": len(inner_split.train_image_ids),
        "n_val_images": len(inner_split.val_image_ids),
        "curves_sample": curves[-5:] if len(curves) >= 5 else curves,
    }
    with open(str(diag_path), "w") as fh:
        json.dump(diagnostics, fh, indent=2)
    logger.info("Experiment diagnostics saved to %s", diag_path)

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train a projection-head metric adaptation for ear_miewid embeddings. "
            "PROJECTION-ONLY: backbone is never loaded or trained."
        )
    )
    p.add_argument(
        "--artifact-root",
        required=True,
        type=Path,
        help="Versioned artifact root (e.g. /path/to/BTEH_reid_artifacts/v1).",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for checkpoint and manifests.",
    )
    p.add_argument("--descriptor", default="ear_miewid", help="Source descriptor name.")
    p.add_argument("--out-dim", type=int, default=512, help="Projection output dimension.")
    p.add_argument("--dropout", type=float, default=0.0, help="Dropout probability.")
    p.add_argument("--hidden-dim", type=int, default=0, help="Hidden dim (0=no hidden layer).")
    p.add_argument(
        "--loss",
        default="both",
        choices=["triplet", "supcon", "both"],
        help="Metric learning loss objective.",
    )
    p.add_argument("--arcface", action="store_true", help="Add ArcFace aux head during training.")
    p.add_argument("--arcface-s", type=float, default=32.0)
    p.add_argument("--arcface-m", type=float, default=0.50)
    p.add_argument("--arcface-weight", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--P", type=int, default=16, help="Identities per batch.")
    p.add_argument("--K", type=int, default=4, help="Samples per identity per batch.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-map-delta", type=float, default=0.005)
    p.add_argument("--min-top1-delta", type=float, default=0.005)
    p.add_argument("--instability-threshold", type=float, default=-0.05)
    p.add_argument("--triplet-margin", type=float, default=0.3)
    p.add_argument("--supcon-temperature", type=float, default=0.07)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--device", default=None, help="'cpu' or 'cuda'.")
    p.add_argument("--partition", default="reference")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        manifest = train_projection(
            artifact_root=args.artifact_root,
            out_dir=args.out_dir,
            descriptor=args.descriptor,
            out_dim=args.out_dim,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim if args.hidden_dim > 0 else None,
            loss_mode=args.loss,
            use_arcface=args.arcface,
            arcface_s=args.arcface_s,
            arcface_m=args.arcface_m,
            arcface_weight=args.arcface_weight,
            epochs=args.epochs,
            P=args.P,
            K=args.K,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            min_map_delta=args.min_map_delta,
            min_top1_delta=args.min_top1_delta,
            instability_threshold=args.instability_threshold,
            triplet_margin=args.triplet_margin,
            supcon_temperature=args.supcon_temperature,
            early_stop_patience=args.early_stop_patience,
            device_str=args.device,
            partition=args.partition,
        )
    except RuntimeError as exc:
        logger.error("HARD FAIL: %s", exc)
        return 2
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        return 1

    adopted = manifest.get("gate", {}).get("adopted", False)
    status = "ADOPTED" if adopted else "REJECTED"
    reason = manifest.get("gate", {}).get("reason", "")
    logger.info("Experiment %s: %s", status, reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
