#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Statistical registration CLI for the fixed-probe evaluation.

MUST be run before any probe scoring.  Writes a content-addressed
``retrospective_registration.json`` that locks the entire evaluation
protocol.  The fixed-probe evaluator verifies the hash on first load
and refuses to proceed if it does not match.

What is registered
------------------
* Frozen split manifest cluster sizes (probe queries per identity) — no
  outcomes, no rankings, no scores.
* OOF ensemble artifact/config hashes (config fingerprint + OOF fingerprint).
* selected-v1 artifact and eval hashes.
* OOF paired-effect variance estimate (from gallery OOF only).
* Exact candidate K, canonical scoring fingerprint, channels, weights,
  calibrator provenance hashes.
* Endpoint formulas (identity-macro MRR primary + 4 secondaries).
* Statistical design: seed 20260714, 10k replicates, alpha 0.05, 80%
  power, operational MDE +0.02, top-1 margin -0.01.
* Multiplicity correction: Holm / max-|T|.
* Conditional uncertainty caveat (underpowered flag from power sim).

Hash contract
-------------
The registration hash is SHA-256 of the canonical JSON body
(sorted keys, no trailing whitespace) EXCLUDING the ``registration_hash``
field itself.  The field is appended as the last key of the document.
Evaluators must re-derive the hash and hard-fail on mismatch.

Usage (write)
-------------
    python pipeline/statistical_registration.py write \\
        --oof-artifacts-dir PATH \\
        --splits-parquet PATH \\
        --selected-v1-eval-hash HASH \\
        --oof-paired-variance FLOAT \\
        [--output PATH]

Usage (verify)
--------------
    python pipeline/statistical_registration.py verify \\
        --registration-file PATH
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.power_simulation import (
    N_ONBOARDING_IDS,
    N_TEMPORAL_IDS,
    OPERATIONAL_DELTA_THRESHOLD,
    SIMULATION_SEED,
    N_REPLICATES,
    ALPHA,
    TARGET_POWER,
    run_power_simulation,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants frozen at registration
# ---------------------------------------------------------------------------

REGISTRATION_FILENAME: str = "retrospective_registration.json"
PROTOCOL_VERSION: str = "retrospective-v1"

# Statistical design (frozen)
FIXED_SEED: int = SIMULATION_SEED                  # 20260714
FIXED_N_REPLICATES: int = N_REPLICATES             # 10_000
FIXED_ALPHA: float = ALPHA                         # 0.05
FIXED_POWER: float = TARGET_POWER                  # 0.80
FIXED_OPERATIONAL_MRR_MDE: float = OPERATIONAL_DELTA_THRESHOLD   # +0.02
FIXED_TOP1_MARGIN: float = -0.01                   # top-1 must not drop by more

# Primary and secondary endpoints
PRIMARY_ENDPOINT: str = "identity_macro_mrr"
SECONDARY_ENDPOINTS: List[str] = [
    "identity_macro_mrr_body_local_only",
    "identity_macro_mrr_ear_local_only",
    "identity_macro_mrr_frozen_ear_isolation",
    "identity_macro_top1_primary",
]
MULTIPLICITY_METHOD: str = "primary_unadjusted_secondaries_holm_max_t"

CONDITIONAL_UNCERTAINTY_CAVEAT: str = (
    "Power simulation is conditional on OOF variance estimate from gallery "
    "data.  If probe-set variance differs materially (>20%) from OOF, "
    "the MDE estimate may be unreliable.  The underpowered flag is set when "
    "MDE > 0.02 and must be disclosed in any candidate/no-candidate report."
)

# Candidate systems registered at evaluation time
CANDIDATE_SYSTEMS: List[str] = [
    "selected_v1",                  # baseline (selected-v1 global channels)
    "selected_v1_plus_body_local",  # selected-v1 + body-local channel
    "selected_v1_plus_ear_local",   # selected-v1 + ear-local channel
    "selected_v1_plus_both_local",  # primary: selected-v1 + body+ear local
    "selected_v1_frozen_ear",       # primary with frozen ear (fine-tune isolation)
    "local_only_exploratory",       # exploratory: local channels only (if available)
    "megadesc_frozen_ear_exploratory",  # exploratory: MegaDescriptor + frozen ear
]
CONFIRMATORY_SYSTEMS: List[str] = [
    "selected_v1",
    "selected_v1_plus_both_local",  # primary contrast challenger
    "selected_v1_plus_body_local",  # secondary contrast 1
    "selected_v1_plus_ear_local",   # secondary contrast 2
    "selected_v1_frozen_ear",       # secondary contrast 3
]


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

class RegistrationHashMismatchError(RuntimeError):
    """Raised when the registration hash does not match the document body."""


class RegistrationOutcomePollutionError(RuntimeError):
    """Raised when outcome columns are detected in registration inputs."""


class RegistrationAlreadyConsumedError(RuntimeError):
    """Raised when probe rows have already been loaded before registration."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical_json(obj: Any) -> str:
    """Compact canonical JSON (sorted keys, no trailing whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_json_file(path: Path) -> str:
    """SHA-256 of the canonical JSON at *path*."""
    with open(path) as fh:
        obj = json.load(fh)
    return _sha256_hex(_canonical_json(obj))


def _hash_file_bytes(path: Path) -> str:
    """SHA-256 of exact file bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Cluster-size extraction (no outcomes)
# ---------------------------------------------------------------------------

#: Columns that must NOT be present when reading cluster sizes (outcome guard)
_OUTCOME_COLUMNS: frozenset = frozenset({
    "fused_score", "rank", "candidate_individual_id",
    "top1_correct", "top5_correct", "reciprocal_rank",
    "rr", "mrr", "ap", "hit",
})

#: Split values for each stratum
_TEMPORAL_SPLITS: frozenset = frozenset({"probe"})
_ONBOARDING_SPLITS: frozenset = frozenset({"held_out_probe"})


def extract_cluster_sizes(
    splits_parquet: Path,
) -> tuple[Dict[str, int], Dict[str, int]]:
    """
    Extract identity cluster sizes from the frozen split manifest.

    Reads ONLY: image_id, individual_id, split.  Refuses to proceed if
    any outcome column is present.  Returns cluster sizes as dicts
    mapping individual_id → n_probe_images.

    Returns
    -------
    temporal_cluster_sizes, onboarding_cluster_sizes
    """
    splits_df = pd.read_parquet(str(splits_parquet))

    # Outcome-pollution guard
    bad_cols = set(splits_df.columns) & _OUTCOME_COLUMNS
    if bad_cols:
        raise RegistrationOutcomePollutionError(
            f"Outcome columns detected in split manifest: {sorted(bad_cols)}. "
            "Registration must use the split manifest only, not an eval output."
        )

    required = {"image_id", "individual_id", "split"}
    missing = required - set(splits_df.columns)
    if missing:
        raise ValueError(f"Split manifest missing required columns: {missing}")

    temporal_df = splits_df[splits_df["split"].isin(_TEMPORAL_SPLITS)].copy()
    onboarding_df = splits_df[splits_df["split"].isin(_ONBOARDING_SPLITS)].copy()

    temporal_cs: Dict[str, int] = (
        temporal_df.groupby("individual_id")["image_id"]
        .nunique()
        .astype(int)
        .to_dict()
    )
    onboarding_cs: Dict[str, int] = (
        onboarding_df.groupby("individual_id")["image_id"]
        .nunique()
        .astype(int)
        .to_dict()
    )
    return temporal_cs, onboarding_cs


# ---------------------------------------------------------------------------
# OOF artifact hash extraction
# ---------------------------------------------------------------------------

def _hash_oof_artifacts(artifacts_dir: Path) -> str:
    """
    Compute a single hash covering config.json + fingerprint.json from the
    OOF artifacts directory.  These are the only files read; no OOF scores
    or calibrator binaries are included.
    """
    config_path = artifacts_dir / "config.json"
    fingerprint_path = artifacts_dir / "fingerprint.json"
    for p in (config_path, fingerprint_path):
        if not p.exists():
            raise FileNotFoundError(
                f"OOF artifact missing for registration: {p}"
            )
    combined = _canonical_json({
        "config": json.loads(config_path.read_text()),
        "fingerprint": json.loads(fingerprint_path.read_text()),
    })
    return _sha256_hex(combined)


def _read_oof_config_fields(artifacts_dir: Path) -> dict:
    """Read frozen_k, channels, and fusion weights from OOF config/metrics."""
    config_path = artifacts_dir / "config.json"
    metrics_path = artifacts_dir / "oof_metrics.json"
    weights_path = artifacts_dir / "fusion_weights.json"

    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    weights = json.loads(weights_path.read_text()) if weights_path.exists() else {}

    return {
        "frozen_k": metrics.get("frozen_k", config.get("k_default", 50)),
        "all_channels": config.get("all_channels", []),
        "fusion_weights": weights,
    }


def _read_oof_fingerprint(artifacts_dir: Path) -> str:
    """Read the pipeline fingerprint from OOF artifacts."""
    fp_path = artifacts_dir / "fingerprint.json"
    if fp_path.exists():
        fp = json.loads(fp_path.read_text())
        return fp.get("config_fingerprint", "")
    return ""


# ---------------------------------------------------------------------------
# Registration document construction
# ---------------------------------------------------------------------------

def build_registration_document(
    temporal_cluster_sizes: Dict[str, int],
    onboarding_cluster_sizes: Dict[str, int],
    oof_artifacts_hash: str,
    oof_scoring_fingerprint: str,
    selected_v1_eval_hash: str,
    oof_paired_variance: float,
    frozen_k: int,
    all_channels: List[str],
    fusion_weights: Dict[str, float],
    power_sim_result: Optional[dict] = None,
    splits_manifest_hash: str = "unspecified",
) -> dict:
    """
    Build the canonical registration document (WITHOUT the hash field).

    Returns the document dict; the caller appends registration_hash.
    """
    doc: dict = {
        "protocol_version": PROTOCOL_VERSION,
        "registered_at": datetime.now(tz=timezone.utc).isoformat(),
        # --- Cluster sizes (no outcomes) ---
        "cluster_sizes": {
            "temporal": {k: int(v) for k, v in sorted(temporal_cluster_sizes.items())},
            "onboarding": {k: int(v) for k, v in sorted(onboarding_cluster_sizes.items())},
        },
        "n_temporal_ids": len(temporal_cluster_sizes),
        "n_onboarding_ids": len(onboarding_cluster_sizes),
        # --- Artifact provenance ---
        "oof_artifacts_hash": oof_artifacts_hash,
        "oof_scoring_fingerprint": oof_scoring_fingerprint,
        "selected_v1_eval_hash": selected_v1_eval_hash,
        "splits_manifest_hash": splits_manifest_hash,
        # --- Locked evaluation configuration ---
        "frozen_k": frozen_k,
        "all_channels": list(all_channels),
        "fusion_weights": {k: float(v) for k, v in sorted(fusion_weights.items())},
        # --- Endpoint formulas ---
        "endpoint_formulas": {
            "reciprocal_rank": (
                "RR = 1/rank(truth_id) if truth_id in ranked_ids else 0; "
                "truth identity counted once."
            ),
            "identity_macro_mrr": (
                "For each identity: mean(RR over its queries); "
                "then equal-weight mean across all identities. Primary endpoint."
            ),
            "query_weighted_mrr": (
                "mean(RR per query) across all known-identity queries. "
                "Reproduces selected-v1 known_mAP=0.473 within 1e-3 tolerance."
            ),
            "identity_macro_top1": (
                "Per-identity mean top-1 hit rate; equal-weight mean across identities."
            ),
            "identity_macro_top5": (
                "Per-identity mean top-5 hit rate; equal-weight mean across identities."
            ),
            "query_weighted_top1": "mean(top-1 hit per query).",
            "query_weighted_top5": "mean(top-5 hit per query).",
        },
        # --- Primary and secondary endpoints ---
        "primary_endpoint": PRIMARY_ENDPOINT,
        "secondary_endpoints": list(SECONDARY_ENDPOINTS),
        "candidate_systems": list(CANDIDATE_SYSTEMS),
        "confirmatory_systems": list(CONFIRMATORY_SYSTEMS),
        # --- Hypothesis ---
        "hypothesis": {
            "primary_delta_threshold": float(FIXED_OPERATIONAL_MRR_MDE),
            "top1_delta_lower_bound": float(FIXED_TOP1_MARGIN),
            "description": (
                "Primary: identity-macro MRR improvement >= +0.02. "
                "Top-1 must not drop by more than 0.01 (margin guard). "
                "CI lower bound > 0 for candidate promotion."
            ),
        },
        # --- Statistical design ---
        "statistical_design": {
            "seed": FIXED_SEED,
            "n_replicates": FIXED_N_REPLICATES,
            "alpha": FIXED_ALPHA,
            "target_power": FIXED_POWER,
            "operational_mde": FIXED_OPERATIONAL_MRR_MDE,
            "test_family": "paired_protocol_stratified_cluster_bootstrap",
            "strata": ["temporal", "onboarding"],
            "sign_flip_null": True,
            "gallery_fixed": True,
            "cluster_unit": "identity",
        },
        "multiplicity_correction": MULTIPLICITY_METHOD,
        # --- Power simulation results (if available) ---
        "power_simulation": power_sim_result if power_sim_result is not None else {},
        # --- Variance plugin ---
        "oof_paired_variance": float(oof_paired_variance),
        # --- Evaluation guard ---
        "evaluation_guards": {
            "no_head_channel": True,
            "no_candidate_truth_forcing": True,
            "probe_consumed_flag": False,  # set to True after first load
            "head_gate_status": "failed_not_applicable",
            "covariate_shift_threshold": 0.05,
        },
        # --- Conditional uncertainty caveat ---
        "conditional_uncertainty_caveat": CONDITIONAL_UNCERTAINTY_CAVEAT,
    }
    return doc


def compute_registration_hash(doc: dict) -> str:
    """
    Compute SHA-256 of the canonical registration document body.

    The ``registration_hash`` field is excluded before hashing.
    """
    body = {k: v for k, v in doc.items() if k != "registration_hash"}
    return _sha256_hex(_canonical_json(body))


# ---------------------------------------------------------------------------
# Write / verify API
# ---------------------------------------------------------------------------

def write_registration(
    output_path: Path,
    temporal_cluster_sizes: Dict[str, int],
    onboarding_cluster_sizes: Dict[str, int],
    oof_artifacts_dir: Path,
    selected_v1_eval_hash: str,
    oof_paired_variance: float,
    *,
    run_power: bool = True,
    overwrite: bool = False,
    splits_parquet: Optional[Path] = None,
) -> str:
    """
    Build and write the registration document.

    Returns the registration hash (hex string).
    """
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Registration file already exists: {output_path}. "
            "Pass overwrite=True to replace."
        )

    oof_artifacts_hash = _hash_oof_artifacts(oof_artifacts_dir)
    oof_scoring_fp = _read_oof_fingerprint(oof_artifacts_dir)
    oof_fields = _read_oof_config_fields(oof_artifacts_dir)
    splits_manifest_hash = (
        _hash_file_bytes(splits_parquet)
        if splits_parquet is not None
        else _sha256_hex(
            _canonical_json(
                {
                    "temporal": temporal_cluster_sizes,
                    "onboarding": onboarding_cluster_sizes,
                }
            )
        )
    )

    power_result: Optional[dict] = None
    if run_power:
        try:
            sim = run_power_simulation(
                temporal_cluster_sizes,
                onboarding_cluster_sizes,
                oof_paired_variance,
            )
            power_result = sim.to_dict()
            if sim.underpowered:
                logger.warning(
                    "UNDERPOWERED: MDE=%.4f > %.2f. Caveat will be recorded.",
                    sim.mde_80,
                    FIXED_OPERATIONAL_MRR_MDE,
                )
        except Exception as exc:
            logger.warning("Power simulation failed: %s. Continuing without.", exc)

    doc = build_registration_document(
        temporal_cluster_sizes=temporal_cluster_sizes,
        onboarding_cluster_sizes=onboarding_cluster_sizes,
        oof_artifacts_hash=oof_artifacts_hash,
        oof_scoring_fingerprint=oof_scoring_fp,
        selected_v1_eval_hash=selected_v1_eval_hash,
        oof_paired_variance=oof_paired_variance,
        frozen_k=oof_fields["frozen_k"],
        all_channels=oof_fields["all_channels"],
        fusion_weights=oof_fields["fusion_weights"],
        power_sim_result=power_result,
        splits_manifest_hash=splits_manifest_hash,
    )

    reg_hash = compute_registration_hash(doc)
    doc["registration_hash"] = reg_hash

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)

    logger.info("Registration written: %s (hash=%s)", output_path, reg_hash[:16])
    return reg_hash


def load_and_verify_registration(registration_path: Path) -> dict:
    """
    Load and verify the registration document.

    Raises RegistrationHashMismatchError if the stored hash does not
    match the recomputed hash.

    Returns the verified registration document.
    """
    with open(registration_path) as fh:
        doc = json.load(fh)

    stored_hash = doc.get("registration_hash", "")
    if not stored_hash:
        raise RegistrationHashMismatchError(
            f"Registration file has no registration_hash field: {registration_path}"
        )

    expected = compute_registration_hash(doc)
    if stored_hash != expected:
        raise RegistrationHashMismatchError(
            f"Registration hash mismatch for {registration_path}. "
            f"Stored={stored_hash[:16]}..., Expected={expected[:16]}... "
            "The registration file may have been modified after writing."
        )

    return doc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Statistical registration for fixed-probe evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # write
    wr = sub.add_parser("write", help="Write the registration document.")
    wr.add_argument("--oof-artifacts-dir", required=True,
                    help="Path to OOF calibration output directory.")
    wr.add_argument("--splits-parquet", required=True,
                    help="Path to the frozen bteh_splits.parquet manifest.")
    wr.add_argument("--selected-v1-eval-hash", required=True,
                    help="SHA-256 hex of the selected-v1 eval summary JSON.")
    wr.add_argument("--oof-paired-variance", type=float, required=True,
                    help="Estimated paired-effect variance from gallery OOF data.")
    wr.add_argument("--output", default=REGISTRATION_FILENAME,
                    help="Output path for the registration JSON.")
    wr.add_argument("--no-power", action="store_true",
                    help="Skip power simulation (faster, for testing).")
    wr.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing registration file.")

    # verify
    vf = sub.add_parser("verify", help="Verify an existing registration file.")
    vf.add_argument("--registration-file", required=True,
                    help="Path to the registration JSON to verify.")

    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "write":
        oof_dir = Path(args.oof_artifacts_dir)
        splits_path = Path(args.splits_parquet)
        output_path = Path(args.output)

        temporal_cs, onboarding_cs = extract_cluster_sizes(splits_path)
        logger.info(
            "Cluster sizes: %d temporal IDs (%d queries), "
            "%d onboarding IDs (%d queries)",
            len(temporal_cs), sum(temporal_cs.values()),
            len(onboarding_cs), sum(onboarding_cs.values()),
        )

        reg_hash = write_registration(
            output_path=output_path,
            temporal_cluster_sizes=temporal_cs,
            onboarding_cluster_sizes=onboarding_cs,
            oof_artifacts_dir=oof_dir,
            selected_v1_eval_hash=args.selected_v1_eval_hash,
            oof_paired_variance=args.oof_paired_variance,
            run_power=not args.no_power,
            overwrite=args.overwrite,
            splits_parquet=splits_path,
        )
        print(json.dumps({"status": "ok", "registration_hash": reg_hash,
                          "output": str(output_path)}))
        return 0

    if args.command == "verify":
        reg_path = Path(args.registration_file)
        try:
            doc = load_and_verify_registration(reg_path)
        except RegistrationHashMismatchError as exc:
            logger.error("HASH MISMATCH: %s", exc)
            return 1
        print(json.dumps({
            "status": "verified",
            "registration_hash": doc["registration_hash"],
            "protocol_version": doc.get("protocol_version"),
            "n_temporal_ids": doc.get("n_temporal_ids"),
            "n_onboarding_ids": doc.get("n_onboarding_ids"),
        }))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
