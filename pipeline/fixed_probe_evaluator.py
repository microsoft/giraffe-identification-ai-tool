#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Fixed-probe evaluator for the paired statistical evaluation.

Design contracts
----------------
* Verifies the registration hash before loading any probe rows.
  Hard-fails on hash mismatch (RegistrationHashMismatchError).
* Verifies splits_manifest_hash (exact file bytes) BEFORE reading any
  probe rows or outcomes — a stale splits file is a hard error.
* Loads probe rows ONCE (first time); sets probe_consumed_flag = True in
  the registration document copy.  Subsequent calls with the same probe
  data raise RegistrationAlreadyConsumedError if probes have already been
  scored (enforced externally via the consumed marker file).
* Uses the exact EnsembleScorer from OOF artifacts (frozen calibrators,
  fusion weights, and local scorers).  No new scoring is performed here
  for real data — this module is the gate that enforces the contract.
* Temporal gallery (split=="gallery") and onboarding combined gallery
  (split=="gallery" ∪ "held_out_gallery") semantics match prior
  normalized eval (step_4c_normalized_eval).
* No head channel.  No candidate truth forcing.
* Writes query-level paired rankings as a Parquet file with system name,
  probe type, and scoring fingerprint columns.
* Output is written atomically (temp-then-rename); consumed marker is
  written only after the rankings file is durably persisted.

Candidate systems scored
------------------------
  selected_v1                   – selected-v1 global channels only (baseline)
  selected_v1_plus_body_local   – selected-v1 + body-local channel
  selected_v1_plus_ear_local    – selected-v1 + ear-local channel
  selected_v1_plus_both_local   – primary: selected-v1 + body+ear local
  selected_v1_frozen_ear        – primary variant with frozen ear for isolation
  local_only_exploratory        – local channels only (if both channels available)
  megadesc_frozen_ear_exploratory – MegaDescriptor + frozen ear (if data available)

Gallery semantics
-----------------
  Temporal probes  → temporal gallery (split=="gallery" only)
  Onboarding probes → combined gallery (gallery ∪ held_out_gallery)
  Probe images are excluded from all gallery sets.
  (Matches step_4c_normalized_eval.run_normalized_eval().)

CLI subcommands
---------------
  score        – Full real-data scoring: verifies all hashes, loads data,
                 instantiates scorers, scores all registered systems, writes
                 probe_rankings.parquet atomically, writes consumed marker.
  load         – Load and verify an existing probe_rankings.parquet.
  postprocess  – Run paired bootstrap + candidate report after scoring.
                 Does NOT execute probes; reads existing rankings only.
                 Writes metrics/report/future_protocol to output_dir.
                 Does NOT mutate production artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.statistical_registration import (
    RegistrationHashMismatchError,
    RegistrationAlreadyConsumedError,
    load_and_verify_registration,
)
from pipeline.eval_metrics import (
    reciprocal_rank,
    top_k_hit,
    compute_all_metrics,
    verify_legacy_map_regression,
)
from pipeline.local_oof_calibration import (
    ALL_CHANNELS,
    CHANNEL_BODY_LOCAL,
    CHANNEL_EAR,
    CHANNEL_EAR_LOCAL,
    CHANNEL_MIEWID,
    GLOBAL_CHANNELS,
    ProbePollutionError,
    _assert_no_probe_ids,
    _filter_gallery_only,
)
from pipeline.ensemble_inference import (
    EnsembleArtifacts,
    EnsembleScorer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROBE_RANKINGS_PARQUET: str = "probe_rankings.parquet"
CONSUMED_MARKER_FILENAME: str = ".probes_consumed"

# Column names in the output parquet
COL_QUERY_IMAGE_ID: str = "query_image_id"
COL_TRUTH_INDIVIDUAL_ID: str = "truth_individual_id"
COL_PROBE_TYPE: str = "probe_type"             # "temporal" or "onboarding"
COL_RANK: str = "rank"
COL_CANDIDATE_ID: str = "candidate_individual_id"
COL_FUSED_SCORE: str = "fused_score"
COL_SYSTEM_NAME: str = "system_name"
COL_CHANNELS_AVAILABLE: str = "channels_available"
COL_SCORING_FINGERPRINT: str = "scoring_fingerprint"
COL_REGISTRATION_HASH: str = "registration_hash"

# Probe split markers
_TEMPORAL_SPLITS: frozenset = frozenset({"probe"})
_ONBOARDING_SPLITS: frozenset = frozenset({"held_out_probe"})
_GALLERY_SPLITS: frozenset = frozenset({"gallery"})
_HELD_OUT_GALLERY_SPLITS: frozenset = frozenset({"held_out_gallery"})


# ---------------------------------------------------------------------------
# Candidate system specifications
# ---------------------------------------------------------------------------

@dataclass
class SystemSpec:
    """
    Specification for one candidate system to be scored.

    channels_override : if not None, use only these channels (others masked).
    exploratory       : if True, system is exploratory; skip if data unavailable.
    use_frozen_ear    : if True, use frozen (non-projected) ear embeddings instead
                        of projected ones (for fine-tune isolation).
    """
    name: str
    channels_override: Optional[List[str]] = None
    exploratory: bool = False
    use_frozen_ear: bool = False
    weights_override: Optional[Dict[str, float]] = None
    description: str = ""


# Registered candidate systems
_SYSTEM_SPECS: List[SystemSpec] = [
    SystemSpec(
        name="selected_v1",
        channels_override=list(GLOBAL_CHANNELS),
        weights_override={"miewid": 0.6, "ear_miewid_projected": 0.4},
        description="selected-v1 global channels only (baseline)",
    ),
    SystemSpec(
        name="selected_v1_plus_body_local",
        channels_override=[CHANNEL_MIEWID, CHANNEL_EAR, CHANNEL_BODY_LOCAL],
        description="selected-v1 + body-local channel",
    ),
    SystemSpec(
        name="selected_v1_plus_ear_local",
        channels_override=[CHANNEL_MIEWID, CHANNEL_EAR, CHANNEL_EAR_LOCAL],
        description="selected-v1 + ear-local channel",
    ),
    SystemSpec(
        name="selected_v1_plus_both_local",
        channels_override=list(ALL_CHANNELS),
        description="primary: selected-v1 + body+ear local channels",
    ),
    SystemSpec(
        name="selected_v1_frozen_ear",
        channels_override=list(ALL_CHANNELS),
        use_frozen_ear=True,
        description="primary with frozen (non-projected) ear — fine-tune isolation",
    ),
    SystemSpec(
        name="local_only_exploratory",
        channels_override=[CHANNEL_BODY_LOCAL, CHANNEL_EAR_LOCAL],
        exploratory=True,
        description="local channels only (exploratory; skipped if unavailable)",
    ),
    SystemSpec(
        name="megadesc_frozen_ear_exploratory",
        channels_override=["megadescriptor", "ear_megadescriptor"],
        exploratory=True,
        description="MegaDescriptor + frozen ear (exploratory; skipped if unavailable)",
    ),
]

SYSTEM_SPECS: Dict[str, SystemSpec] = {s.name: s for s in _SYSTEM_SPECS}


# ---------------------------------------------------------------------------
# Scoring fingerprint helpers
# ---------------------------------------------------------------------------

def _scoring_fingerprint(
    system_name: str,
    registration_hash: str,
    channels: List[str],
    fusion_weights: Dict[str, float],
    frozen_k: int,
) -> str:
    """Content-addressed fingerprint for a single system's scoring configuration."""
    payload = {
        "system": system_name,
        "registration_hash": registration_hash,
        "channels": sorted(channels),
        "weights": {k: round(float(v), 8) for k, v in sorted(fusion_weights.items())},
        "frozen_k": frozen_k,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Query record helpers (no real scoring)
# ---------------------------------------------------------------------------

@dataclass
class QueryRankRecord:
    """Single query result for one system."""
    query_image_id: str
    truth_individual_id: Optional[str]
    probe_type: str                         # "temporal" or "onboarding"
    system_name: str
    ranked_ids: List[str]                   # ordered by score, best first
    fused_scores: List[float]               # corresponding scores
    channels_available: List[str]
    scoring_fingerprint: str
    registration_hash: str


def _records_to_dataframe(records: List[QueryRankRecord]) -> pd.DataFrame:
    """Convert a list of QueryRankRecord objects to a per-rank DataFrame."""
    rows = []
    for rec in records:
        for rank, (cid, score) in enumerate(
            zip(rec.ranked_ids, rec.fused_scores), start=1
        ):
            rows.append({
                COL_QUERY_IMAGE_ID: rec.query_image_id,
                COL_TRUTH_INDIVIDUAL_ID: rec.truth_individual_id,
                COL_PROBE_TYPE: rec.probe_type,
                COL_SYSTEM_NAME: rec.system_name,
                COL_RANK: rank,
                COL_CANDIDATE_ID: cid,
                COL_FUSED_SCORE: float(score),
                COL_CHANNELS_AVAILABLE: ",".join(rec.channels_available),
                COL_SCORING_FINGERPRINT: rec.scoring_fingerprint,
                COL_REGISTRATION_HASH: rec.registration_hash,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Gallery splitting helpers
# ---------------------------------------------------------------------------

def _build_gallery_splits(
    splits_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the manifest into temporal_probe, onboarding_probe,
    gallery, and held_out_gallery DataFrames.

    Probe images are excluded from gallery sets (matching normalized eval).
    """
    def _load(split_vals):
        mask = splits_df["split"].isin(split_vals)
        df = splits_df[mask][["image_id", "individual_id", "session_id"]].copy()
        df = df.drop_duplicates(subset="image_id").reset_index(drop=True)
        df["image_id"] = df["image_id"].astype(str)
        df["individual_id"] = df["individual_id"].astype(str)
        df["session_id"] = df.get("session_id", pd.Series(dtype=str)).fillna("unknown").astype(str)
        return df

    temporal_probe_df = _load(_TEMPORAL_SPLITS)
    onboarding_probe_df = _load(_ONBOARDING_SPLITS)
    gallery_df = _load(_GALLERY_SPLITS)
    held_out_gallery_df = _load(_HELD_OUT_GALLERY_SPLITS)

    # Exclude probe images from galleries (safety)
    all_probe_ids = set(temporal_probe_df["image_id"]) | set(onboarding_probe_df["image_id"])
    gallery_df = gallery_df[~gallery_df["image_id"].isin(all_probe_ids)].reset_index(drop=True)
    held_out_gallery_df = held_out_gallery_df[
        ~held_out_gallery_df["image_id"].isin(all_probe_ids)
    ].reset_index(drop=True)

    # Temporal gallery = gallery only
    # Onboarding gallery = gallery ∪ held_out_gallery
    return temporal_probe_df, onboarding_probe_df, gallery_df, held_out_gallery_df


def build_onboarding_combined_gallery(
    gallery_df: pd.DataFrame,
    held_out_gallery_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combined gallery for onboarding probes (gallery ∪ held_out_gallery)."""
    return pd.concat([gallery_df, held_out_gallery_df], ignore_index=True)


# ---------------------------------------------------------------------------
# System-specific scorer builder
# ---------------------------------------------------------------------------

def build_system_scorer(
    spec: SystemSpec,
    artifacts: EnsembleArtifacts,
    gallery_df: pd.DataFrame,
    crop_df: pd.DataFrame,
    embedding_matrices: Dict[str, np.ndarray],
    descriptor_mappings: Dict[str, pd.DataFrame],
    local_scorer_body,
    local_scorer_ear,
) -> Optional[EnsembleScorer]:
    """
    Build an EnsembleScorer for *spec* by masking unavailable channels.

    Returns None if the system is exploratory and required channels are absent.
    """
    from copy import deepcopy

    requested_channels = spec.channels_override or list(artifacts.all_channels)
    missing_channels = []
    for channel in requested_channels:
        if channel == CHANNEL_BODY_LOCAL:
            available = local_scorer_body is not None
        elif channel == CHANNEL_EAR_LOCAL:
            available = local_scorer_ear is not None
        else:
            available = channel in embedding_matrices
        if not available:
            missing_channels.append(channel)

    if missing_channels:
        if spec.exploratory:
            logger.info(
                "Exploratory system '%s' skipped: missing channels %s.",
                spec.name,
                missing_channels,
            )
            return None
        raise ValueError(
            f"System {spec.name!r} is missing required channels: "
            f"{missing_channels}"
        )

    # Build overridden artifacts (masked channels + weights)
    override_channels = requested_channels
    source_weights = spec.weights_override or artifacts.fusion_weights
    masked_weights = {
        ch: float(source_weights.get(ch, 0.0))
        for ch in override_channels
    }
    total_w = sum(masked_weights.values())
    if total_w > 0:
        masked_weights = {ch: w / total_w for ch, w in masked_weights.items()}

    # Body/ear local scorers: only include if the channel is requested
    use_body = CHANNEL_BODY_LOCAL in override_channels
    use_ear = CHANNEL_EAR_LOCAL in override_channels

    patched_artifacts = EnsembleArtifacts(
        frozen_k=artifacts.frozen_k,
        fusion_weights=masked_weights,
        calibrators_global=artifacts.calibrators_global,
        calibrator_body=artifacts.calibrator_body if use_body else None,
        calibrator_ear=artifacts.calibrator_ear if use_ear else None,
        oof_metrics=artifacts.oof_metrics,
        config=artifacts.config,
        fingerprint=artifacts.fingerprint,
        all_channels=override_channels,
    )

    # For frozen ear: replace projected ear embeddings with frozen ear
    patched_emb = dict(embedding_matrices)
    patched_desc = dict(descriptor_mappings)
    if spec.use_frozen_ear and "ear_miewid" in embedding_matrices:
        patched_emb[CHANNEL_EAR] = embedding_matrices["ear_miewid"]
        if "ear_miewid" in descriptor_mappings:
            patched_desc[CHANNEL_EAR] = descriptor_mappings["ear_miewid"]

    return EnsembleScorer(
        artifacts=patched_artifacts,
        gallery_df=gallery_df,
        crop_df=crop_df,
        embedding_matrices=patched_emb,
        descriptor_mappings=patched_desc,
        local_scorer_body=local_scorer_body if use_body else None,
        local_scorer_ear=local_scorer_ear if use_ear else None,
    )


# ---------------------------------------------------------------------------
# Core: probe scoring stub (contracts enforced, real scoring delegated)
# ---------------------------------------------------------------------------

def score_probe_with_scorer(
    query_image_id: str,
    query_session_id: str,
    truth_individual_id: Optional[str],
    probe_type: str,
    scorer: EnsembleScorer,
    system_name: str,
    registration_hash: str,
    frozen_k: int,
    fusion_weights: Dict[str, float],
    *,
    shortlist: Optional[List[str]] = None,
) -> QueryRankRecord:
    """
    Score one probe query using *scorer*.

    No head channel.  No candidate truth forcing.
    If *shortlist* is provided, only those candidates are scored;
    otherwise all gallery identities are scored (K not re-applied here).
    """
    results = scorer.score(
        query_image_id=query_image_id,
        query_session_id=query_session_id,
        candidate_ids=shortlist,
        query_individual_id=None,   # truth NOT passed to scorer; no forcing
        source_fingerprint=registration_hash,
        split_fingerprint=registration_hash,
    )

    ranked_ids = [r.individual_id for r in results]
    fused_scores = [r.fused_score for r in results]
    configured_channels = list(scorer.artifacts.all_channels)
    channels_available = sorted(
        {
            channel
            for result in results
            for channel in result.channels_available
        }
    )

    fp = _scoring_fingerprint(
        system_name=system_name,
        registration_hash=registration_hash,
        channels=configured_channels,
        fusion_weights=fusion_weights,
        frozen_k=frozen_k,
    )

    return QueryRankRecord(
        query_image_id=query_image_id,
        truth_individual_id=truth_individual_id,
        probe_type=probe_type,
        system_name=system_name,
        ranked_ids=ranked_ids,
        fused_scores=fused_scores,
        channels_available=channels_available,
        scoring_fingerprint=fp,
        registration_hash=registration_hash,
    )


# ---------------------------------------------------------------------------
# Fixed-probe evaluator (the top-level orchestration entry point)
# ---------------------------------------------------------------------------

class FixedProbeEvaluator:
    """
    Orchestrates fixed-probe evaluation against a verified registration.

    Parameters
    ----------
    registration_path : path to retrospective_registration.json.
    output_dir        : directory where probe_rankings.parquet is written.
    """

    def __init__(
        self,
        registration_path: Path,
        output_dir: Path,
    ):
        # Hard-fail on hash mismatch
        self.registration = load_and_verify_registration(registration_path)
        self.registration_path = registration_path
        self.registration_hash: str = self.registration["registration_hash"]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._consumed_marker = self.output_dir / CONSUMED_MARKER_FILENAME
        self._probe_records: List[QueryRankRecord] = []

    @property
    def is_consumed(self) -> bool:
        """True if probes have already been loaded/scored."""
        return self._consumed_marker.exists()

    def _assert_not_consumed(self) -> None:
        if self.is_consumed:
            raise RegistrationAlreadyConsumedError(
                f"Probes have already been consumed for registration "
                f"{self.registration_hash[:16]}. "
                "Fixed-probe evaluation may only be run once per registration."
            )

    def score_probes(
        self,
        splits_df: pd.DataFrame,
        artifacts: EnsembleArtifacts,
        crop_df: pd.DataFrame,
        embedding_matrices: Dict[str, np.ndarray],
        descriptor_mappings: Dict[str, pd.DataFrame],
        local_scorer_body=None,
        local_scorer_ear=None,
        systems: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Score all probe images with the requested systems.

        Gallery semantics:
          Temporal probes → temporal gallery (split=="gallery")
          Onboarding probes → combined gallery (gallery ∪ held_out_gallery)

        No head channel.  No candidate truth forcing.

        Parameters
        ----------
        splits_df      : full split manifest with 'split', 'image_id',
                         'individual_id', 'session_id' columns.
        artifacts      : frozen EnsembleArtifacts (OOF output, read-only).
        crop_df        : crop manifest.
        embedding_matrices : {channel: np.ndarray}.
        descriptor_mappings : {channel: DataFrame}.
        local_scorer_body, local_scorer_ear : canonical local scorers.
        systems        : list of system names to score (defaults to all
                         non-exploratory systems).

        Returns
        -------
        DataFrame of query-level paired rankings.
        """
        self._assert_not_consumed()

        (
            temporal_probe_df,
            onboarding_probe_df,
            gallery_df,
            held_out_gallery_df,
        ) = _build_gallery_splits(splits_df)

        if temporal_probe_df.empty and onboarding_probe_df.empty:
            raise ValueError("No probe rows found in splits_df.")

        combined_gallery_df = build_onboarding_combined_gallery(
            gallery_df, held_out_gallery_df
        )

        systems_to_score = systems or [
            s.name for s in _SYSTEM_SPECS if not s.exploratory
        ]

        all_records: List[QueryRankRecord] = []

        for sys_name in systems_to_score:
            spec = SYSTEM_SPECS.get(sys_name)
            if spec is None:
                logger.warning("Unknown system '%s'; skipping.", sys_name)
                continue

            logger.info("Building scorer for system '%s' ...", sys_name)

            # Build temporal scorer (gallery only)
            temporal_scorer = build_system_scorer(
                spec, artifacts, gallery_df, crop_df,
                embedding_matrices, descriptor_mappings,
                local_scorer_body, local_scorer_ear,
            )
            # Build onboarding scorer (combined gallery)
            onboarding_scorer = build_system_scorer(
                spec, artifacts, combined_gallery_df, crop_df,
                embedding_matrices, descriptor_mappings,
                local_scorer_body, local_scorer_ear,
            )

            if temporal_scorer is None and onboarding_scorer is None:
                logger.info("System '%s' skipped (no scorers available).", sys_name)
                continue

            logger.info(
                "Scoring %d temporal + %d onboarding probes for system '%s' ...",
                len(temporal_probe_df), len(onboarding_probe_df), sys_name,
            )

            # --- Temporal probes ---
            if temporal_scorer is not None:
                for _, row in temporal_probe_df.iterrows():
                    rec = score_probe_with_scorer(
                        query_image_id=str(row["image_id"]),
                        query_session_id=str(row.get("session_id", "unknown")),
                        truth_individual_id=str(row["individual_id"])
                            if pd.notna(row.get("individual_id")) else None,
                        probe_type="temporal",
                        scorer=temporal_scorer,
                        system_name=sys_name,
                        registration_hash=self.registration_hash,
                        frozen_k=artifacts.frozen_k,
                        fusion_weights=dict(temporal_scorer.fusion_weights),
                    )
                    all_records.append(rec)

            # --- Onboarding probes ---
            if onboarding_scorer is not None:
                for _, row in onboarding_probe_df.iterrows():
                    rec = score_probe_with_scorer(
                        query_image_id=str(row["image_id"]),
                        query_session_id=str(row.get("session_id", "unknown")),
                        truth_individual_id=str(row["individual_id"])
                            if pd.notna(row.get("individual_id")) else None,
                        probe_type="onboarding",
                        scorer=onboarding_scorer,
                        system_name=sys_name,
                        registration_hash=self.registration_hash,
                        frozen_k=artifacts.frozen_k,
                        fusion_weights=dict(onboarding_scorer.fusion_weights),
                    )
                    all_records.append(rec)

        self._probe_records = all_records

        # Atomic output: write to temp path, then rename, then write consumed marker.
        # This ensures that a crash during write does not permanently consume the
        # one-touch token without persisting results.
        rankings_df = _records_to_dataframe(all_records)
        output_path = self.output_dir / PROBE_RANKINGS_PARQUET
        tmp_path = self.output_dir / (PROBE_RANKINGS_PARQUET + ".tmp")
        try:
            rankings_df.to_parquet(str(tmp_path), index=False)
            os.replace(str(tmp_path), str(output_path))
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        # Write consumed marker ONLY after rankings are durably persisted.
        self._consumed_marker.write_text(self.registration_hash)
        logger.info(
            "Probe rankings written: %s (%d rows)",
            output_path, len(rankings_df),
        )
        return rankings_df

    def load_existing_rankings(self) -> pd.DataFrame:
        """
        Load previously written probe rankings (must already be consumed).

        Validates that the registration hash column matches.
        """
        output_path = self.output_dir / PROBE_RANKINGS_PARQUET
        if not output_path.exists():
            raise FileNotFoundError(
                f"Probe rankings not found: {output_path}. "
                "Run score_probes() first."
            )
        df = pd.read_parquet(str(output_path))
        # Validate registration hash in the data
        if COL_REGISTRATION_HASH in df.columns:
            hashes = df[COL_REGISTRATION_HASH].unique()
            if len(hashes) != 1 or hashes[0] != self.registration_hash:
                raise RegistrationHashMismatchError(
                    f"Rankings file registration hash mismatch. "
                    f"Expected {self.registration_hash[:16]}, found {hashes}."
                )
        return df


# ---------------------------------------------------------------------------
# Splits-manifest hash verification helper
# ---------------------------------------------------------------------------

def _verify_splits_manifest_hash(
    splits_parquet: Path,
    registration: dict,
) -> None:
    """
    Verify the splits-manifest hash (exact file bytes) against the
    registration document BEFORE any probe rows are read.

    Raises SplitsManifestHashMismatchError on mismatch.
    """
    reg_hash = registration.get("splits_manifest_hash", "")
    if not reg_hash or reg_hash == "unspecified":
        logger.warning(
            "splits_manifest_hash is absent or 'unspecified' in registration; "
            "skipping splits manifest byte-level verification."
        )
        return
    actual_hash = hashlib.sha256()
    with open(splits_parquet, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            actual_hash.update(chunk)
    actual_hex = actual_hash.hexdigest()
    if actual_hex != reg_hash:
        raise SplitsManifestHashMismatchError(
            f"Splits manifest hash mismatch. "
            f"Registration recorded {reg_hash[:16]}..., "
            f"actual file = {actual_hex[:16]}... "
            "The splits parquet file differs from what was registered. "
            "Do not proceed — evaluation integrity is compromised."
        )
    logger.info("Splits manifest hash verified: %s", actual_hex[:16])


class SplitsManifestHashMismatchError(RuntimeError):
    """Raised when the splits manifest file hash does not match the registration."""


# ---------------------------------------------------------------------------
# OOF artifact fingerprint verification helper
# ---------------------------------------------------------------------------

def _verify_oof_artifacts(
    artifacts_dir: Path,
    registration: dict,
) -> None:
    """
    Verify OOF artifact hashes and the selected-v1 eval hash recorded in
    the registration document.

    Checks:
      - oof_artifacts_hash (config.json + fingerprint.json)
      - oof_scoring_fingerprint (config_fingerprint from fingerprint.json)

    Raises OOFArtifactHashMismatchError on mismatch.
    """
    from pipeline.statistical_registration import (
        _hash_oof_artifacts,
        _read_oof_fingerprint,
    )

    reg_oof_hash = registration.get("oof_artifacts_hash", "")
    if reg_oof_hash:
        actual_oof_hash = _hash_oof_artifacts(artifacts_dir)
        if actual_oof_hash != reg_oof_hash:
            raise OOFArtifactHashMismatchError(
                f"OOF artifacts hash mismatch. "
                f"Registration: {reg_oof_hash[:16]}..., "
                f"Actual: {actual_oof_hash[:16]}... "
                "OOF artifact directory has changed since registration."
            )
        logger.info("OOF artifacts hash verified: %s", actual_oof_hash[:16])

    reg_fp = registration.get("oof_scoring_fingerprint", "")
    if reg_fp:
        actual_fp = _read_oof_fingerprint(artifacts_dir)
        if actual_fp and actual_fp != reg_fp:
            raise OOFArtifactHashMismatchError(
                f"OOF scoring fingerprint mismatch. "
                f"Registration: {reg_fp!r}, Actual: {actual_fp!r}. "
                "OOF artifact directory has changed since registration."
            )
        logger.info("OOF scoring fingerprint verified: %s", actual_fp)


class OOFArtifactHashMismatchError(RuntimeError):
    """Raised when OOF artifact hashes do not match the registration."""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fixed_probe_evaluator",
        description="Fixed-probe evaluator CLI for the paired statistical evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # score: full real-data wiring
    # ------------------------------------------------------------------
    sc = sub.add_parser(
        "score",
        help=(
            "Score all registered systems against the fixed probe set. "
            "Verifies registration hash, splits manifest hash, and OOF artifact "
            "hashes before reading any probe data. Writes probe_rankings.parquet "
            "atomically and sets one-touch consumed marker."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sc.add_argument(
        "--registration-file", required=True,
        help="Path to retrospective_registration.json (verified before any probe read).",
    )
    sc.add_argument(
        "--splits-parquet", required=True,
        help=(
            "Path to bteh_splits.parquet (full manifest with all splits). "
            "Byte hash must match splits_manifest_hash in registration."
        ),
    )
    sc.add_argument(
        "--oof-artifacts-dir", required=True,
        help="Path to OOF calibration output directory (frozen artifacts).",
    )
    sc.add_argument(
        "--crop-manifest", required=True,
        help="Path to bteh_crop_manifest.parquet.",
    )
    sc.add_argument(
        "--ref-embeddings-dir", required=True,
        help=(
            "Directory containing reference embedding .npy files and "
            "_mapping.parquet files (miewid, ear_miewid_projected)."
        ),
    )
    sc.add_argument(
        "--query-embeddings-dir", default=None,
        help=(
            "Directory containing query embedding files. "
            "Defaults to --ref-embeddings-dir if not provided."
        ),
    )
    sc.add_argument(
        "--output-dir", required=True,
        help="Directory for probe_rankings.parquet and consumed marker.",
    )
    sc.add_argument(
        "--cache-dir", default=None,
        help="Directory for persistent LightGlue feature cache.",
    )
    sc.add_argument(
        "--device", default="cpu",
        help="Torch device for LightGlue (e.g. 'cpu', 'cuda:0').",
    )
    sc.add_argument(
        "--disable-cudnn", action="store_true",
        help="Disable cuDNN for LightGlue (use for reproducibility).",
    )
    sc.add_argument(
        "--max-keypoints", type=int, default=1024,
        help="Maximum keypoints for LightGlue.",
    )
    sc.add_argument(
        "--systems", nargs="*", default=None,
        help=(
            "System names to score. Defaults to all non-exploratory systems. "
            "Exploratory systems are included if explicitly listed."
        ),
    )
    sc.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Verify all hashes and artifacts without executing any probe scoring. "
            "Exits after validation report."
        ),
    )

    # ------------------------------------------------------------------
    # load: check an existing rankings file
    # ------------------------------------------------------------------
    ld = sub.add_parser(
        "load",
        help=(
            "Load and verify an existing probe_rankings.parquet. "
            "Checks registration hash in rankings, reports summary statistics. "
            "Does not execute probes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ld.add_argument(
        "--registration-file", required=True,
        help="Path to retrospective_registration.json.",
    )
    ld.add_argument(
        "--rankings-dir", required=True,
        help="Directory containing probe_rankings.parquet and consumed marker.",
    )

    # ------------------------------------------------------------------
    # postprocess: paired bootstrap + candidate report (no probe execution)
    # ------------------------------------------------------------------
    pp = sub.add_parser(
        "postprocess",
        help=(
            "Run paired bootstrap and candidate report from existing probe rankings. "
            "Verifies registration hash in rankings. "
            "Runs 10k bootstrap (fixed seed). "
            "Writes metrics, candidate_report.json, future_session_protocol.json. "
            "Does NOT execute probes. Does NOT mutate production artifacts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pp.add_argument(
        "--registration-file", required=True,
        help="Path to retrospective_registration.json.",
    )
    pp.add_argument(
        "--rankings-dir", required=True,
        help="Directory containing probe_rankings.parquet.",
    )
    pp.add_argument(
        "--output-dir", required=True,
        help="Directory to write bootstrap metrics, candidate_report.json, future_session_protocol.json.",
    )
    pp.add_argument(
        "--candidate-system", default="selected_v1_plus_both_local",
        help="System name to evaluate as the primary challenger.",
    )
    pp.add_argument(
        "--baseline-system", default="selected_v1",
        help="System name to use as baseline.",
    )
    pp.add_argument(
        "--n-replicates", type=int, default=10_000,
        help="Number of bootstrap replicates (default: 10000).",
    )
    pp.add_argument(
        "--covariate-shift-flag", action="store_true",
        help=(
            "Raise covariate shift flag in the report. "
            "Set if baseline query-RR drops > 0.05 vs OOF mean."
        ),
    )
    pp.add_argument(
        "--runtime-p95-seconds",
        type=float,
        default=None,
        help="Measured or pre-registered projected p95 local rerank latency.",
    )
    pp.add_argument(
        "--coverage",
        type=float,
        default=None,
        help="Fraction of probe queries with every required local channel.",
    )

    return p


def _cmd_score(args) -> int:
    """
    Execute the 'score' subcommand.

    Order of operations (fail-fast):
    1. Verify registration hash (bytes of file unchanged).
    2. Verify splits manifest hash (exact bytes match registration record).
    3. Verify OOF artifact hash and scoring fingerprint.
    4. Check one-touch consumed marker.
    5. Load full split manifest.
    6. Load miewid + ear_miewid_projected embedding matrices and mappings.
    7. Load OOF ensemble artifacts (calibrators, weights, frozen K).
    8. Instantiate StrictLocalMatcher + shared cache + body/ear scorers.
    9. Score all registered systems.
    10. Write probe_rankings.parquet atomically.
    11. Write consumed marker.
    """
    from pipeline.statistical_registration import (
        load_and_verify_registration,
        RegistrationAlreadyConsumedError,
    )
    from pipeline.ensemble_inference import load_ensemble_artifacts
    from pipeline.local_oof_calibration import (
        _load_embedding_matrices_and_mappings,
        _instantiate_local_scorers,
        GLOBAL_CHANNELS,
        LOCAL_IDENTITY_SCORER_MAX_SESSIONS,
    )

    reg_path = Path(args.registration_file)
    splits_path = Path(args.splits_parquet)
    oof_dir = Path(args.oof_artifacts_dir)
    crop_path = Path(args.crop_manifest)
    ref_emb_dir = Path(args.ref_embeddings_dir)
    query_emb_dir = (
        Path(args.query_embeddings_dir) if args.query_embeddings_dir else ref_emb_dir
    )
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    # --- Step 1: Verify registration hash ---
    logger.info("Verifying registration hash ...")
    registration = load_and_verify_registration(reg_path)
    logger.info("Registration hash verified: %s", registration["registration_hash"][:16])

    # --- Step 2: Verify splits manifest hash (exact bytes) BEFORE reading probes ---
    logger.info("Verifying splits manifest hash ...")
    _verify_splits_manifest_hash(splits_path, registration)

    # --- Step 3: Verify OOF artifact hash and fingerprint ---
    logger.info("Verifying OOF artifact hashes ...")
    _verify_oof_artifacts(oof_dir, registration)

    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_ok",
            "registration_hash": registration["registration_hash"],
            "splits_manifest_hash_verified": True,
            "oof_artifacts_verified": True,
            "note": "dry-run: no probe scoring performed.",
        }, indent=2))
        return 0

    # --- Step 4: Check consumed marker (one-touch semantics) ---
    output_dir.mkdir(parents=True, exist_ok=True)
    consumed_marker = output_dir / CONSUMED_MARKER_FILENAME
    if consumed_marker.exists():
        from pipeline.statistical_registration import RegistrationAlreadyConsumedError
        raise RegistrationAlreadyConsumedError(
            f"Probes have already been consumed for registration "
            f"{registration['registration_hash'][:16]}. "
            f"Consumed marker: {consumed_marker}. "
            "Fixed-probe evaluation may only be run once per registration."
        )

    # --- Step 5: Load full split manifest ---
    logger.info("Loading full split manifest: %s", splits_path)
    splits_df = pd.read_parquet(str(splits_path))
    required_cols = {"image_id", "individual_id", "split"}
    missing = required_cols - set(splits_df.columns)
    if missing:
        raise ValueError(f"Split manifest missing required columns: {missing}")
    if "session_id" not in splits_df.columns:
        splits_df["session_id"] = "unknown"
    splits_df["session_id"] = splits_df["session_id"].fillna("unknown").astype(str)

    # --- Step 6: Load embedding matrices and mappings ---
    logger.info("Loading embedding matrices (%s) ...", ref_emb_dir)
    all_gallery_ids = set(
        splits_df.loc[splits_df["split"].isin({"gallery", "held_out_gallery"}), "image_id"]
        .astype(str)
    )
    embedding_matrices, descriptor_mappings = _load_embedding_matrices_and_mappings(
        embeddings_dir=ref_emb_dir,
        channels=list(GLOBAL_CHANNELS),
        gallery_ids=all_gallery_ids,
    )

    if query_emb_dir != ref_emb_dir:
        all_probe_ids = set(
            splits_df.loc[splits_df["split"].isin({"probe", "held_out_probe"}), "image_id"]
            .astype(str)
        )
        q_mats, q_maps = _load_embedding_matrices_and_mappings(
            embeddings_dir=query_emb_dir,
            channels=list(GLOBAL_CHANNELS),
            gallery_ids=all_probe_ids,
        )
        for ch in q_mats:
            if ch in embedding_matrices:
                offset = len(embedding_matrices[ch])
                q_map_reindexed = q_maps[ch].copy()
                q_map_reindexed["embedding_row"] = (
                    q_map_reindexed["embedding_row"].astype(int) + offset
                )
                embedding_matrices[ch] = np.concatenate(
                    [embedding_matrices[ch], q_mats[ch]], axis=0
                )
                descriptor_mappings[ch] = pd.concat(
                    [descriptor_mappings[ch], q_map_reindexed], ignore_index=True
                )
            else:
                embedding_matrices[ch] = q_mats[ch]
                descriptor_mappings[ch] = q_maps[ch]

    # --- Step 7: Load OOF ensemble artifacts ---
    logger.info("Loading OOF ensemble artifacts: %s", oof_dir)
    artifacts = load_ensemble_artifacts(oof_dir)

    # --- Step 8: Instantiate local scorers ---
    logger.info(
        "Instantiating StrictLocalMatcher (device=%s, max_keypoints=%d) ...",
        args.device, args.max_keypoints,
    )
    local_scorer_body, local_scorer_ear = _instantiate_local_scorers(
        device=args.device,
        disable_cudnn=args.disable_cudnn,
        max_keypoints=args.max_keypoints,
        max_sessions=LOCAL_IDENTITY_SCORER_MAX_SESSIONS,
        cache_dir=cache_dir,
    )

    # --- Step 9: Load crop manifest ---
    logger.info("Loading crop manifest: %s", crop_path)
    crop_df = pd.read_parquet(str(crop_path))

    # --- Step 10: Score all registered systems ---
    evaluator = FixedProbeEvaluator(reg_path, output_dir)
    rankings_df = evaluator.score_probes(
        splits_df=splits_df,
        artifacts=artifacts,
        crop_df=crop_df,
        embedding_matrices=embedding_matrices,
        descriptor_mappings=descriptor_mappings,
        local_scorer_body=local_scorer_body,
        local_scorer_ear=local_scorer_ear,
        systems=args.systems,
    )

    print(json.dumps({
        "status": "ok",
        "registration_hash": registration["registration_hash"],
        "output_dir": str(output_dir),
        "rankings_rows": len(rankings_df),
        "systems_scored": sorted(rankings_df["system_name"].unique().tolist())
            if "system_name" in rankings_df.columns else [],
    }, indent=2))
    return 0


def _cmd_load(args) -> int:
    """Execute the 'load' subcommand: verify and summarise existing rankings."""
    from pipeline.statistical_registration import load_and_verify_registration

    reg_path = Path(args.registration_file)
    rankings_dir = Path(args.rankings_dir)

    logger.info("Verifying registration: %s", reg_path)
    registration = load_and_verify_registration(reg_path)
    reg_hash = registration["registration_hash"]

    evaluator = FixedProbeEvaluator(reg_path, rankings_dir)

    if not evaluator.is_consumed:
        print(json.dumps({
            "status": "not_consumed",
            "registration_hash": reg_hash,
            "consumed_marker": str(evaluator._consumed_marker),
            "note": "Probes have not been scored yet. Run 'score' first.",
        }, indent=2))
        return 0

    rankings_df = evaluator.load_existing_rankings()

    systems = (
        sorted(rankings_df["system_name"].unique().tolist())
        if "system_name" in rankings_df.columns
        else []
    )
    n_queries = (
        rankings_df["query_image_id"].nunique()
        if "query_image_id" in rankings_df.columns
        else 0
    )
    probe_types = (
        sorted(rankings_df["probe_type"].unique().tolist())
        if "probe_type" in rankings_df.columns
        else []
    )

    print(json.dumps({
        "status": "ok",
        "registration_hash": reg_hash,
        "rankings_rows": len(rankings_df),
        "n_unique_queries": n_queries,
        "systems": systems,
        "probe_types": probe_types,
        "consumed_marker_present": True,
    }, indent=2))
    return 0


def _cmd_postprocess(args) -> int:
    """
    Execute the 'postprocess' subcommand.

    Loads existing probe rankings, verifies registration hash, runs
    paired bootstrap with fixed seed, writes metrics/report/protocol.
    Does NOT execute probes. Does NOT mutate production artifacts.
    """
    from pipeline.statistical_registration import load_and_verify_registration
    from pipeline.paired_bootstrap import (
        build_query_records_from_dataframe,
        run_paired_bootstrap,
        _per_identity_mrr_from_records,
    )
    from pipeline.candidate_report import (
        build_decision_report,
        build_future_session_protocol,
        write_decision_report,
        write_future_session_protocol,
    )
    from pipeline.eval_metrics import identity_macro_mrr
    from pipeline.power_simulation import SIMULATION_SEED

    reg_path = Path(args.registration_file)
    rankings_dir = Path(args.rankings_dir)
    output_dir = Path(args.output_dir)

    logger.info("Verifying registration: %s", reg_path)
    registration = load_and_verify_registration(reg_path)
    reg_hash = registration["registration_hash"]

    # Load and verify existing rankings
    evaluator = FixedProbeEvaluator(reg_path, rankings_dir)
    rankings_df = evaluator.load_existing_rankings()

    # Verify registration hash in the rankings data
    if COL_REGISTRATION_HASH in rankings_df.columns:
        found_hashes = rankings_df[COL_REGISTRATION_HASH].unique().tolist()
        if len(found_hashes) != 1 or found_hashes[0] != reg_hash:
            raise RegistrationHashMismatchError(
                f"Rankings registration hash mismatch. "
                f"Expected {reg_hash[:16]}, found {found_hashes}."
            )
    logger.info("Rankings registration hash verified: %s", reg_hash[:16])

    # Build identity lists from registration
    cluster_sizes = registration.get("cluster_sizes", {})
    temporal_ids = sorted(cluster_sizes.get("temporal", {}).keys())
    onboarding_ids = sorted(cluster_sizes.get("onboarding", {}).keys())

    # Build QueryRecord lists for each system
    systems_in_rankings = (
        rankings_df["system_name"].unique().tolist()
        if "system_name" in rankings_df.columns
        else []
    )
    records_by_system = {}
    for sys_name in systems_in_rankings:
        records_by_system[sys_name] = build_query_records_from_dataframe(
            rankings_df, sys_name
        )
    required_systems = {
        args.baseline_system,
        args.candidate_system,
    }
    missing_systems = sorted(required_systems - set(records_by_system))
    if missing_systems:
        raise ValueError(
            "Rankings are missing systems required for postprocessing: "
            f"{missing_systems}. Available systems: {sorted(records_by_system)}."
        )

    logger.info(
        "Running paired bootstrap: %d replicates, seed=%d, "
        "%d temporal IDs, %d onboarding IDs ...",
        args.n_replicates, SIMULATION_SEED,
        len(temporal_ids), len(onboarding_ids),
    )

    bootstrap_result = run_paired_bootstrap(
        records_by_system=records_by_system,
        temporal_ids=temporal_ids,
        onboarding_ids=onboarding_ids,
        system_a=args.baseline_system,
        system_b_primary=args.candidate_system,
        system_b_body="selected_v1_plus_body_local",
        system_b_ear="selected_v1_plus_ear_local",
        system_b_frozen="selected_v1_frozen_ear",
        n_replicates=args.n_replicates,
        seed=SIMULATION_SEED,
    )
    selected_eval_hash = str(registration.get("selected_v1_eval_hash", ""))
    if re.fullmatch(r"[0-9a-f]{64}", selected_eval_hash):
        verify_legacy_map_regression(
            bootstrap_result.system_metrics[args.baseline_system][
                "query_weighted_mrr"
            ]
        )
    else:
        logger.warning(
            "Skipping selected-v1 regression enforcement because registration "
            "contains a non-production eval hash."
        )

    # Per-stratum deltas for subset consistency gate
    def _stratum_delta(probe_type: str) -> Optional[float]:
        if "probe_type" not in rankings_df.columns:
            return None
        sub = rankings_df[rankings_df["probe_type"] == probe_type]
        if sub.empty:
            return None
        recs_a = build_query_records_from_dataframe(sub, args.baseline_system)
        recs_b = build_query_records_from_dataframe(sub, args.candidate_system)
        if not recs_a or not recs_b:
            return None
        mrr_a = identity_macro_mrr(_per_identity_mrr_from_records(recs_a))
        mrr_b = identity_macro_mrr(_per_identity_mrr_from_records(recs_b))
        return mrr_b - mrr_a

    temporal_delta = _stratum_delta("temporal")
    onboarding_delta = _stratum_delta("onboarding")

    report = build_decision_report(
        bootstrap_result,
        registration,
        temporal_delta=temporal_delta,
        onboarding_delta=onboarding_delta,
        covariate_shift_flag=args.covariate_shift_flag,
        candidate_system=args.candidate_system,
        baseline_system=args.baseline_system,
        runtime_p95_seconds=args.runtime_p95_seconds,
        coverage=args.coverage,
    )
    future_protocol = build_future_session_protocol(
        registration,
        bootstrap_result,
        covariate_shift_flag=args.covariate_shift_flag,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = write_decision_report(report, output_dir)
    protocol_path = write_future_session_protocol(future_protocol, output_dir)

    # Write bootstrap metrics summary
    metrics_path = output_dir / "bootstrap_metrics.json"
    metrics_out = {
        "registration_hash": reg_hash,
        "candidate_system": args.candidate_system,
        "baseline_system": args.baseline_system,
        "n_replicates": args.n_replicates,
        "seed": SIMULATION_SEED,
        "system_metrics": bootstrap_result.system_metrics,
        "primary_contrast": {
            "contrast_name": bootstrap_result.primary.contrast_name,
            "point_delta": bootstrap_result.primary.point_delta,
            "ci_lo": bootstrap_result.primary.ci_lo,
            "ci_hi": bootstrap_result.primary.ci_hi,
            "p_value": bootstrap_result.primary.p_value,
            "p_value_holm": bootstrap_result.primary.p_value_holm,
            "reject_h0": bootstrap_result.primary.reject_h0,
        },
        "secondary_contrasts": [
            {
                "contrast_name": c.contrast_name,
                "point_delta": c.point_delta,
                "ci_lo": c.ci_lo,
                "ci_hi": c.ci_hi,
                "p_value_holm": c.p_value_holm,
            }
            for c in bootstrap_result.secondaries
        ],
    }
    with open(metrics_path, "w") as fh:
        json.dump(metrics_out, fh, indent=2, sort_keys=True)
    logger.info("Bootstrap metrics written: %s", metrics_path)

    print(json.dumps({
        "status": "ok",
        "registration_hash": reg_hash,
        "decision": report.decision,
        "primary_delta": bootstrap_result.primary.point_delta,
        "primary_ci_lo": bootstrap_result.primary.ci_lo,
        "primary_ci_hi": bootstrap_result.primary.ci_hi,
        "report": str(report_path),
        "future_protocol": str(protocol_path),
        "bootstrap_metrics": str(metrics_path),
    }, indent=2))
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "score":
        return _cmd_score(args)
    if args.command == "load":
        return _cmd_load(args)
    if args.command == "postprocess":
        return _cmd_postprocess(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
