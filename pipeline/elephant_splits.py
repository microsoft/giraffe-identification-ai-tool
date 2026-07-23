#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Deterministic, versioned split generator for canonical elephant datasets.

Two evaluation protocols are supported:

1. Known-identity temporal split
   For each identity that has images in at least two distinct sessions,
   the latest session(s) are reserved as probe; all earlier sessions form
   the gallery/train pool.  Identities with only one session are placed in
   the gallery with a non-evaluable flag.

2. Unseen-identity onboarding protocol
   A deterministic held-out fold of complete identities is reserved to
   simulate newly onboarded elephants.  Within the held-out set, a small
   gallery and separate-session probes are defined.  Identities with
   insufficient independent sessions are marked unsupported rather than
   leaking into training.

Split rules
-----------
* Sessions are derived from canonical manifest session_id / session_source
  columns. No re-parsing of paths or filenames is done here.
* Exact/near-duplicate groups (same content_hash) must not cross splits.
  Duplicate rows are excluded and only the primary representative participates.
* All derivatives of a source image (body crop, ear crops, augmented copies)
  must stay together — enforced via the image_id.
* Split manifests are versioned with a fingerprint derived from the input
  manifest fingerprint.

Output columns (added to a copy of the manifest):
  split           : "train", "probe", "gallery", "held_out_gallery",
                    "held_out_probe", "excluded"
  split_protocol  : "temporal", "unseen_identity", "non_evaluable", "excluded"
  evaluable       : bool — True when the identity has enough sessions for evaluation
  fold            : unseen-identity fold index (int, -1 for non-fold rows)

Usage
-----
    python pipeline/elephant_splits.py \\
        --manifest PATH           \\
        --output PATH             \\
        [--n-unseen-folds N]      \\  # number of unseen-identity folds (default: 5)
        [--min-sessions-temporal N] \\  # min sessions per identity for temporal eval (default: 2)
        [--seed N]                   # random seed for fold assignment (default: 42)
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from configs.config_artifacts import ARTIFACT_SCHEMA_VERSION
from utils.artifact_schema import fingerprint_dataframe_columns

logger = logging.getLogger(__name__)

# Minimum number of distinct sessions an identity must have to be eligible
# for the temporal evaluation protocol.
DEFAULT_MIN_SESSIONS_TEMPORAL = 2

# Default number of unseen-identity folds.
DEFAULT_N_UNSEEN_FOLDS = 5

# Minimum number of sessions within a held-out identity to form a gallery +
# separate-session probe (otherwise mark as unsupported for unseen-identity eval).
MIN_SESSIONS_FOR_UNSEEN_EVAL = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fingerprint_df(df: pd.DataFrame) -> str:
    """Deterministic fingerprint of a split manifest."""
    return fingerprint_dataframe_columns(
        df,
        [
            "image_id",
            "split",
            "split_protocol",
            "evaluable",
            "fold",
            "session_id",
        ],
        sort_by=["image_id"],
    )


def _validate_no_cross_split_duplicates(df: pd.DataFrame) -> list[str]:
    """
    Return error messages for content_hash groups that appear in more than
    one non-excluded split.
    """
    errors = []
    active = df[df["split"] != "excluded"].copy()
    if active.empty or "content_hash" not in active.columns:
        return errors
    active = active.dropna(subset=["content_hash"])
    cross = (
        active.groupby("content_hash")["split"]
        .nunique()
    )
    leakers = cross[cross > 1].index.tolist()
    if leakers:
        errors.append(
            f"{len(leakers)} content_hash groups span multiple splits: "
            f"{leakers[:5]}"
        )
    return errors


# ---------------------------------------------------------------------------
# Core split logic
# ---------------------------------------------------------------------------

def generate_splits(
    manifest: pd.DataFrame,
    min_sessions_temporal: int = DEFAULT_MIN_SESSIONS_TEMPORAL,
    n_unseen_folds: int = DEFAULT_N_UNSEEN_FOLDS,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign split labels to the manifest and return an annotated copy.

    Only rows with include_status in {"included", "duplicate_primary"} participate.
    Excluded rows receive split="excluded".
    """
    rng = np.random.default_rng(seed)

    df = manifest.copy()
    df["split"] = "excluded"
    df["split_protocol"] = "excluded"
    df["evaluable"] = False
    df["fold"] = -1

    # Work only with eligible rows (included + duplicate_primary which keeps a rep.)
    eligible_mask = df["include_status"].isin(["included", "duplicate_primary"])
    eligible_idx = df.index[eligible_mask].tolist()

    if not eligible_idx:
        logger.warning("No eligible rows found in manifest; all splits are empty.")
        return df

    eligible = df.loc[eligible_idx].copy()

    # --- Unseen-identity fold assignment ---
    # Assign each named identity (non-unresolved) to a seeded deterministic fold.
    named_ids = sorted(
        eligible[
            (eligible["individual_id"].notna()) &
            (eligible["individual_id"] != "unresolved")
        ]["individual_id"].unique()
    )
    shuffled_ids = rng.permutation(named_ids).tolist()
    fold_map: dict[str, int] = {}
    for i, ind_id in enumerate(shuffled_ids):
        fold_map[ind_id] = i % n_unseen_folds

    # --- Per-identity session analysis ---
    # Count distinct sessions per identity to determine evaluability.
    id_sessions = (
        eligible[eligible["individual_id"].notna()]
        .groupby("individual_id")["session_id"]
        .nunique()
        .to_dict()
    )

    # --- Temporal split assignment ---
    # For each named identity with enough sessions, the latest session(s) → probe;
    # all earlier sessions → train/gallery.
    temporal_probe_sessions: dict[str, set] = {}  # individual_id → set of session_ids
    temporal_gallery_sessions: dict[str, set] = {}

    for ind_id in named_ids:
        n_sess = id_sessions.get(ind_id, 0)
        ind_rows = eligible[eligible["individual_id"] == ind_id]
        sessions = sorted(ind_rows["session_id"].dropna().unique())

        if n_sess >= min_sessions_temporal:
            # Reserve the latest session as probe
            probe_sess = {sessions[-1]}
            gallery_sess = set(sessions[:-1])
        else:
            probe_sess = set()
            gallery_sess = set(sessions)

        temporal_probe_sessions[ind_id] = probe_sess
        temporal_gallery_sessions[ind_id] = gallery_sess

    # --- Build unseen-identity held-out folds ---
    # Identities in each fold are completely held out from training.
    # Within each held-out identity: one session → held_out_gallery,
    # a different session → held_out_probe (if ≥2 sessions).

    unseen_gallery_sessions: dict[str, set] = {}
    unseen_probe_sessions: dict[str, set] = {}
    unseen_evaluable: dict[str, bool] = {}

    for ind_id in named_ids:
        ind_rows = eligible[eligible["individual_id"] == ind_id]
        sessions = sorted(ind_rows["session_id"].dropna().unique())
        if len(sessions) >= MIN_SESSIONS_FOR_UNSEEN_EVAL:
            unseen_gallery_sessions[ind_id] = {sessions[0]}
            unseen_probe_sessions[ind_id] = {sessions[-1]}
            unseen_evaluable[ind_id] = True
        else:
            unseen_gallery_sessions[ind_id] = set(sessions)
            unseen_probe_sessions[ind_id] = set()
            unseen_evaluable[ind_id] = False

    # --- Assign splits ---
    # For each eligible row, determine its split based on its identity and session.
    # Priority:
    #  1. Held-out (unseen identity fold) rows
    #  2. Temporal probe rows (for non-held-out identities)
    #  3. Gallery/train rows (for non-held-out identities)
    #  4. Unresolved UUID dir rows → excluded

    # We use fold 0 as the held-out fold for all assignments
    # (in practice the caller may use a specific fold; this provides all fold labels)

    for idx in eligible_idx:
        row = df.loc[idx]
        ind_id = row["individual_id"]
        sess_id = row.get("session_id")

        if pd.isna(ind_id) or ind_id == "unresolved":
            df.loc[idx, "split"] = "excluded"
            df.loc[idx, "split_protocol"] = "excluded"
            continue

        fold_idx = fold_map.get(ind_id, -1)
        df.loc[idx, "fold"] = fold_idx

        # Determine evaluability for temporal
        n_sess = id_sessions.get(ind_id, 0)
        evaluable = n_sess >= min_sessions_temporal

        # --- Unseen identity protocol ---
        if fold_idx == 0:
            # This identity is the held-out fold (fold 0)
            in_gallery_sess = sess_id in unseen_gallery_sessions.get(ind_id, set())
            in_probe_sess = sess_id in unseen_probe_sessions.get(ind_id, set())
            if in_probe_sess and unseen_evaluable.get(ind_id, False):
                df.loc[idx, "split"] = "held_out_probe"
                df.loc[idx, "split_protocol"] = "unseen_identity"
                df.loc[idx, "evaluable"] = True
            elif in_gallery_sess or not unseen_evaluable.get(ind_id, False):
                df.loc[idx, "split"] = "held_out_gallery"
                df.loc[idx, "split_protocol"] = "unseen_identity"
                df.loc[idx, "evaluable"] = unseen_evaluable.get(ind_id, False)
            else:
                # Remaining sessions of held-out identity also go to held_out_gallery
                df.loc[idx, "split"] = "held_out_gallery"
                df.loc[idx, "split_protocol"] = "unseen_identity"
                df.loc[idx, "evaluable"] = False
        else:
            # --- Temporal protocol for non-held-out identities ---
            in_probe_sess = sess_id in temporal_probe_sessions.get(ind_id, set())
            if in_probe_sess and evaluable:
                df.loc[idx, "split"] = "probe"
                df.loc[idx, "split_protocol"] = "temporal"
                df.loc[idx, "evaluable"] = True
            else:
                df.loc[idx, "split"] = "gallery"
                df.loc[idx, "split_protocol"] = "temporal"
                df.loc[idx, "evaluable"] = evaluable

    # --- Validate no cross-split duplicates ---
    dup_errors = _validate_no_cross_split_duplicates(df)
    if dup_errors:
        raise RuntimeError("Cross-split duplicate groups:\n" + "\n".join(dup_errors))

    return df


def validate_splits(df: pd.DataFrame) -> list[str]:
    """Return integrity errors for a split manifest (empty = OK)."""
    errors = []

    valid_splits = {
        "train", "probe", "gallery", "held_out_gallery",
        "held_out_probe", "excluded",
    }
    bad = set(df["split"].dropna().unique()) - valid_splits
    if bad:
        errors.append(f"Unknown split values: {bad}")

    # No individual may appear in both probe and gallery
    active = df[df["split"].isin({"probe", "gallery"})]
    if not active.empty:
        probe_ids = set(active[active["split"] == "probe"]["individual_id"].dropna())
        gallery_ids = set(active[active["split"] == "gallery"]["individual_id"].dropna())
        overlap = probe_ids & gallery_ids
        # An individual may have IMAGES in both probe and gallery only if the
        # split protocol is temporal (some sessions gallery, one session probe).
        # What must not happen is the same session appearing in both.
        for ind_id in overlap:
            ind_probe = active[
                (active["split"] == "probe") & (active["individual_id"] == ind_id)
            ]
            ind_gallery = active[
                (active["split"] == "gallery") & (active["individual_id"] == ind_id)
            ]
            shared_sessions = (
                set(ind_probe["session_id"].dropna()) &
                set(ind_gallery["session_id"].dropna())
            )
            if shared_sessions:
                errors.append(
                    f"Individual {ind_id}: sessions appear in both probe and gallery: "
                    f"{shared_sessions}"
                )

    # Cross-split duplicate check
    errors.extend(_validate_no_cross_split_duplicates(df))

    return errors


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate deterministic canonical elephant dataset splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to a canonical elephant image manifest parquet",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the normalized splits parquet",
    )
    parser.add_argument(
        "--schema-version",
        default=ARTIFACT_SCHEMA_VERSION,
    )
    parser.add_argument(
        "--n-unseen-folds",
        type=int,
        default=DEFAULT_N_UNSEEN_FOLDS,
        help=f"Number of unseen-identity folds (default: {DEFAULT_N_UNSEEN_FOLDS})",
    )
    parser.add_argument(
        "--min-sessions-temporal",
        type=int,
        default=DEFAULT_MIN_SESSIONS_TEMPORAL,
        help=f"Min sessions for temporal eval (default: {DEFAULT_MIN_SESSIONS_TEMPORAL})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load existing splits and validate without regenerating",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    if args.validate_only:
        if not output_path.exists():
            logger.error("Splits file not found: %s", output_path)
            return 1
        df = pd.read_parquet(output_path)
        errors = validate_splits(df)
        if errors:
            for e in errors:
                logger.error(e)
            return 1
        logger.info("Splits valid: %d rows", len(df))
        return 0

    if not manifest_path.exists():
        logger.error(
            "Canonical image manifest not found at %s.", manifest_path
        )
        return 1

    manifest = pd.read_parquet(manifest_path)
    logger.info("Loaded manifest: %d rows", len(manifest))

    splits_df = generate_splits(
        manifest,
        min_sessions_temporal=args.min_sessions_temporal,
        n_unseen_folds=args.n_unseen_folds,
        seed=args.seed,
    )

    errors = validate_splits(splits_df)
    if errors:
        for e in errors:
            logger.error(e)
        return 1

    # Summary
    split_counts = splits_df["split"].value_counts().to_dict()
    logger.info("Split counts: %s", split_counts)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fingerprint = _fingerprint_df(splits_df)
    logger.info("Splits fingerprint: %s", fingerprint)

    splits_df.to_parquet(output_path, index=False)
    logger.info("Splits written to: %s", output_path)

    sidecar = output_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": args.schema_version,
                "splits_fingerprint": fingerprint,
                "n_unseen_folds": args.n_unseen_folds,
                "min_sessions_temporal": args.min_sessions_temporal,
                "seed": args.seed,
                "split_counts": {k: int(v) for k, v in split_counts.items()},
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
