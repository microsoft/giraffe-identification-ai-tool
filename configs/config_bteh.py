# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
BTEH data-root and artifact configuration.

Source root  : read-only BTEH image tree — never modified by the pipeline.
Artifact root: versioned, writable outputs entirely separate from the source.

Configure via environment variables (copy .env.template to .env):
  BTEH_SOURCE_ROOT    - absolute path to the read-only BTEH image tree
                        default: <repo-parent>/shared/BTEH
  BTEH_ARTIFACT_ROOT  - absolute path to the writable artifact tree
                        default: <repo-parent>/shared/BTEH_reid_artifacts

Bumping ARTIFACT_SCHEMA_VERSION invalidates all prior artifacts; they must be
rebuilt from the canonical manifest.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from configs.config_artifacts import ARTIFACT_SCHEMA_VERSION

load_dotenv()

# Absolute path of the repository root (parent of this file's directory).
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Schema version
# Bump when the manifest or artifact schema changes incompatibly.
# All artifact subdirectories are nested under this version prefix so
# artifacts built under different schema versions never collide.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Source root — read-only
# ---------------------------------------------------------------------------
_source_default = str(REPO_ROOT.parent / "shared" / "BTEH")
BTEH_SOURCE_ROOT: Path = Path(os.getenv("BTEH_SOURCE_ROOT", _source_default))

# ---------------------------------------------------------------------------
# Artifact root — writable, versioned
# ---------------------------------------------------------------------------
_artifact_default = str(REPO_ROOT.parent / "shared" / "BTEH_reid_artifacts")
BTEH_ARTIFACT_ROOT: Path = Path(os.getenv("BTEH_ARTIFACT_ROOT", _artifact_default))

# Versioned subdirectory under the artifact root.
ARTIFACT_VERSION_ROOT: Path = BTEH_ARTIFACT_ROOT / ARTIFACT_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Artifact subdirectory names (under ARTIFACT_VERSION_ROOT)
# ---------------------------------------------------------------------------
MANIFEST_SUBDIR: str = "manifests"
SPLITS_SUBDIR: str = "splits"
CROPS_SUBDIR: str = "crops"
EMBEDDINGS_SUBDIR_BTEH: str = "embeddings"
FAISS_SUBDIR_BTEH: str = "faiss_index"
CALIBRATION_SUBDIR_BTEH: str = "calibration"
CHECKPOINTS_SUBDIR: str = "checkpoints"
REPORTS_SUBDIR: str = "reports"
CONTACT_SHEETS_SUBDIR: str = "contact_sheets"

# Canonical manifest filename
MANIFEST_FILENAME: str = "bteh_image_manifest.parquet"
SPLITS_FILENAME: str = "bteh_splits.parquet"

# ---------------------------------------------------------------------------
# Experiment artifact paths
# Head region crops and manifests are written under a separate experiment
# namespace; they never overwrite production selected-v1 artifacts.
# ---------------------------------------------------------------------------
EXPERIMENT_SUBDIR: str = "experiments"
FULL_LOCAL_ENSEMBLE_SUBDIR: str = "full_local_ensemble"
EXPERIMENT_ROOT: Path = (
    ARTIFACT_VERSION_ROOT / EXPERIMENT_SUBDIR / FULL_LOCAL_ENSEMBLE_SUBDIR
)
HEAD_CROPS_SUBDIR: str = "head_crops"
HEAD_MANIFEST_FILENAME: str = "head_manifest.parquet"
# the working tree (e.g. symlinked or configured to a local path).
# Referenced by scripts/repo_hygiene.py.
# ---------------------------------------------------------------------------
REPO_IGNORED_ARTIFACT_DIRS: list[str] = [
    "bteh_artifacts",
    "BTEH_reid_artifacts",
    "crops",
    "ear_crops",
    "contact_sheets",
]

# ---------------------------------------------------------------------------
# UUID / unresolved directory detection
# These regex patterns match top-level BTEH directories that are review buckets
# rather than named identities.  Such directories must never be treated as
# individual identity labels.
# ---------------------------------------------------------------------------
import re  # noqa: E402 — after constants block for clarity

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
HEX32_PATTERN = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)


def is_uuid_dir(name: str) -> bool:
    """Return True if *name* looks like a UUID or a 32-char hex string."""
    stripped = name.strip()
    return bool(UUID_PATTERN.match(stripped) or HEX32_PATTERN.match(stripped))


# ---------------------------------------------------------------------------
# Herd-suffix canonicalization
# Strip parenthetical herd annotations from folder names to get a clean
# display name; preserve the herd label as a separate metadata field.
# ---------------------------------------------------------------------------
_HERD_SUFFIX_RE = re.compile(r"\s*\(([^)]+)\)\s*$")


def split_herd_suffix(folder_name: str) -> tuple[str, str | None]:
    """
    Return (display_name, herd_label) for a raw BTEH folder name.

    >>> split_herd_suffix("Beauty (Herd 4) ")
    ('Beauty', 'Herd 4')
    >>> split_herd_suffix("Balu")
    ('Balu', None)
    """
    stripped = folder_name.strip()
    m = _HERD_SUFFIX_RE.search(stripped)
    if m:
        return stripped[: m.start()].strip(), m.group(1).strip()
    return stripped, None


def canonical_individual_id(display_name: str) -> str:
    """
    Derive a stable, lowercase, underscore-separated individual ID from a
    display name.  IDs are prefixed with ``bteh_`` to namespace them.

    >>> canonical_individual_id("Half Moon")
    'bteh_half_moon'
    >>> canonical_individual_id("Samala (Kunene_s son)")
    'bteh_samala__kunene_s_son_'
    """
    clean = display_name.strip().lower()
    # Replace spaces with underscores; keep other characters as-is so
    # parenthetical sub-labels (e.g. for special cases) survive but do not
    # collide with clean names.
    clean = re.sub(r"[ \t]+", "_", clean)
    return f"bteh_{clean}"
