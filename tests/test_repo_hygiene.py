"""
Repository hygiene tests.

Checks:
  1. Expected paths are covered by .gitignore.
  2. Sensitive/generated files are not tracked in git.
  3. Tracked text files contain no /home/<user>/ absolute paths.

These tests are intentionally lightweight — they call git commands on the
actual repository, not on synthetic data.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + list(args),
        capture_output=True,
        text=True,
        cwd=str(cwd or REPO_ROOT),
    )


def _check_ignored(rel_path: str) -> bool:
    """Return True if *rel_path* is covered by .gitignore."""
    result = _git("check-ignore", "--quiet", "--no-index", rel_path)
    return result.returncode == 0


def _tracked_files() -> list[str]:
    result = _git("ls-files")
    if result.returncode != 0:
        return []
    return result.stdout.splitlines()


# ---------------------------------------------------------------------------
# .gitignore coverage
# ---------------------------------------------------------------------------

class TestGitignoreCoverage:
    """Each expected path must be covered by .gitignore."""

    @pytest.mark.parametrize("rel_path", [
        "elephant_catalog_progress.json",
        "elephant_catalog_run.log",
        "generate_excel_catalog.py",
        "elephant_catalog.xlsx",
        "elephant_catalog_test.xlsx",
        # artifact root patterns (verified as paths, not directories)
        "bteh_artifacts/manifest.parquet",
        "BTEH_reid_artifacts/v1/manifests/bteh_image_manifest.parquet",
        "ELPephants_reid_artifacts/v1/manifests/elpephants_image_manifest.parquet",
        # generated outputs
        "crops/some_image.jpg",
        "ear_crops/some_crop.jpg",
        "embeddings/miewid.npy",
        "faiss_index/body.faiss",
        "calibration/isotonic.pkl",
        "checkpoints/epoch_01.pt",
        "reports/evaluation.html",
        "contact_sheets/page_01.jpg",
    ])
    def test_path_is_ignored(self, rel_path):
        assert _check_ignored(rel_path), (
            f"{rel_path!r} is NOT covered by .gitignore — add a rule to .gitignore"
        )


# ---------------------------------------------------------------------------
# No sensitive/generated files tracked
# ---------------------------------------------------------------------------

class TestNoSensitiveFilesTracked:
    """Sensitive or generated files must not appear in git ls-files."""

    def test_no_catalog_xlsx_tracked(self):
        tracked = _tracked_files()
        catalog_files = [f for f in tracked if "_catalog" in f.lower() and f.endswith(".xlsx")]
        assert catalog_files == [], (
            f"Catalog xlsx files are tracked: {catalog_files}\n"
            "Run: git rm --cached <file> && git commit"
        )

    def test_no_progress_json_tracked(self):
        tracked = _tracked_files()
        prog = [f for f in tracked if "catalog_progress" in f]
        assert prog == [], (
            f"Progress JSON is tracked: {prog}\n"
            "Run: git rm --cached <file> && git commit"
        )

    def test_no_local_catalog_generator_tracked(self):
        tracked = _tracked_files()
        generators = [f for f in tracked if f == "generate_excel_catalog.py"]
        assert generators == [], (
            "The local catalog generator contains participant-specific paths and "
            f"must remain untracked: {generators}"
        )

    def test_no_env_file_tracked(self):
        tracked = _tracked_files()
        env_files = [f for f in tracked if f == ".env"]
        assert env_files == [], f".env is tracked: {env_files}"

    def test_no_checkpoints_tracked(self):
        tracked = _tracked_files()
        chk = [f for f in tracked if "checkpoints/" in f]
        assert chk == [], f"Checkpoint files are tracked: {chk}"

    def test_no_faiss_tracked(self):
        tracked = _tracked_files()
        faiss = [f for f in tracked if "faiss_index/" in f]
        assert faiss == [], f"FAISS index files are tracked: {faiss}"

    def test_no_generated_crops_tracked(self):
        tracked = _tracked_files()
        crops = [
            f for f in tracked
            if f.startswith(("crops/", "ear_crops/", "processed_images/"))
        ]
        assert crops == [], f"Generated crop files are tracked: {crops}"


# ---------------------------------------------------------------------------
# No participant-specific /home/<user>/ paths in tracked text files
# ---------------------------------------------------------------------------

import re

_HOME_RE = re.compile(r"/home/[^/\s]+/")

_TEXT_EXTENSIONS = {
    ".py", ".yaml", ".yml", ".json", ".txt", ".md", ".toml",
    ".cfg", ".ini", ".sh", ".template", ".csv",
}

# Files that may legitimately document placeholder /home/ patterns
_SCAN_EXCLUDES = {
    ".env.template",   # template may document placeholder paths
    "elephant_name_to_path.csv",  # not tracked but in case it appears
    "scripts/repo_hygiene.py",
    "tests/test_repo_hygiene.py",
}

# Untracked files the user owns that should not be scanned even if git sees them
_USER_UNTRACKED = {
    "elephant_catalog.xlsx",
    "elephant_catalog_test.xlsx",
    "elephant_catalog_progress.json",
    "generate_excel_catalog.py",
}


class TestNoParticipantPaths:
    """Tracked text files must not contain /home/<user>/ absolute paths."""

    def test_no_home_paths_in_tracked_files(self):
        tracked = _tracked_files()
        violations = []

        for rel_str in tracked:
            if rel_str in _SCAN_EXCLUDES or rel_str in _USER_UNTRACKED:
                continue
            p = REPO_ROOT / rel_str
            if p.suffix.lower() not in _TEXT_EXTENSIONS:
                continue
            if not p.is_file():
                continue
            try:
                text = p.read_text(errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if _HOME_RE.search(line):
                    violations.append(f"{rel_str}:{i}: {line.strip()[:120]}")

        assert violations == [], (
            "Participant-specific /home/<user>/ paths found in tracked files:\n"
            + "\n".join(f"  {v}" for v in violations[:20])
            + ("\n  ... (truncated)" if len(violations) > 20 else "")
            + "\n\nReplace with environment variables or documented placeholders."
        )
