#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Repository hygiene verification script.

Checks:
  1. Expected paths are covered by .gitignore (git check-ignore).
  2. Sensitive / generated files are not already tracked (git ls-files).
  3. Tracked text files contain no participant-specific absolute paths
     matching /home/<user>/... patterns.

Exit codes:
  0  all checks pass
  1  one or more checks failed (details printed to stdout)

Usage:
    python scripts/repo_hygiene.py [--repo-root PATH] [--strict]

    --strict  treat warnings as failures (default: only hard failures count)
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths that MUST be ignored by .gitignore
# ---------------------------------------------------------------------------
MUST_BE_IGNORED: list[str] = [
    "elephant_catalog_progress.json",
    "elephant_catalog_run.log",
    "generate_excel_catalog.py",
    # catalog xlsx patterns — test with actual suffixed names
    "elephant_catalog.xlsx",
    "elephant_catalog_test.xlsx",
    # artifact roots
    "bteh_artifacts/",
    "BTEH_reid_artifacts/",
    "ELPephants_reid_artifacts/",
    # generated sub-dirs
    "crops/dummy.jpg",
    "ear_crops/dummy.jpg",
    "embeddings/dummy.npy",
    "faiss_index/dummy.faiss",
    "calibration/dummy.pkl",
    "checkpoints/dummy.pt",
    "reports/dummy.html",
    "contact_sheets/dummy.jpg",
]

# ---------------------------------------------------------------------------
# Patterns that must NOT appear in git ls-files output
# ---------------------------------------------------------------------------
MUST_NOT_BE_TRACKED_PATTERNS: list[re.Pattern] = [
    re.compile(r"elephant_catalog_progress\.json"),
    re.compile(r"^generate_excel_catalog\.py$"),
    re.compile(r".*_catalog.*\.(xlsx|csv)$", re.IGNORECASE),
    re.compile(r"\.env$"),
    re.compile(r"checkpoints/"),
    re.compile(r"faiss_index/"),
    re.compile(r"embeddings/"),
    re.compile(r"crops/"),
    re.compile(r"ear_crops/"),
    re.compile(r"reports/"),
    re.compile(r"contact_sheets/"),
]

# ---------------------------------------------------------------------------
# Absolute-path patterns that must not appear in tracked text files
# ---------------------------------------------------------------------------
HOME_PATH_RE = re.compile(r"/home/[^/\s]+/")

# Text file extensions to scan
TEXT_EXTENSIONS: set[str] = {
    ".py", ".yaml", ".yml", ".json", ".txt", ".md", ".toml",
    ".cfg", ".ini", ".sh", ".template", ".csv",
}

# Files whose content is expected to contain /home/ patterns (e.g. .env.template
# documenting a placeholder) — excluded from the scan.
SCAN_EXCLUDES: set[str] = {
    "scripts/repo_hygiene.py",
    "tests/test_repo_hygiene.py",
}


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))


def check_gitignore(repo_root: Path) -> list[str]:
    """Return list of failure messages for paths that should be ignored but aren't."""
    failures = []
    for rel_path in MUST_BE_IGNORED:
        # Create a dummy file/dir entry for the check when needed
        test_path = repo_root / rel_path
        result = run(
            ["git", "check-ignore", "--quiet", "--no-index", rel_path],
            cwd=repo_root,
        )
        if result.returncode != 0:
            failures.append(f"NOT ignored by .gitignore: {rel_path}")
    return failures


def check_tracked_files(repo_root: Path) -> list[str]:
    """Return failure messages for sensitive files already tracked in git."""
    result = run(["git", "ls-files"], cwd=repo_root)
    if result.returncode != 0:
        return [f"git ls-files failed: {result.stderr.strip()}"]

    tracked = result.stdout.splitlines()
    failures = []
    for path_str in tracked:
        for pat in MUST_NOT_BE_TRACKED_PATTERNS:
            if pat.search(path_str):
                failures.append(f"Sensitive/generated file is tracked: {path_str}")
                break
    return failures


def check_home_paths(repo_root: Path) -> list[str]:
    """Return warning messages for tracked text files containing /home/<user>/ paths."""
    result = run(["git", "ls-files"], cwd=repo_root)
    if result.returncode != 0:
        return []

    tracked = result.stdout.splitlines()
    warnings = []
    for rel_str in tracked:
        if rel_str in SCAN_EXCLUDES:
            continue
        p = repo_root / rel_str
        if p.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if not p.is_file():
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue

        lines_with_home = [
            (i + 1, line.rstrip())
            for i, line in enumerate(text.splitlines())
            if HOME_PATH_RE.search(line)
        ]
        if lines_with_home:
            for lineno, line in lines_with_home:
                warnings.append(
                    f"Participant-specific path in tracked file "
                    f"{rel_str}:{lineno}: {line[:120]}"
                )
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Repository hygiene check")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Path to repository root (default: parent of this script's directory)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat /home/<user> warnings as failures",
    )
    args = parser.parse_args()

    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
    else:
        repo_root = Path(__file__).resolve().parent.parent

    print(f"Repository root: {repo_root}\n")

    all_ok = True

    # --- Check 1: .gitignore coverage ---
    print("=== Check 1: .gitignore coverage ===")
    gitignore_failures = check_gitignore(repo_root)
    if gitignore_failures:
        for msg in gitignore_failures:
            print(f"  FAIL  {msg}")
        all_ok = False
    else:
        print("  PASS  all expected paths are covered by .gitignore")

    # --- Check 2: tracked file scan ---
    print("\n=== Check 2: no sensitive/generated files tracked ===")
    tracked_failures = check_tracked_files(repo_root)
    if tracked_failures:
        for msg in tracked_failures:
            print(f"  FAIL  {msg}")
        all_ok = False
    else:
        print("  PASS  no sensitive/generated files found in git ls-files")

    # --- Check 3: participant-specific absolute paths ---
    print("\n=== Check 3: no participant-specific /home/<user>/ paths in tracked files ===")
    home_warnings = check_home_paths(repo_root)
    if home_warnings:
        level = "FAIL" if args.strict else "WARN"
        for msg in home_warnings:
            print(f"  {level}  {msg}")
        if args.strict:
            all_ok = False
    else:
        print("  PASS  no /home/<user>/ paths found in tracked text files")

    print()
    if all_ok:
        print("All hygiene checks passed.")
        return 0
    else:
        print("One or more hygiene checks FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
