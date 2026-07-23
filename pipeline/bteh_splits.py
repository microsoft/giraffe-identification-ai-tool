#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""Backward-compatible BTEH entry point for shared elephant split logic."""

import sys

from configs.config_bteh import (
    ARTIFACT_VERSION_ROOT,
    MANIFEST_FILENAME,
    MANIFEST_SUBDIR,
    SPLITS_FILENAME,
    SPLITS_SUBDIR,
)
from pipeline.elephant_splits import (
    DEFAULT_MIN_SESSIONS_TEMPORAL,
    DEFAULT_N_UNSEEN_FOLDS,
    MIN_SESSIONS_FOR_UNSEEN_EVAL,
    _fingerprint_df,
    _validate_no_cross_split_duplicates,
    generate_splits,
    main as _shared_main,
    validate_splits,
)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    def _has_option(option: str) -> bool:
        return option in args or any(
            argument.startswith(f"{option}=") for argument in args
        )

    if not _has_option("--manifest"):
        args.extend(
            [
                "--manifest",
                str(
                    ARTIFACT_VERSION_ROOT
                    / MANIFEST_SUBDIR
                    / MANIFEST_FILENAME
                ),
            ]
        )
    if not _has_option("--output"):
        args.extend(
            [
                "--output",
                str(
                    ARTIFACT_VERSION_ROOT
                    / SPLITS_SUBDIR
                    / SPLITS_FILENAME
                ),
            ]
        )
    return _shared_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
