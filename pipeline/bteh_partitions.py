#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""Backward-compatible BTEH entry point for shared crop partition logic."""

import sys

from configs.config_bteh import (
    ARTIFACT_VERSION_ROOT,
    CROPS_SUBDIR,
    EMBEDDINGS_SUBDIR_BTEH,
    SPLITS_FILENAME,
    SPLITS_SUBDIR,
)
from pipeline.elephant_partitions import (
    PARTITION_SPLITS,
    build_crop_partitions,
    main as _shared_main,
)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    def _has_option(option: str) -> bool:
        return option in args or any(
            argument.startswith(f"{option}=") for argument in args
        )

    defaults = {
        "--crop-manifest": str(
            ARTIFACT_VERSION_ROOT / CROPS_SUBDIR / "crop_manifest.parquet"
        ),
        "--splits": str(
            ARTIFACT_VERSION_ROOT / SPLITS_SUBDIR / SPLITS_FILENAME
        ),
        "--output-root": str(
            ARTIFACT_VERSION_ROOT / EMBEDDINGS_SUBDIR_BTEH
        ),
    }
    for option, value in defaults.items():
        if not _has_option(option):
            args.extend([option, value])
    return _shared_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
