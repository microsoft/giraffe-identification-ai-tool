#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""Source-neutral entry point for normalized elephant crop-quality pilots."""

from pipeline.bteh_crop_pilot import *  # noqa: F403
from pipeline.bteh_crop_pilot import main


if __name__ == "__main__":
    raise SystemExit(main())
