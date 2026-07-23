# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""Build normalized reference/query crops from canonical elephant splits."""

import argparse
from pathlib import Path

import pandas as pd

from utils.artifact_schema import CROP_MANIFEST_COLUMNS
from pipeline.elephant_splits import _fingerprint_df


PARTITION_SPLITS = {
    "reference": {"gallery", "held_out_gallery"},
    "query": {"probe", "held_out_probe"},
}


def build_crop_partitions(
    crop_manifest: pd.DataFrame,
    splits: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    required_split_columns = {
        "image_id",
        "split",
        "split_protocol",
        "fold",
        "evaluable",
    }
    missing_crops = set(CROP_MANIFEST_COLUMNS) - set(crop_manifest.columns)
    missing_splits = required_split_columns - set(splits.columns)
    if missing_crops:
        raise ValueError(f"crop manifest is missing columns: {sorted(missing_crops)}")
    if missing_splits:
        raise ValueError(f"split manifest is missing columns: {sorted(missing_splits)}")
    if splits["image_id"].duplicated().any():
        raise ValueError("split manifest image_id must be unique")

    split_meta = splits[list(required_split_columns)]
    merged = crop_manifest.merge(
        split_meta,
        on="image_id",
        how="left",
        validate="many_to_one",
    )
    if merged["split"].isna().any():
        missing = sorted(merged.loc[merged["split"].isna(), "image_id"].unique())
        raise ValueError(f"crop rows are missing split assignments: {missing[:10]}")

    source_fingerprints = set(
        merged["source_fingerprint"].dropna().astype(str)
    )
    split_fingerprints = set(
        merged["split_fingerprint"].dropna().astype(str)
    )
    if len(source_fingerprints) != 1 or len(split_fingerprints) != 1:
        raise ValueError(
            "crop manifest must contain exactly one source and split fingerprint"
        )
    actual_split_fingerprint = _fingerprint_df(splits)
    crop_split_fingerprint = next(iter(split_fingerprints))
    if crop_split_fingerprint != actual_split_fingerprint:
        raise ValueError(
            "crop manifest split_fingerprint does not match the supplied "
            f"splits file: crop={crop_split_fingerprint!r}, "
            f"splits={actual_split_fingerprint!r}"
        )

    partitions = {
        name: merged[merged["split"].isin(labels)].copy()
        for name, labels in PARTITION_SPLITS.items()
    }
    assigned = set().union(
        *(set(frame["image_id"].astype(str)) for frame in partitions.values())
    )
    expected = set(merged["image_id"].astype(str))
    if assigned != expected:
        missing = sorted(expected - assigned)
        raise ValueError(f"active crop images were not partitioned: {missing[:10]}")
    return partitions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build reference/query crops from normalized elephant splits"
    )
    parser.add_argument(
        "--crop-manifest",
        required=True,
    )
    parser.add_argument(
        "--splits",
        required=True,
    )
    parser.add_argument(
        "--output-root",
        required=True,
    )
    args = parser.parse_args(argv)

    crops = pd.read_parquet(args.crop_manifest)
    splits = pd.read_parquet(args.splits)
    partitions = build_crop_partitions(crops, splits)
    output_root = Path(args.output_root)
    for name, frame in partitions.items():
        output = output_root / name / "crop_manifest.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output, index=False)
        print(
            f"{name}: {frame['image_id'].nunique()} images, "
            f"{(frame['detector_status'] == 'accepted').sum()} accepted crops -> {output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
