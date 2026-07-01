# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Import elephant data from the local inventory spreadsheet into the pipeline's
reference_dir / query_dir structure.

Usage:
    python pipeline/import_from_inventory.py \
        --xlsx data/elephant_image_embedding_inventory.xlsx \
        [--images-root data/elephant-images]

Outputs:
    {root_dir}/reference_dir/metadata_reference.csv   — labeled individuals
    {root_dir}/query_dir/metadata_query.csv            — unlabeled / reviewing
"""

import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from dotenv import load_dotenv

from configs.config_elephant import ID_COL, IMAGE_ID_COL, VIEWPOINT_COL
from utils.helpers_matching import load_data_dirs

load_dotenv()


def extract_image_id(image_filename: str) -> str:
    """UUID prefix before the first underscore."""
    return image_filename.split("_")[0] if "_" in image_filename else image_filename


def build_metadata(df: pd.DataFrame, images_root: str, root_dir: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        blob_path = row.get("blob_path", "")
        if not isinstance(blob_path, str) or not blob_path:
            continue

        local_abs = os.path.join(images_root, blob_path)
        if not os.path.isfile(local_abs):
            print(f"  Warning: file not found, skipping: {local_abs}")
            continue

        # path_relative_to_root is relative to root_dir
        path_rel = os.path.relpath(local_abs, root_dir)

        image_filename = row.get("image_filename", os.path.basename(blob_path))
        image_id = extract_image_id(str(image_filename))

        rows.append({
            IMAGE_ID_COL: image_id,
            "path_relative_to_root": path_rel,
            ID_COL: row.get("individual_id", None),
            "individual_name": row.get("individual_name", None),
            VIEWPOINT_COL: "unknown",
            "ai_found_torso": False,
            "image_status": row.get("image_status", None),
            "image_confidence": row.get("image_confidence", None),
            "sex": row.get("sex", None),
            "estimated_age": row.get("estimated_age", None),
            "herd": row.get("herd", None),
            "markings": row.get("markings", None),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Import inventory xlsx into pipeline metadata CSVs")
    parser.add_argument("--xlsx", default="data/elephant_image_embedding_inventory.xlsx")
    parser.add_argument("--images-root", default="data/elephant-images",
                        help="Absolute or relative path to the root of local image files")
    args = parser.parse_args()

    root_dir, _ = load_data_dirs()
    images_root = os.path.abspath(args.images_root)
    xlsx_path = os.path.abspath(args.xlsx)

    print(f"root_dir        : {root_dir}")
    print(f"images_root     : {images_root}")
    print(f"inventory xlsx  : {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    print(f"\nInventory rows  : {len(df)}")

    labeled   = df[df["individual_id"].notna()].copy()
    unlabeled = df[df["individual_id"].isna()].copy()
    print(f"Labeled (ref)   : {len(labeled)} images, {labeled['individual_id'].nunique()} individuals")
    print(f"Unlabeled (query): {len(unlabeled)} images")

    # Build metadata dataframes
    ref_meta   = build_metadata(labeled,   images_root, root_dir)
    query_meta = build_metadata(unlabeled, images_root, root_dir)

    # Write reference metadata
    ref_dir = os.path.join(root_dir, "reference_dir")
    os.makedirs(ref_dir, exist_ok=True)
    ref_csv = os.path.join(ref_dir, "metadata_reference.csv")
    ref_meta.to_csv(ref_csv, index=False)
    print(f"\nReference metadata → {ref_csv}  ({len(ref_meta)} rows)")
    print(ref_meta.groupby(ID_COL).size().sort_values(ascending=False).to_string())

    # Write query metadata
    query_dir = os.path.join(root_dir, "query_dir")
    os.makedirs(query_dir, exist_ok=True)
    query_csv = os.path.join(query_dir, "metadata_query.csv")
    query_meta.to_csv(query_csv, index=False)
    print(f"\nQuery metadata → {query_csv}  ({len(query_meta)} rows)")
    print(f"image_status breakdown:\n{query_meta['image_status'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
