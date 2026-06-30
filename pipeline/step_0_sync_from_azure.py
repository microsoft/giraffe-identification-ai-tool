# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
import pstats
import cProfile
import argparse
import itertools

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from azure.blob_client import ElephantBlobClient
from azure.cosmos_client import GaneshaCosmosClient
from configs.config_elephant import ID_COL, IMAGE_ID_COL, VIEWPOINT_COL, VIEWPOINT_VALUES
from utils.helpers_matching import load_data_dirs, print_memory_usage


def _sync_blobs(blob_client, cosmos_client, image_dir, root_dir, partition, limit, dry_run):
    individuals_df = cosmos_client.fetch_individuals()  # noqa: F841 — available for future join enrichment
    inventory_df = cosmos_client.fetch_image_inventory()

    if inventory_df.empty:
        print("Warning: image inventory from Cosmos DB is empty.")
        return pd.DataFrame(
            columns=[IMAGE_ID_COL, "path_relative_to_root", ID_COL, VIEWPOINT_COL, "ai_found_torso"]
        )

    blob_set = set(blob_client.list_blobs())

    partition_dir = os.path.join(root_dir, partition + "_dir")
    checkpoint_path = os.path.join(partition_dir, "metadata_" + partition + ".csv")

    rows = []
    downloaded = 0
    skipped = 0

    item_iter = inventory_df.iterrows()
    total = len(inventory_df)
    if limit and limit > 0:
        item_iter = itertools.islice(item_iter, limit)
        total = min(total, limit)

    for count, (_, inv_row) in enumerate(item_iter, start=1):
        blob_path = inv_row.get("blob_path", "")
        image_id = inv_row.get(IMAGE_ID_COL, "")
        individual_id = inv_row.get(ID_COL, "")

        if not blob_path:
            print(f"  Skipping row {count}: missing blob_path (image_id={image_id})")
            skipped += 1
            continue

        filename = os.path.basename(blob_path)
        dest_abs = os.path.join(image_dir, filename)
        path_relative_to_root = os.path.relpath(dest_abs, root_dir)

        already_exists = os.path.isfile(dest_abs) and os.path.getsize(dest_abs) > 0

        if already_exists:
            skipped += 1
        elif dry_run:
            print(f"  [dry-run] Would download: {blob_path} -> {dest_abs}")
            downloaded += 1
        else:
            if blob_path not in blob_set:
                print(f"  Warning: blob not found in storage: {blob_path}")
                skipped += 1
                continue
            try:
                blob_client.download(blob_path, dest_abs)
                downloaded += 1
            except Exception as exc:
                print(f"  Error downloading {blob_path}: {exc}")
                skipped += 1
                continue

        rows.append(
            {
                IMAGE_ID_COL: image_id,
                "path_relative_to_root": path_relative_to_root,
                ID_COL: individual_id,
                VIEWPOINT_COL: "unknown",
                "ai_found_torso": False,
            }
        )

        if count % 100 == 0:
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
            print(f"  [{count}/{total}] downloaded={downloaded} skipped={skipped} (checkpoint saved)")

    print(f"\nSync complete: downloaded={downloaded}, skipped={skipped}, total_rows={len(rows)}")
    return pd.DataFrame(
        rows,
        columns=[IMAGE_ID_COL, "path_relative_to_root", ID_COL, VIEWPOINT_COL, "ai_found_torso"],
    )


def main(partition, dry_run, limit):
    profiler = cProfile.Profile()
    profiler.enable()

    root_dir, _ = load_data_dirs()

    partition_dir = os.path.join(root_dir, partition + "_dir")
    image_dir = os.path.join(partition_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    cosmos_client = GaneshaCosmosClient.from_env()
    blob_client = ElephantBlobClient.from_env()

    metadata_df = _sync_blobs(blob_client, cosmos_client, image_dir, root_dir, partition, limit, dry_run)

    csv_path = os.path.join(partition_dir, "metadata_" + partition + ".csv")
    metadata_df.to_csv(csv_path, index=False)
    print(f"\nMetadata CSV saved to: {csv_path}")
    print(f"Total images in CSV: {len(metadata_df)}")

    if not metadata_df.empty and ID_COL in metadata_df.columns:
        print("\nImages per individual:")
        print(metadata_df[ID_COL].value_counts().to_string())

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats()

    print_memory_usage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pull elephant images + metadata from Azure into local dirs (step 0)"
    )
    parser.add_argument(
        "--partition", type=str, default="reference", help="Partition name (default: reference)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List blobs to download without actually downloading"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max images to process (0 = no limit, useful for testing)"
    )
    args = parser.parse_args()
    main(args.partition, args.dry_run, args.limit)
