# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

import os
import sys
import json
import pickle
import pstats
import cProfile
import argparse
import logging
import numpy as np
import pandas as pd
import faiss
import cv2
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers_matching import load_data_dirs, load_metadata_file
from utils.helpers_matching import print_memory_usage, log_to_file, restore_stdout
from configs.config_elephant import (
    ACTIVE_DESCRIPTORS,
    EAR_DESCRIPTORS,
    EMBEDDINGS_SUBDIR,
    FAISS_SUBDIR,
    INDEX_PARQUET_FILENAME,
    CALIBRATION_DIR,
    LOCAL_MATCHER_BACKEND,
    LOCAL_MATCHER_KEYPOINTS,
    LOCAL_MATCHER_MIN_INLIERS,
    NUM_RECOMMENDED_IDS,
    MATCH_ACCEPT_THRESHOLD,
)
from configs.config_artifacts import ARTIFACT_SCHEMA_VERSION
from models.embedder import GlobalEmbedder
from models.local_matcher import LocalMatcher
from models.calibration import Calibrator
from models.fusion import WildFusionMatcher, Recommendation
from utils.artifact_schema import (
    CROP_MANIFEST_COLUMNS,
    DESCRIPTOR_MAPPING_COLUMNS,
    assert_descriptor_mapping_integrity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy giraffe WildFusionMatcher factory
# Used ONLY when --legacy-giraffe is passed explicitly.
# ---------------------------------------------------------------------------

def _build_wildfusion_legacy(root_dir: str, skip_local: bool = False) -> WildFusionMatcher:
    """Build a WildFusionMatcher using the old pickle-based reference metadata.

    This function is intentionally kept separate from the BTEH normalized route
    and is only reachable via --legacy-giraffe.  It must not be called from the
    BTEH code path.
    """
    reference_dir = os.path.join(root_dir, "reference_dir")
    faiss_dir     = os.path.join(reference_dir, FAISS_SUBDIR)
    calib_dir     = os.path.join(root_dir, CALIBRATION_DIR)

    embedders = {}

    # FAISS indexes and ref_meta (legacy pickle format)
    faiss_indexes = {}
    ref_meta      = {}
    for desc in ACTIVE_DESCRIPTORS:
        index_path = os.path.join(faiss_dir, f"{desc}.index")
        if not os.path.isfile(index_path):
            logger.error("FAISS index not found: %s  — run step_2 for reference first.", index_path)
            sys.exit(1)
        faiss_indexes[desc] = faiss.read_index(index_path)

        meta_path = os.path.join(faiss_dir, f"reference_{desc}_meta.pkl")
        if not os.path.isfile(meta_path):
            logger.error("FAISS meta not found: %s", meta_path)
            sys.exit(1)
        with open(meta_path, "rb") as fh:
            ref_meta[desc] = pickle.load(fh)

    local_matcher = None if skip_local else LocalMatcher(
        backend=LOCAL_MATCHER_BACKEND,
        max_keypoints=LOCAL_MATCHER_KEYPOINTS,
        min_inliers=LOCAL_MATCHER_MIN_INLIERS,
    )

    calibrators = {}
    for name in list(ACTIVE_DESCRIPTORS) + ["local"]:
        cal_path = os.path.join(calib_dir, f"{name}.pkl")
        if os.path.isfile(cal_path):
            cal = Calibrator().load(cal_path)
            calibrators[name] = cal
            logger.info("Loaded calibrator '%s' (method=%s).", name, cal.method)

    return WildFusionMatcher(
        embedders=embedders,
        faiss_indexes=faiss_indexes,
        ref_meta=ref_meta,
        local_matcher=local_matcher,
        calibrators=calibrators,
        skip_local=skip_local,
    )


# ---------------------------------------------------------------------------
# BTEH normalized step_3 route
# ---------------------------------------------------------------------------

def _load_bteh_reference(
    artifact_dir: str,
    descriptor_names: list[str],
    *,
    schema_version: str | None = None,
    expected_source_fingerprint: str | None = None,
    expected_split_fingerprint: str | None = None,
) -> tuple[dict, dict, dict]:
    """Load and validate reference descriptor mappings, FAISS indexes.

    Returns
    -------
    faiss_indexes  : {desc: faiss.Index}
    ref_mappings   : {desc: pd.DataFrame} — normalized descriptor mapping
    ref_matrices   : {desc: np.ndarray}   — embedding matrix (for integrity check only)
    """
    faiss_indexes: dict = {}
    ref_mappings: dict = {}
    ref_matrices: dict = {}

    for desc in descriptor_names:
        mapping_path = os.path.join(artifact_dir, f"{desc}_mapping.parquet")
        matrix_path  = os.path.join(artifact_dir, f"{desc}.npy")
        index_path   = os.path.join(artifact_dir, f"{desc}.index")

        for label, path in (
            ("mapping parquet", mapping_path),
            ("embedding matrix", matrix_path),
            ("FAISS index", index_path),
        ):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"BTEH reference {label} for descriptor {desc!r} not found: {path}"
                )

        mapping_df = pd.read_parquet(mapping_path)
        matrix     = np.load(matrix_path).astype(np.float32)
        index      = faiss.read_index(index_path)

        # Validate schema/fingerprints/vector integrity and FAISS ntotal.
        assert_descriptor_mapping_integrity(
            mapping_df,
            matrix,
            index,
            is_reference=True,
            schema_version=schema_version,
            expected_source_fingerprint=expected_source_fingerprint,
            expected_split_fingerprint=expected_split_fingerprint,
        )

        # Use the mapping DataFrame as ref_meta so WildFusionMatcher can do
        # exact faiss_row → (individual_id, image_id, crop_path, viewpoint) lookups.
        faiss_indexes[desc] = index
        ref_mappings[desc]  = mapping_df
        ref_matrices[desc]  = matrix
        logger.info(
            "Loaded reference descriptor %r: %d rows, FAISS ntotal=%d",
            desc, len(mapping_df), index.ntotal,
        )

    return faiss_indexes, ref_mappings, ref_matrices


def _single_mapping_fingerprint(
    mapping: pd.DataFrame,
    column: str,
    descriptor_name: str,
) -> str | None:
    values = {
        str(value)
        for value in mapping[column].dropna().unique()
        if str(value)
    }
    if len(values) > 1:
        raise ValueError(
            f"descriptor {descriptor_name!r} has multiple {column} values: "
            f"{sorted(values)}"
        )
    return next(iter(values), None)


def _build_bteh_query_records(
    query_mapping_df: pd.DataFrame,
    query_matrix: np.ndarray,
    descriptor_name: str,
) -> list[dict]:
    """Build query descriptor records for one descriptor.

    Returns a list of dicts each containing:
      embedding_row  : row index into query_matrix
      crop_id        : stable crop ID
      image_id       : parent image ID
      individual_id  : identity (for evaluation; may be empty for true unknowns)
      crop_kind      : 'body' or 'ear'
      crop_path      : absolute path to the crop image
      embedding      : the actual query vector (embedded in the record so
                       WildFusionMatcher._query_vector_from_mapping can use it)
    """
    records = []
    for _, row in query_mapping_df.iterrows():
        emb_row = int(row["embedding_row"])
        if emb_row < 0 or emb_row >= len(query_matrix):
            raise AssertionError(
                f"query descriptor {descriptor_name!r} embedding_row={emb_row} "
                f"is out of range for matrix length {len(query_matrix)}"
            )
        records.append(
            {
                "descriptor_name": descriptor_name,
                "embedding_row":   emb_row,
                "crop_id":         str(row["crop_id"]),
                "image_id":        str(row["image_id"]),
                "individual_id":   str(row.get("individual_id", "")),
                "crop_kind":       str(row["crop_kind"]),
                "crop_path":       str(row.get("crop_path", "")),
                "embedding":       query_matrix[emb_row].astype(np.float32),
            }
        )
    return records


def _group_query_records_by_image(
    all_query_records: dict[str, list[dict]],
) -> dict[str, dict[str, list[dict]]]:
    """Group query records by image_id.

    Returns {image_id: {desc: [records for that image and descriptor]}}
    """
    grouped: dict[str, dict[str, list[dict]]] = {}
    for desc, records in all_query_records.items():
        for rec in records:
            img_id = rec["image_id"]
            grouped.setdefault(img_id, {}).setdefault(desc, []).append(rec)
    return grouped


def run_bteh_step3_normalized(
    query_artifact_dir: str,
    reference_artifact_dir: str,
    output_path: str,
    *,
    descriptor_names: list[str] | None = None,
    calibration_dir: str | None = None,
    skip_local: bool = True,
    schema_version: str | None = None,
    expected_source_fingerprint: str | None = None,
    expected_split_fingerprint: str | None = None,
    query_crop_manifest_path: str | None = None,
) -> pd.DataFrame:
    """Normalized BTEH step-3 matching.

    Loads one mapping parquet + matrix per descriptor, validates
    schema/fingerprints/vector integrity and reference FAISS ntotal.
    Builds zero-or-more query descriptor records per image (body 0..1,
    ears 0..2) using each descriptor's own embedding rows.  Searches all
    available query ears against the reference and aggregates the best
    compatible ear evidence without anatomical side assumptions.

    Does NOT use:
      - megadescriptor_row or shared positional index parquet
      - filename-stem IDs
      - *_cropped_torso_zoomed fallback paths
      - legacy pickle reference metadata

    Parameters
    ----------
    query_artifact_dir      : directory containing query {desc}_mapping.parquet + {desc}.npy
    reference_artifact_dir  : directory containing reference {desc}_mapping.parquet + {desc}.npy + {desc}.index
    output_path             : where to write the results parquet
    descriptor_names        : list of descriptor names to match; defaults to ACTIVE_DESCRIPTORS
    calibration_dir         : optional directory with {desc}.pkl calibrators
    skip_local              : skip LightGlue local re-ranking (fast mode)
    schema_version          : expected artifact schema version for validation
    expected_source_fingerprint : if set, all artifacts must carry this value
    expected_split_fingerprint  : if set, all artifacts must carry this value
    """
    _schema = schema_version or ARTIFACT_SCHEMA_VERSION

    if descriptor_names is None:
        descriptor_names = list(ACTIVE_DESCRIPTORS)

    # ------------------------------------------------------------------
    # Load and validate reference artifacts
    # ------------------------------------------------------------------
    faiss_indexes, ref_mappings, _ = _load_bteh_reference(
        reference_artifact_dir,
        descriptor_names,
        schema_version=_schema,
        expected_source_fingerprint=expected_source_fingerprint,
        expected_split_fingerprint=expected_split_fingerprint,
    )
    empty_reference_channels = [
        desc for desc, mapping in ref_mappings.items() if mapping.empty
    ]
    for desc in empty_reference_channels:
        logger.info(
            "Reference descriptor %r has no rows; skipping channel.",
            desc,
        )
        faiss_indexes.pop(desc, None)
        ref_mappings.pop(desc, None)
    if not ref_mappings:
        raise RuntimeError(
            "No non-empty reference descriptor artifacts found in: "
            + reference_artifact_dir
        )
    reference_fingerprints = {
        desc: {
            column: _single_mapping_fingerprint(mapping, column, desc)
            for column in (
                "source_fingerprint",
                "split_fingerprint",
                "model_preprocess_fingerprint",
            )
        }
        for desc, mapping in ref_mappings.items()
    }
    for column in ("source_fingerprint", "split_fingerprint"):
        values = {
            fingerprints[column]
            for fingerprints in reference_fingerprints.values()
        }
        if len(values) > 1:
            raise ValueError(
                f"reference descriptor channels mix {column} values: "
                f"{sorted(map(str, values))}"
            )
    reference_run_fingerprints = {
        column: next(
            iter(
                {
                    fingerprints[column]
                    for fingerprints in reference_fingerprints.values()
                }
            )
        )
        for column in ("source_fingerprint", "split_fingerprint")
    }

    # ------------------------------------------------------------------
    # Load and validate query artifacts per descriptor
    # ------------------------------------------------------------------
    all_query_records: dict[str, list[dict]] = {}
    for desc in descriptor_names:
        if desc not in ref_mappings:
            continue
        q_mapping_path = os.path.join(query_artifact_dir, f"{desc}_mapping.parquet")
        q_matrix_path  = os.path.join(query_artifact_dir, f"{desc}.npy")
        if not os.path.isfile(q_mapping_path) or not os.path.isfile(q_matrix_path):
            logger.info(
                "Query artifacts for descriptor %r not found; skipping channel.", desc
            )
            continue
        q_mapping = pd.read_parquet(q_mapping_path)
        q_matrix  = np.load(q_matrix_path).astype(np.float32)
        if q_mapping.empty:
            assert_descriptor_mapping_integrity(
                q_mapping,
                q_matrix,
                None,
                is_reference=False,
                schema_version=_schema,
            )
            logger.info(
                "Query descriptor %r has no rows; skipping channel.",
                desc,
            )
            continue
        for column, reference_value in reference_fingerprints[desc].items():
            query_value = _single_mapping_fingerprint(
                q_mapping,
                column,
                desc,
            )
            if query_value != reference_value:
                raise ValueError(
                    f"query/reference {column} mismatch for descriptor "
                    f"{desc!r}: query={query_value!r}, "
                    f"reference={reference_value!r}"
                )
        # Validate query artifacts (no FAISS index for queries).
        assert_descriptor_mapping_integrity(
            q_mapping,
            q_matrix,
            None,
            is_reference=False,
            schema_version=_schema,
            expected_source_fingerprint=expected_source_fingerprint,
            expected_split_fingerprint=expected_split_fingerprint,
            expected_model_fingerprint=reference_fingerprints[desc][
                "model_preprocess_fingerprint"
            ],
        )
        all_query_records[desc] = _build_bteh_query_records(q_mapping, q_matrix, desc)
        logger.info(
            "Loaded query descriptor %r: %d records", desc, len(all_query_records[desc])
        )

    if not all_query_records and query_crop_manifest_path is None:
        raise RuntimeError(
            "No query descriptor artifacts found in: " + query_artifact_dir
        )

    # ------------------------------------------------------------------
    # Load calibrators (optional)
    # ------------------------------------------------------------------
    calibrators: dict = {}
    if calibration_dir:
        for name in list(descriptor_names) + ["local"]:
            cal_path = os.path.join(calibration_dir, f"{name}.pkl")
            if os.path.isfile(cal_path):
                cal = Calibrator().load(cal_path)
                calibrators[name] = cal
                logger.info("Loaded calibrator '%s' (method=%s).", name, cal.method)

    # ------------------------------------------------------------------
    # Build WildFusionMatcher using normalized ref_meta (DataFrames, not
    # positional lists) and query_embedding_matrices.
    # ------------------------------------------------------------------
    local_matcher = (
        None
        if skip_local
        else LocalMatcher(
            backend=LOCAL_MATCHER_BACKEND,
            max_keypoints=LOCAL_MATCHER_KEYPOINTS,
            min_inliers=LOCAL_MATCHER_MIN_INLIERS,
        )
    )

    matcher = WildFusionMatcher(
        embedders={},
        faiss_indexes=faiss_indexes,
        ref_meta=ref_mappings,   # DataFrames → _mapping_meta uses column-based lookup
        local_matcher=local_matcher,
        calibrators=calibrators,
        skip_local=skip_local,
    )

    # ------------------------------------------------------------------
    # Run matching: one identify_from_mappings call per image.
    # All available query ear records are passed; the matcher aggregates
    # best compatible ear evidence without anatomical side assumptions.
    # ------------------------------------------------------------------
    grouped = _group_query_records_by_image(all_query_records)
    result_rows = []

    for image_id, desc_records in tqdm(grouped.items(), desc="BTEH matching"):
        # Determine the query crop_path from the body descriptor if present;
        # fall back to the first available crop for local matching.
        query_crop_path = ""
        body_records = [
            rec
            for recs in desc_records.values()
            for rec in recs
            if rec.get("crop_kind") == "body"
        ]
        if body_records:
            query_crop_path = body_records[0].get("crop_path", "")

        query_crop_bgr = None
        if not skip_local and query_crop_path and os.path.isfile(query_crop_path):
            query_crop_bgr = cv2.imread(query_crop_path)
            if query_crop_bgr is None:
                logger.debug("Could not read query crop: %s", query_crop_path)

        recommendations = matcher.identify_from_mappings(desc_records, query_crop_bgr)

        individual_id = next(
            (
                rec["individual_id"]
                for recs in desc_records.values()
                for rec in recs
                if rec.get("individual_id")
            ),
            "",
        )

        row: dict = {
            "image_id":      image_id,
            "individual_id": individual_id,
            "query_crop_path": query_crop_path,
            "matching_status": (
                "matched"
                if recommendations and recommendations[0].fused_sim >= MATCH_ACCEPT_THRESHOLD
                else "not_matched"
            ),
        }
        for rank, rec in enumerate(recommendations[:NUM_RECOMMENDED_IDS], start=1):
            row[f"match_individual_{rank}"] = rec.individual_id
            row[f"match_image_{rank}"]      = rec.image_id
            row[f"match_fused_sim_{rank}"]  = rec.fused_sim
            row[f"match_local_inliers_{rank}"] = rec.local_inliers
            for desc, sim in rec.global_sims.items():
                row[f"match_{desc}_sim_{rank}"] = sim
        result_rows.append(row)

    if query_crop_manifest_path is not None:
        query_crops = pd.read_parquet(query_crop_manifest_path)
        required = set(CROP_MANIFEST_COLUMNS)
        missing = required - set(query_crops.columns)
        if missing:
            raise ValueError(
                "query crop manifest is missing columns: "
                f"{sorted(missing)}"
            )
        schema_value = _single_mapping_fingerprint(
            query_crops,
            "schema_version",
            "query crop manifest",
        )
        if schema_value != _schema:
            raise ValueError(
                "query crop manifest schema_version mismatch: "
                f"expected {_schema!r}, found {schema_value!r}"
            )
        for column, reference_value in reference_run_fingerprints.items():
            crop_value = _single_mapping_fingerprint(
                query_crops,
                column,
                "query crop manifest",
            )
            if crop_value != reference_value:
                raise ValueError(
                    f"query crop manifest {column} mismatch: "
                    f"crop={crop_value!r}, reference={reference_value!r}"
                )
        identities_per_image = (
            query_crops.groupby("image_id")["individual_id"].nunique(
                dropna=False
            )
        )
        ambiguous = identities_per_image[identities_per_image > 1]
        if not ambiguous.empty:
            raise ValueError(
                "query crop manifest has conflicting identities for image IDs: "
                f"{ambiguous.index.astype(str).tolist()[:10]}"
            )
        query_identities = (
            query_crops.drop_duplicates("image_id")
            .set_index("image_id")["individual_id"]
            .to_dict()
        )
        missing_descriptor_crops: dict[str, list[str]] = {}
        for desc in ref_mappings:
            crop_kind = "ear" if desc in EAR_DESCRIPTORS else "body"
            expected_crop_ids = {
                str(crop_id)
                for crop_id in query_crops.loc[
                    query_crops["detector_status"].eq("accepted")
                    & query_crops["crop_kind"].eq(crop_kind),
                    "crop_id",
                ]
            }
            represented_crop_ids = {
                str(record["crop_id"])
                for record in all_query_records.get(desc, [])
            }
            missing = sorted(expected_crop_ids - represented_crop_ids)
            if missing:
                missing_descriptor_crops[desc] = missing[:10]
        if missing_descriptor_crops:
            raise RuntimeError(
                "accepted query crops have no descriptor records: "
                f"{missing_descriptor_crops}"
            )
        matched_ids = {str(row["image_id"]) for row in result_rows}
        for image_id, individual_id in sorted(
            query_identities.items(),
            key=lambda item: str(item[0]),
        ):
            if str(image_id) in matched_ids:
                continue
            result_rows.append(
                {
                    "image_id": str(image_id),
                    "individual_id": str(individual_id),
                    "query_crop_path": "",
                    "matching_status": "not_matched",
                }
            )

    results_df = pd.DataFrame(result_rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    results_df.to_parquet(output_path, index=False)
    logger.info(
        "Normalized matching complete: %d images → %s",
        len(results_df),
        output_path,
    )
    return results_df


run_normalized_step3 = run_bteh_step3_normalized


def _normalized_main_step3(*, use_bteh_defaults: bool = False):
    """CLI entry point for normalized elephant step-3 matching."""
    default_query_dir = None
    default_reference_dir = None
    default_query_crop_manifest = None
    if use_bteh_defaults:
        from configs.config_bteh import (
            ARTIFACT_VERSION_ROOT,
            EMBEDDINGS_SUBDIR_BTEH,
        )
        default_query_dir = str(
            ARTIFACT_VERSION_ROOT / EMBEDDINGS_SUBDIR_BTEH / "query"
        )
        default_reference_dir = str(
            ARTIFACT_VERSION_ROOT / EMBEDDINGS_SUBDIR_BTEH / "reference"
        )
        default_query_crop_manifest = str(
            ARTIFACT_VERSION_ROOT
            / EMBEDDINGS_SUBDIR_BTEH
            / "query"
            / "crop_manifest.parquet"
        )
    parser = argparse.ArgumentParser(
        description=(
            "Normalized elephant matching: load descriptor mappings and "
            "run identify_from_mappings per image."
        )
    )
    parser.add_argument(
        "--query-artifact-dir",
        default=default_query_dir,
        required=not use_bteh_defaults,
        help="Directory containing query {desc}_mapping.parquet + {desc}.npy",
    )
    parser.add_argument(
        "--reference-artifact-dir",
        default=default_reference_dir,
        required=not use_bteh_defaults,
        help="Directory containing reference {desc}_mapping.parquet + {desc}.npy + {desc}.index",
    )
    parser.add_argument(
        "--query-crop-manifest",
        default=default_query_crop_manifest,
        required=not use_bteh_defaults,
        help=(
            "Query crop manifest used to retain images with no accepted crops "
            "as not_matched results."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the output results parquet.",
    )
    parser.add_argument(
        "--calibration-dir",
        default=None,
        help="Optional directory with calibrator pickle files.",
    )
    parser.add_argument(
        "--descriptors",
        nargs="+",
        default=list(ACTIVE_DESCRIPTORS),
        help="Descriptor names to use (default: all ACTIVE_DESCRIPTORS).",
    )
    parser.add_argument(
        "--enable-local",
        action="store_true",
        default=False,
        help="Enable LightGlue local re-ranking (disabled by default).",
    )
    parser.add_argument("--schema-version", default=ARTIFACT_SCHEMA_VERSION)
    parser.add_argument("--source-fingerprint", default=None)
    parser.add_argument("--split-fingerprint", default=None)
    args = parser.parse_args()

    run_bteh_step3_normalized(
        query_artifact_dir=args.query_artifact_dir,
        reference_artifact_dir=args.reference_artifact_dir,
        output_path=args.output,
        descriptor_names=args.descriptors,
        calibration_dir=args.calibration_dir,
        skip_local=not args.enable_local,
        schema_version=args.schema_version,
        expected_source_fingerprint=args.source_fingerprint,
        expected_split_fingerprint=args.split_fingerprint,
        query_crop_manifest_path=args.query_crop_manifest,
    )


def _bteh_main_step3():
    _normalized_main_step3(use_bteh_defaults=True)



# ---------------------------------------------------------------------------
# Metadata column management
# ---------------------------------------------------------------------------

def _viz_payload_to_json(viz_payload: dict) -> str:
    """Serialize viz_payload (numpy arrays) to a JSON string for CSV storage."""
    if not viz_payload:
        return ""
    try:
        serializable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in viz_payload.items()
        }
        return json.dumps(serializable)
    except Exception:
        return ""


def add_columns_for_matching_results(query_metadata: pd.DataFrame) -> pd.DataFrame:
    cols = ["matching_attempt", "matching_status"]
    for i in range(1, NUM_RECOMMENDED_IDS + 1):
        cols += [
            f"match_individual_{i}",
            f"match_image_{i}",
            f"match_viewpoint_{i}",
            f"match_global_sim_{i}",
            f"match_local_count_{i}",
            f"match_local_sim_{i}",
            f"match_fused_sim_{i}",
            f"viz_payload_{i}",
        ]
        for desc in ACTIVE_DESCRIPTORS:
            cols.append(f"match_{desc}_sim_{i}")
    str_cols = {"matching_attempt", "matching_status"}
    for i in range(1, NUM_RECOMMENDED_IDS + 1):
        str_cols |= {f"match_individual_{i}", f"match_image_{i}", f"match_viewpoint_{i}", f"viz_payload_{i}"}

    for col in cols:
        if col not in query_metadata.columns:
            query_metadata[col] = np.nan
        if col in str_cols:
            query_metadata[col] = query_metadata[col].astype(object)
    return query_metadata


def fill_matching_results(
    query_metadata: pd.DataFrame,
    query_image_path: str,
    recommendations: list,
) -> pd.DataFrame:
    matching_index = query_metadata[query_metadata["path_relative_to_root"] == query_image_path].index

    if matching_index.empty:
        return query_metadata

    status = "not_matched"
    if recommendations and recommendations[0].fused_sim >= MATCH_ACCEPT_THRESHOLD:
        status = "matched"

    query_metadata.loc[matching_index, "matching_status"] = status

    for i, rec in enumerate(recommendations[:NUM_RECOMMENDED_IDS], start=1):
        query_metadata.loc[matching_index, f"match_individual_{i}"] = rec.individual_id
        query_metadata.loc[matching_index, f"match_image_{i}"]      = rec.image_id
        query_metadata.loc[matching_index, f"match_viewpoint_{i}"]  = rec.viewpoint
        global_sim_val = float(np.mean(list(rec.global_sims.values()))) if rec.global_sims else 0.0
        query_metadata.loc[matching_index, f"match_global_sim_{i}"] = global_sim_val
        for desc in ACTIVE_DESCRIPTORS:
            query_metadata.loc[matching_index, f"match_{desc}_sim_{i}"] = rec.global_sims.get(desc, np.nan)
        query_metadata.loc[matching_index, f"match_local_count_{i}"] = rec.local_inliers
        query_metadata.loc[matching_index, f"match_local_sim_{i}"]   = rec.local_sim
        query_metadata.loc[matching_index, f"match_fused_sim_{i}"]   = rec.fused_sim
        query_metadata.loc[matching_index, f"viz_payload_{i}"]       = _viz_payload_to_json(rec.viz_payload)

    return query_metadata


# ---------------------------------------------------------------------------
# Query sweep
# ---------------------------------------------------------------------------

def sweep_over_query_images(
    metadata_filepath: str,
    query_metadata: pd.DataFrame,
    embeddings_per_desc: dict,
    query_index_df: pd.DataFrame,
    wildfusion: WildFusionMatcher,
    skip_local: bool = False,
) -> pd.DataFrame:
    query_metadata = add_columns_for_matching_results(query_metadata)

    # Build a quick lookup: image_id → row-index in each embedding matrix
    img_id_to_row: dict[str, int] = {}
    for _, idx_row in query_index_df.iterrows():
        img_id_to_row[str(idx_row["image_id"])] = int(idx_row.get("megadescriptor_row", idx_row.name))

    for idx, row in tqdm(query_metadata.iterrows(), total=len(query_metadata), desc="Matching"):

        if idx % 100 == 0:
            query_metadata.to_csv(metadata_filepath, index=False)

        query_image_path = row["path_relative_to_root"]
        query_metadata.loc[idx, "matching_attempt"] = "failed"

        # Skip rows that have already been processed
        if "matching_status" in row and row["matching_status"] in {"not_matched", "matched"}:
            query_metadata.loc[idx, "matching_attempt"] = "existing"
            continue

        # Resolve image_id for this row
        img_id = str(row.get("image_id", "")) if "image_id" in row.index else ""
        if not img_id:
            img_id = os.path.splitext(os.path.basename(query_image_path))[0]

        # Look up precomputed embedding row index
        emb_row = img_id_to_row.get(img_id)
        if emb_row is None:
            logger.warning("No embedding row found for image_id='%s'. Skipping.", img_id)
            continue

        # Build per-descriptor query embedding dict from pre-computed matrices
        query_embedding_per_desc = {}
        for desc in ACTIVE_DESCRIPTORS:
            if desc in embeddings_per_desc:
                query_embedding_per_desc[desc] = embeddings_per_desc[desc][emb_row]

        # Resolve crop path for local matcher
        crop_path = ""
        if "crop_path" in row.index and pd.notna(row["crop_path"]):
            crop_path = str(row["crop_path"])
        else:
            # Match the crop path reconstruction used in step_2
            orig_path = query_image_path
            parts = orig_path.rsplit(".", 1)
            img_filename = os.path.basename(orig_path)
            ext = parts[1] if len(parts) == 2 else "jpg"
            stem = img_filename.rsplit(".", 1)[0]
            root_dir, _ = load_data_dirs()
            from configs.config_elephant import CROP_SUBDIR
            crop_path = os.path.join(root_dir, CROP_SUBDIR, "zoomed_version", f"{stem}_cropped_torso_zoomed.{ext}")

        if skip_local:
            query_crop_bgr = None
        else:
            query_crop_bgr = cv2.imread(crop_path) if os.path.isfile(crop_path) else None
            if query_crop_bgr is None:
                logger.debug("Query crop not found for '%s'. Local matching will be skipped.", query_image_path)

        query_metadata.loc[idx, "matching_attempt"] = "success"

        recommendations = wildfusion.identify(query_embedding_per_desc, query_crop_bgr)
        query_metadata = fill_matching_results(query_metadata, query_image_path, recommendations)

    return query_metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Legacy giraffe matching route (--legacy-giraffe mode only)."""
    parser = argparse.ArgumentParser(description="Run WildFusion matching — legacy giraffe mode (step 3 --legacy-giraffe)")
    parser.add_argument(
        "--skip-local",
        action="store_true",
        default=False,
        dest="skip_local",
        help="Skip LightGlue local re-ranking (fast, CPU-safe mode)",
    )
    args = parser.parse_args()

    profiler = cProfile.Profile()
    profiler.enable()

    root_dir, _ = load_data_dirs()

    log_file_std_output, log_file_err_output = log_to_file(root_dir, "matching_algorithm")

    # Load query metadata
    metadata_query_filepath = os.path.join(root_dir, "query_dir", "metadata_query.csv")
    query_metadata = load_metadata_file(metadata_query_filepath)

    # Load pre-computed query embeddings and index parquet
    query_embeddings_dir = os.path.join(root_dir, "query_dir", EMBEDDINGS_SUBDIR)
    embeddings_per_desc: dict[str, np.ndarray] = {}
    for desc in ACTIVE_DESCRIPTORS:
        npy_path = os.path.join(query_embeddings_dir, f"query_{desc}.npy")
        if not os.path.isfile(npy_path):
            logger.error("Query embeddings not found: %s  — run step_2 --partition query first.", npy_path)
            sys.exit(1)
        embeddings_per_desc[desc] = np.load(npy_path)
        logger.info("Loaded query embeddings '%s' shape=%s", desc, embeddings_per_desc[desc].shape)

    query_index_parquet = os.path.join(query_embeddings_dir, f"query_{INDEX_PARQUET_FILENAME}")
    if not os.path.isfile(query_index_parquet):
        logger.error("Query index parquet not found: %s", query_index_parquet)
        sys.exit(1)
    query_index_df = pd.read_parquet(query_index_parquet)

    # Build WildFusionMatcher using legacy pickle-based reference metadata.
    wildfusion = _build_wildfusion_legacy(root_dir, skip_local=args.skip_local)

    # Run matching
    query_metadata = sweep_over_query_images(
        metadata_query_filepath,
        query_metadata,
        embeddings_per_desc,
        query_index_df,
        wildfusion,
        skip_local=args.skip_local,
    )
    query_metadata.to_csv(metadata_query_filepath, index=False)
    logger.info("Matching complete. Results saved to %s", metadata_query_filepath)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats()

    print_memory_usage()

    restore_stdout(log_file_std_output, log_file_err_output)


if __name__ == "__main__":
    _top = argparse.ArgumentParser(add_help=False)
    _top.add_argument("--bteh", action="store_true", default=False)
    _top.add_argument("--normalized", action="store_true", default=False)
    _top.add_argument("--legacy-giraffe", action="store_true", default=False, dest="legacy_giraffe")
    _mode, _remaining = _top.parse_known_args()

    if sum((_mode.bteh, _mode.normalized, _mode.legacy_giraffe)) > 1:
        print(
            "ERROR: --bteh, --normalized, and --legacy-giraffe are mutually exclusive.",
            file=sys.stderr,
        )
        sys.exit(1)

    if _mode.bteh:
        sys.argv = [sys.argv[0]] + _remaining
        _bteh_main_step3()
    elif _mode.normalized:
        sys.argv = [sys.argv[0]] + _remaining
        _normalized_main_step3()
    elif _mode.legacy_giraffe:
        sys.argv = [sys.argv[0]] + _remaining
        main()
    else:
        print(
            "ERROR: Specify --normalized, --bteh, or --legacy-giraffe "
            "(legacy giraffe CSV/pickle route).\n"
            "  Normalized:       python step_3_run_initial_matching.py --normalized --help\n"
            "  Normalized BTEH:  python step_3_run_initial_matching.py --bteh --help\n"
            "  Legacy giraffe:   python step_3_run_initial_matching.py --legacy-giraffe --help",
            file=sys.stderr,
        )
        sys.exit(1)
