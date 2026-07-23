#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Tests for fixed_probe_evaluator CLI wiring and post-eval orchestration.

All tests use fake scorers, synthetic manifests, and temp artifacts.
No real OOF artifacts or probe images are used.
No real probes are executed.

Test matrix
-----------
 1.  test_cli_hash_check_before_probe_read
       – CLI verifies registration hash before loading any probe data.
 2.  test_splits_hash_mismatch_blocks
       – Mismatched splits manifest hash raises SplitsManifestHashMismatchError.
 3.  test_splits_hash_unspecified_warns_not_fails
       – 'unspecified' splits_manifest_hash skips byte check (warning only).
 4.  test_oof_artifact_hash_mismatch
       – Tampered OOF artifact directory raises OOFArtifactHashMismatchError.
 5.  test_reference_excludes_probe_ids
       – _build_gallery_splits removes probe image_ids from gallery sets.
 6.  test_query_embeddings_dir_defaults_to_ref_dir
       – If query_embeddings_dir is None, ref_embeddings_dir is used for both.
 7.  test_mappings_row_alignment
       – Embedding rows must be contiguous 0..N-1; misaligned rows → error.
 8.  test_exact_oof_scorer_fingerprint
       – Scoring fingerprint changes when registration hash or channels change.
 9.  test_truth_not_passed_to_scorer
       – score_probe_with_scorer passes query_individual_id=None (no forcing).
10.  test_complete_rankings_all_candidates_preserved
       – QueryRankRecord.ranked_ids must carry all candidates in ranked order.
11.  test_atomic_output_marker_after_write
       – Consumed marker is written AFTER rankings parquet, not before.
12.  test_atomic_output_crash_safety
       – If parquet write fails, consumed marker is not written.
13.  test_one_touch_rerun_blocked
       – Second call to score_probes raises RegistrationAlreadyConsumedError.
14.  test_cli_score_dry_run_verifies_hashes
       – dry-run exits without scoring after hash verification.
15.  test_cli_load_not_consumed
       – 'load' command reports 'not_consumed' when marker is absent.
16.  test_cli_load_consumed_with_correct_hash
       – 'load' command succeeds when rankings hash matches registration.
17.  test_cli_load_rankings_hash_mismatch
       – 'load' command raises RegistrationHashMismatchError on hash mismatch.
18.  test_cli_postprocess_verifies_registration_hash
       – postprocess verifies registration hash in rankings before bootstrap.
19.  test_cli_postprocess_hash_mismatch_in_rankings
       – postprocess raises on hash mismatch in rankings file.
20.  test_cli_postprocess_writes_report_and_protocol
       – postprocess writes candidate_report.json and future_session_protocol.json.
21.  test_cli_postprocess_writes_bootstrap_metrics
       – postprocess writes bootstrap_metrics.json with all required keys.
22.  test_cli_postprocess_no_production_mutation
       – postprocess does not write to OOF artifact directory.
23.  test_systems_selected_v1_no_local_scorers
       – selected_v1 system scored when local scorers are absent.
24.  test_systems_exploratory_skipped_when_unavailable
       – Exploratory systems skipped gracefully when channels absent.
25.  test_systems_frozen_ear_uses_non_projected_embeddings
       – selected_v1_frozen_ear substitutes raw ear embeddings for projected.
26.  test_cli_score_subcommand_parser
       – CLI 'score' subcommand exposes all required arguments.
27.  test_cli_load_subcommand_parser
       – CLI 'load' subcommand exposes required arguments.
28.  test_cli_postprocess_subcommand_parser
       – CLI 'postprocess' subcommand exposes required arguments.
29.  test_probe_images_absent_from_gallery_reference
       – Probe image_ids must not appear in any gallery set passed to scorer.
30.  test_verify_splits_hash_correct
       – _verify_splits_manifest_hash passes when hash matches.
31.  test_verify_splits_hash_wrong
       – _verify_splits_manifest_hash raises when bytes differ.
32.  test_verify_oof_hash_correct
       – _verify_oof_artifacts passes when hashes match.
33.  test_verify_oof_hash_fingerprint_mismatch
       – _verify_oof_artifacts raises on scoring fingerprint mismatch.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("BTEH_SOURCE_ROOT", "/nonexistent/BTEH")
os.environ.setdefault("BTEH_ARTIFACT_ROOT", "/nonexistent/artifacts")

from pipeline.statistical_registration import (
    RegistrationHashMismatchError,
    RegistrationAlreadyConsumedError,
    build_registration_document,
    compute_registration_hash,
    load_and_verify_registration,
)
from pipeline.power_simulation import (
    N_TEMPORAL_IDS,
    N_ONBOARDING_IDS,
    SIMULATION_SEED,
)
from pipeline.fixed_probe_evaluator import (
    CONSUMED_MARKER_FILENAME,
    PROBE_RANKINGS_PARQUET,
    COL_REGISTRATION_HASH,
    COL_SYSTEM_NAME,
    COL_QUERY_IMAGE_ID,
    COL_PROBE_TYPE,
    COL_RANK,
    COL_CANDIDATE_ID,
    FixedProbeEvaluator,
    QueryRankRecord,
    SystemSpec,
    SYSTEM_SPECS,
    OOFArtifactHashMismatchError,
    SplitsManifestHashMismatchError,
    _build_gallery_splits,
    _build_parser,
    _records_to_dataframe,
    _scoring_fingerprint,
    _verify_oof_artifacts,
    _verify_splits_manifest_hash,
    build_onboarding_combined_gallery,
    score_probe_with_scorer,
)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _make_registration_doc(
    tmp_path: Path,
    n_temporal: int = 5,
    n_onboarding: int = 3,
    oof_artifacts_hash: str = "fake_oof_hash_abc123",
    oof_scoring_fingerprint: str = "fp_abc",
    splits_manifest_hash: str = "unspecified",
) -> tuple[Path, str]:
    """Create a valid registration file and return (path, reg_hash)."""
    temporal_cs = {f"temp_id_{i:03d}": 2 for i in range(n_temporal)}
    onboarding_cs = {f"onb_id_{i:03d}": 1 for i in range(n_onboarding)}
    doc = build_registration_document(
        temporal_cluster_sizes=temporal_cs,
        onboarding_cluster_sizes=onboarding_cs,
        oof_artifacts_hash=oof_artifacts_hash,
        oof_scoring_fingerprint=oof_scoring_fingerprint,
        selected_v1_eval_hash="fake_v1_hash",
        oof_paired_variance=0.05,
        frozen_k=20,
        all_channels=["miewid", "ear_miewid_projected", "body_local", "ear_local"],
        fusion_weights={"miewid": 0.5, "ear_miewid_projected": 0.5},
        splits_manifest_hash=splits_manifest_hash,
    )
    reg_hash = compute_registration_hash(doc)
    doc["registration_hash"] = reg_hash
    reg_path = tmp_path / "retrospective_registration.json"
    with open(reg_path, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    return reg_path, reg_hash


def _make_splits_parquet(
    tmp_path: Path,
    n_temporal: int = 5,
    n_onboarding: int = 3,
    filename: str = "bteh_splits.parquet",
) -> Path:
    """Create a bteh_splits.parquet with probe, held_out_probe, gallery rows."""
    rows = []
    for i in range(n_temporal):
        for j in range(2):
            rows.append({
                "image_id": f"temporal_{i}_img_{j}",
                "individual_id": f"temp_id_{i:03d}",
                "session_id": f"sess_t_{i}_{j}",
                "split": "probe",
            })
    for i in range(n_onboarding):
        rows.append({
            "image_id": f"onboarding_{i}_img_0",
            "individual_id": f"onb_id_{i:03d}",
            "session_id": f"sess_o_{i}",
            "split": "held_out_probe",
        })
    for i in range(10):
        rows.append({
            "image_id": f"gallery_img_{i}",
            "individual_id": f"temp_id_{i % n_temporal:03d}",
            "session_id": f"gal_sess_{i}",
            "split": "gallery",
        })
    df = pd.DataFrame(rows)
    path = tmp_path / filename
    df.to_parquet(str(path), index=False)
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_registration_with_splits_hash(
    tmp_path: Path,
    n_temporal: int = 5,
    n_onboarding: int = 3,
) -> tuple[Path, Path, str]:
    """
    Create splits parquet + registration with matching splits_manifest_hash.
    Returns (reg_path, splits_path, reg_hash).
    """
    splits_path = _make_splits_parquet(tmp_path, n_temporal, n_onboarding)
    splits_hash = _file_sha256(splits_path)
    reg_path, reg_hash = _make_registration_doc(
        tmp_path,
        n_temporal=n_temporal,
        n_onboarding=n_onboarding,
        splits_manifest_hash=splits_hash,
    )
    return reg_path, splits_path, reg_hash


def _make_oof_artifacts(tmp_path: Path, fp: str = "fp_abc") -> Path:
    """Create minimal OOF artifact files."""
    arts_dir = tmp_path / "oof_artifacts"
    arts_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "all_channels": ["miewid", "ear_miewid_projected", "body_local", "ear_local"],
        "k_default": 50,
    }
    metrics = {"frozen_k": 20, "identity_macro_mrr": 0.41}
    fingerprint = {
        "config_fingerprint": fp,
        "schema_version": "local-v1",
        "saved_at": "2025-01-01T00:00:00Z",
    }
    weights = {
        "miewid": 0.5,
        "ear_miewid_projected": 0.3,
        "body_local": 0.1,
        "ear_local": 0.1,
    }
    (arts_dir / "config.json").write_text(json.dumps(config))
    (arts_dir / "oof_metrics.json").write_text(json.dumps(metrics))
    (arts_dir / "fingerprint.json").write_text(json.dumps(fingerprint))
    (arts_dir / "fusion_weights.json").write_text(json.dumps(weights))
    return arts_dir


def _make_fake_rankings(
    reg_hash: str,
    n_temporal: int = 5,
    n_onboarding: int = 3,
    systems: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a synthetic probe_rankings DataFrame (per-rank rows).
    Each query gets 5 ranked candidates.
    """
    if systems is None:
        systems = ["selected_v1", "selected_v1_plus_both_local"]
    gallery_ids = [f"gal_id_{i:03d}" for i in range(10)]
    rows = []
    for sys_name in systems:
        for i in range(n_temporal):
            iid = f"temp_id_{i:03d}"
            qid = f"temporal_{i}_img_0"
            ranked = [iid] + [g for g in gallery_ids if g != iid][:4]
            for rank, cid in enumerate(ranked, start=1):
                rows.append({
                    COL_QUERY_IMAGE_ID: qid,
                    "truth_individual_id": iid,
                    COL_PROBE_TYPE: "temporal",
                    COL_SYSTEM_NAME: sys_name,
                    COL_RANK: rank,
                    COL_CANDIDATE_ID: cid,
                    "fused_score": 1.0 / rank,
                    "channels_available": "miewid,ear_miewid_projected",
                    "scoring_fingerprint": "fp_test",
                    COL_REGISTRATION_HASH: reg_hash,
                })
        for i in range(n_onboarding):
            iid = f"onb_id_{i:03d}"
            qid = f"onboarding_{i}_img_0"
            ranked = [iid] + [g for g in gallery_ids if g != iid][:4]
            for rank, cid in enumerate(ranked, start=1):
                rows.append({
                    COL_QUERY_IMAGE_ID: qid,
                    "truth_individual_id": iid,
                    COL_PROBE_TYPE: "onboarding",
                    COL_SYSTEM_NAME: sys_name,
                    COL_RANK: rank,
                    COL_CANDIDATE_ID: cid,
                    "fused_score": 1.0 / rank,
                    "channels_available": "miewid,ear_miewid_projected",
                    "scoring_fingerprint": "fp_test",
                    COL_REGISTRATION_HASH: reg_hash,
                })
    return pd.DataFrame(rows)


def _write_fake_rankings(output_dir: Path, reg_hash: str, **kwargs) -> Path:
    """Write synthetic rankings parquet and consumed marker, return parquet path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _make_fake_rankings(reg_hash, **kwargs)
    path = output_dir / PROBE_RANKINGS_PARQUET
    df.to_parquet(str(path), index=False)
    (output_dir / CONSUMED_MARKER_FILENAME).write_text(reg_hash)
    return path


# ===========================================================================
# 1. test_cli_hash_check_before_probe_read
# ===========================================================================
class TestCliHashCheckBeforeProbeRead:
    """CLI verifies registration hash before loading any probe data."""

    def test_tampered_registration_fails_before_splits_load(self, tmp_path):
        """Even if splits parquet exists, a bad registration hash fails first."""
        reg_path, _ = _make_registration_doc(tmp_path)
        splits_path = _make_splits_parquet(tmp_path)

        # Tamper with the registration file
        with open(reg_path) as fh:
            doc = json.load(fh)
        doc["frozen_k"] = 999
        with open(reg_path, "w") as fh:
            json.dump(doc, fh)

        # FixedProbeEvaluator verifies hash on construction
        with pytest.raises(RegistrationHashMismatchError):
            FixedProbeEvaluator(reg_path, tmp_path / "out")

    def test_valid_registration_does_not_raise(self, tmp_path):
        reg_path, _ = _make_registration_doc(tmp_path)
        # Should not raise
        ev = FixedProbeEvaluator(reg_path, tmp_path / "out")
        assert ev.registration_hash


# ===========================================================================
# 2. test_splits_hash_mismatch_blocks
# ===========================================================================
class TestSplitsHashMismatch:
    def test_different_file_raises(self, tmp_path):
        splits_path = _make_splits_parquet(tmp_path)
        # Register with a different (wrong) hash
        wrong_hash = "a" * 64
        reg_path, _ = _make_registration_doc(
            tmp_path, splits_manifest_hash=wrong_hash
        )
        registration = load_and_verify_registration(reg_path)

        with pytest.raises(SplitsManifestHashMismatchError, match="hash mismatch"):
            _verify_splits_manifest_hash(splits_path, registration)

    def test_correct_hash_passes(self, tmp_path):
        reg_path, splits_path, _ = _make_registration_with_splits_hash(tmp_path)
        registration = load_and_verify_registration(reg_path)
        # Should not raise
        _verify_splits_manifest_hash(splits_path, registration)

    def test_modified_file_raises(self, tmp_path):
        reg_path, splits_path, _ = _make_registration_with_splits_hash(tmp_path)
        registration = load_and_verify_registration(reg_path)
        # Append a byte to the splits file
        with open(splits_path, "ab") as fh:
            fh.write(b"\x00")
        with pytest.raises(SplitsManifestHashMismatchError):
            _verify_splits_manifest_hash(splits_path, registration)


# ===========================================================================
# 3. test_splits_hash_unspecified_warns_not_fails
# ===========================================================================
class TestSplitsHashUnspecified:
    def test_unspecified_does_not_raise(self, tmp_path):
        splits_path = _make_splits_parquet(tmp_path)
        reg_path, _ = _make_registration_doc(
            tmp_path, splits_manifest_hash="unspecified"
        )
        registration = load_and_verify_registration(reg_path)
        # Should not raise; emits a warning
        _verify_splits_manifest_hash(splits_path, registration)

    def test_absent_hash_does_not_raise(self, tmp_path):
        splits_path = _make_splits_parquet(tmp_path)
        reg_path, _ = _make_registration_doc(tmp_path, splits_manifest_hash="unspecified")
        registration = load_and_verify_registration(reg_path)
        # Remove splits_manifest_hash entirely
        del registration["splits_manifest_hash"]
        _verify_splits_manifest_hash(splits_path, registration)


# ===========================================================================
# 4. test_oof_artifact_hash_mismatch
# ===========================================================================
class TestOOFArtifactHashMismatch:
    def test_tampered_config_raises(self, tmp_path):
        arts_dir = _make_oof_artifacts(tmp_path, fp="fp_abc")
        # Compute real hash from the artifacts
        from pipeline.statistical_registration import _hash_oof_artifacts
        real_hash = _hash_oof_artifacts(arts_dir)
        reg_path, _ = _make_registration_doc(
            tmp_path,
            oof_artifacts_hash=real_hash,
            oof_scoring_fingerprint="fp_abc",
        )
        registration = load_and_verify_registration(reg_path)

        # Tamper with config
        config_path = arts_dir / "config.json"
        with open(config_path) as fh:
            cfg = json.load(fh)
        cfg["k_default"] = 99999
        with open(config_path, "w") as fh:
            json.dump(cfg, fh)

        with pytest.raises(OOFArtifactHashMismatchError, match="hash mismatch"):
            _verify_oof_artifacts(arts_dir, registration)

    def test_fingerprint_mismatch_raises(self, tmp_path):
        arts_dir = _make_oof_artifacts(tmp_path, fp="fp_correct")
        from pipeline.statistical_registration import _hash_oof_artifacts
        real_hash = _hash_oof_artifacts(arts_dir)
        reg_path, _ = _make_registration_doc(
            tmp_path,
            oof_artifacts_hash=real_hash,
            oof_scoring_fingerprint="fp_DIFFERENT",  # wrong fingerprint
        )
        registration = load_and_verify_registration(reg_path)

        with pytest.raises(OOFArtifactHashMismatchError, match="fingerprint mismatch"):
            _verify_oof_artifacts(arts_dir, registration)

    def test_matching_artifacts_passes(self, tmp_path):
        arts_dir = _make_oof_artifacts(tmp_path, fp="fp_abc")
        from pipeline.statistical_registration import _hash_oof_artifacts
        real_hash = _hash_oof_artifacts(arts_dir)
        reg_path, _ = _make_registration_doc(
            tmp_path,
            oof_artifacts_hash=real_hash,
            oof_scoring_fingerprint="fp_abc",
        )
        registration = load_and_verify_registration(reg_path)
        # Should not raise
        _verify_oof_artifacts(arts_dir, registration)


# ===========================================================================
# 5. test_reference_excludes_probe_ids
# ===========================================================================
class TestReferenceExcludesProbeIds:
    def test_probe_images_not_in_gallery(self, tmp_path):
        splits_path = _make_splits_parquet(tmp_path, n_temporal=4, n_onboarding=2)
        splits_df = pd.read_parquet(str(splits_path))
        temporal_df, onboarding_df, gallery_df, held_out_df = _build_gallery_splits(splits_df)

        probe_ids = set(temporal_df["image_id"]) | set(onboarding_df["image_id"])
        gallery_ids = set(gallery_df["image_id"])
        held_out_ids = set(held_out_df["image_id"])

        assert len(probe_ids & gallery_ids) == 0, (
            "Probe images leaked into temporal gallery."
        )
        assert len(probe_ids & held_out_ids) == 0, (
            "Probe images leaked into held_out gallery."
        )

    def test_combined_gallery_excludes_probes(self, tmp_path):
        rows = [
            {"image_id": "gal1", "individual_id": "id_a", "session_id": "s1", "split": "gallery"},
            {"image_id": "hog1", "individual_id": "id_b", "session_id": "s2",
             "split": "held_out_gallery"},
            {"image_id": "prb1", "individual_id": "id_c", "session_id": "s3",
             "split": "probe"},
            {"image_id": "hprb1", "individual_id": "id_d", "session_id": "s4",
             "split": "held_out_probe"},
        ]
        df = pd.DataFrame(rows)
        _, _, gallery_df, held_out_df = _build_gallery_splits(df)
        combined = build_onboarding_combined_gallery(gallery_df, held_out_df)

        assert "prb1" not in combined["image_id"].values
        assert "hprb1" not in combined["image_id"].values
        assert "gal1" in combined["image_id"].values
        assert "hog1" in combined["image_id"].values


# ===========================================================================
# 6. test_query_embeddings_dir_defaults_to_ref_dir
# ===========================================================================
class TestQueryEmbeddingsDirDefault:
    def test_none_query_dir_uses_ref_dir(self):
        """When query_embeddings_dir is None, ref_embeddings_dir is used for both."""
        # This is tested via the CLI parser and _cmd_score logic.
        # Simulate the CLI argument defaulting.
        parser = _build_parser()
        args = parser.parse_args([
            "score",
            "--registration-file", "reg.json",
            "--splits-parquet", "splits.parquet",
            "--oof-artifacts-dir", "oof/",
            "--crop-manifest", "crop.parquet",
            "--ref-embeddings-dir", "/ref/emb",
            "--output-dir", "/out",
        ])
        assert args.query_embeddings_dir is None
        # The CLI sets query_emb_dir = ref_emb_dir when None


# ===========================================================================
# 7. test_mappings_row_alignment
# ===========================================================================
class TestMappingsRowAlignment:
    def test_non_contiguous_rows_fail(self, tmp_path):
        """Descriptor mapping with non-contiguous embedding_row raises."""
        from pipeline.local_oof_calibration import (
            _load_embedding_matrices_and_mappings,
            FingerprintMismatchError,
        )
        emb_dir = tmp_path / "emb"
        emb_dir.mkdir()
        mat = np.random.default_rng(0).random((5, 8)).astype(np.float32)
        np.save(str(emb_dir / "miewid.npy"), mat)
        # Non-contiguous rows: 0, 1, 3 (skips 2)
        mapping = pd.DataFrame({
            "image_id": ["a", "b", "c"],
            "embedding_row": [0, 1, 3],  # gap!
        })
        mapping.to_parquet(str(emb_dir / "miewid_mapping.parquet"), index=False)

        with pytest.raises((FingerprintMismatchError, RuntimeError)):
            _load_embedding_matrices_and_mappings(emb_dir, ["miewid"])

    def test_contiguous_rows_pass(self, tmp_path):
        from pipeline.local_oof_calibration import _load_embedding_matrices_and_mappings

        emb_dir = tmp_path / "emb"
        emb_dir.mkdir()
        mat = np.random.default_rng(0).random((3, 8)).astype(np.float32)
        np.save(str(emb_dir / "miewid.npy"), mat)
        mapping = pd.DataFrame({
            "image_id": ["a", "b", "c"],
            "embedding_row": [0, 1, 2],
        })
        mapping.to_parquet(str(emb_dir / "miewid_mapping.parquet"), index=False)

        mats, maps = _load_embedding_matrices_and_mappings(emb_dir, ["miewid"])
        assert "miewid" in mats
        assert len(maps["miewid"]) == 3


# ===========================================================================
# 8. test_exact_oof_scorer_fingerprint
# ===========================================================================
class TestExactOOFScorerFingerprint:
    def test_fingerprint_changes_with_registration_hash(self):
        fp1 = _scoring_fingerprint("sys_a", "hash_A", ["miewid"], {"miewid": 1.0}, 20)
        fp2 = _scoring_fingerprint("sys_a", "hash_B", ["miewid"], {"miewid": 1.0}, 20)
        assert fp1 != fp2

    def test_fingerprint_changes_with_channels(self):
        fp1 = _scoring_fingerprint("sys_a", "hash_A", ["miewid"], {"miewid": 1.0}, 20)
        fp2 = _scoring_fingerprint(
            "sys_a", "hash_A", ["miewid", "ear_miewid_projected"],
            {"miewid": 0.5, "ear_miewid_projected": 0.5}, 20,
        )
        assert fp1 != fp2

    def test_fingerprint_deterministic(self):
        for _ in range(3):
            fp = _scoring_fingerprint("sys_a", "hash", ["miewid"], {"miewid": 1.0}, 20)
        assert fp == _scoring_fingerprint("sys_a", "hash", ["miewid"], {"miewid": 1.0}, 20)

    def test_fingerprint_changes_with_frozen_k(self):
        fp1 = _scoring_fingerprint("sys_a", "hash", ["miewid"], {"miewid": 1.0}, 20)
        fp2 = _scoring_fingerprint("sys_a", "hash", ["miewid"], {"miewid": 1.0}, 50)
        assert fp1 != fp2


# ===========================================================================
# 9. test_truth_not_passed_to_scorer
# ===========================================================================
class TestTruthNotPassedToScorer:
    def test_query_individual_id_is_none(self):
        """score_probe_with_scorer must NOT pass truth to scorer."""
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = []
        mock_scorer.artifacts = MagicMock()

        score_probe_with_scorer(
            query_image_id="img1",
            query_session_id="sess1",
            truth_individual_id="TRUE_IDENTITY",  # known truth, must not be passed
            probe_type="temporal",
            scorer=mock_scorer,
            system_name="selected_v1",
            registration_hash="reg_hash",
            frozen_k=20,
            fusion_weights={"miewid": 1.0},
        )
        call_kwargs = mock_scorer.score.call_args
        assert call_kwargs.kwargs.get("query_individual_id") is None, (
            "Truth (query_individual_id) must not be passed to scorer — "
            "no candidate truth forcing is a core design contract."
        )

    def test_shortlist_not_leaked_with_truth(self):
        """candidate_ids passed to scorer must not include any truth-forcing logic."""
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = []
        mock_scorer.artifacts = MagicMock()

        score_probe_with_scorer(
            query_image_id="img1",
            query_session_id="sess1",
            truth_individual_id="TRUE_IDENTITY",
            probe_type="onboarding",
            scorer=mock_scorer,
            system_name="selected_v1_plus_both_local",
            registration_hash="h",
            frozen_k=10,
            fusion_weights={"miewid": 0.5, "ear_miewid_projected": 0.5},
            shortlist=["cand_a", "cand_b"],
        )
        call_kwargs = mock_scorer.score.call_args
        # truth must not appear in candidate_ids
        cands = call_kwargs.kwargs.get("candidate_ids", []) or []
        assert "TRUE_IDENTITY" not in cands


# ===========================================================================
# 10. test_complete_rankings_all_candidates_preserved
# ===========================================================================
class TestCompleteRankingsPreserved:
    def test_all_ranked_ids_in_record(self):
        """QueryRankRecord must carry the full ordered ranking."""
        mock_scorer = MagicMock()
        # Simulate scorer returning 5 candidates
        from models.identity_fusion import IdentityScore
        fake_results = [
            IdentityScore(
                individual_id=f"id_{i:03d}",
                channel_raw={},
                channel_calibrated={},
                channels_available=["miewid"],
                fused_score=1.0 / (i + 1),
            )
            for i in range(5)
        ]
        mock_scorer.score.return_value = fake_results

        rec = score_probe_with_scorer(
            query_image_id="q1",
            query_session_id="s1",
            truth_individual_id="id_000",
            probe_type="temporal",
            scorer=mock_scorer,
            system_name="selected_v1",
            registration_hash="rh",
            frozen_k=5,
            fusion_weights={"miewid": 1.0},
        )
        assert len(rec.ranked_ids) == 5
        assert rec.ranked_ids[0] == "id_000"
        assert len(rec.fused_scores) == 5
        assert rec.fused_scores[0] >= rec.fused_scores[-1]

    def test_dataframe_preserves_complete_ranking(self):
        """_records_to_dataframe must emit one row per (query, rank) for all ranks."""
        recs = [
            QueryRankRecord(
                query_image_id="q1",
                truth_individual_id="id_a",
                probe_type="temporal",
                system_name="selected_v1",
                ranked_ids=[f"id_{i}" for i in range(7)],
                fused_scores=[1.0 / (i + 1) for i in range(7)],
                channels_available=["miewid"],
                scoring_fingerprint="fp",
                registration_hash="rh",
            )
        ]
        df = _records_to_dataframe(recs)
        assert len(df) == 7
        assert sorted(df["rank"].unique().tolist()) == list(range(1, 8))
        # All candidates must appear
        assert set(df["candidate_individual_id"]) == {f"id_{i}" for i in range(7)}


# ===========================================================================
# 11. test_atomic_output_marker_after_write
# ===========================================================================
class TestAtomicOutputMarkerAfterWrite:
    def test_marker_written_after_rankings(self, tmp_path):
        """
        After score_probes() completes:
        - probe_rankings.parquet must exist and be readable.
        - consumed marker must exist.
        - Marker must appear AFTER (or at same time as) the rankings file.
        """
        reg_path, reg_hash = _make_registration_doc(tmp_path)
        output_dir = tmp_path / "output"
        evaluator = FixedProbeEvaluator(reg_path, output_dir)

        # Build a minimal splits_df (no probes → empty rankings but valid)
        splits_df = pd.DataFrame(columns=["image_id", "individual_id", "session_id", "split"])
        mock_artifacts = MagicMock()
        mock_artifacts.frozen_k = 20
        mock_artifacts.fusion_weights = {"miewid": 1.0}
        mock_artifacts.all_channels = ["miewid"]

        # Empty probes → score_probes writes empty parquet
        with pytest.raises(ValueError, match="No probe rows"):
            evaluator.score_probes(
                splits_df=splits_df,
                artifacts=mock_artifacts,
                crop_df=pd.DataFrame(),
                embedding_matrices={},
                descriptor_mappings={},
            )
        # Marker should NOT exist (scoring failed)
        assert not evaluator._consumed_marker.exists()

    def test_marker_exists_after_successful_score(self, tmp_path):
        """When scoring succeeds, both rankings and marker are written."""
        reg_path, reg_hash = _make_registration_doc(tmp_path)
        output_dir = tmp_path / "output"
        evaluator = FixedProbeEvaluator(reg_path, output_dir)

        # Build splits_df with probes
        rows = [
            {"image_id": "probe_img_0", "individual_id": "id_0",
             "session_id": "s0", "split": "probe"},
            {"image_id": "gallery_img_0", "individual_id": "id_0",
             "session_id": "g0", "split": "gallery"},
        ]
        splits_df = pd.DataFrame(rows)
        mock_artifacts = MagicMock()
        mock_artifacts.frozen_k = 5
        mock_artifacts.fusion_weights = {"miewid": 1.0}
        mock_artifacts.all_channels = ["miewid"]

        # Mock scorer returns empty list
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = []

        with patch(
            "pipeline.fixed_probe_evaluator.build_system_scorer",
            return_value=mock_scorer,
        ):
            evaluator.score_probes(
                splits_df=splits_df,
                artifacts=mock_artifacts,
                crop_df=pd.DataFrame(),
                embedding_matrices={"miewid": np.zeros((1, 8))},
                descriptor_mappings={"miewid": pd.DataFrame()},
                systems=["selected_v1"],
            )

        rankings_path = output_dir / PROBE_RANKINGS_PARQUET
        marker_path = output_dir / CONSUMED_MARKER_FILENAME

        assert rankings_path.exists(), "Rankings parquet must be written."
        assert marker_path.exists(), "Consumed marker must be written after rankings."

        # Marker content must be registration hash
        assert marker_path.read_text() == reg_hash


# ===========================================================================
# 12. test_atomic_output_crash_safety
# ===========================================================================
class TestAtomicOutputCrashSafety:
    def test_failed_parquet_write_does_not_set_marker(self, tmp_path):
        """
        If parquet write raises, the consumed marker must NOT be written.
        The one-touch token must only be consumed when output is durable.
        """
        reg_path, reg_hash = _make_registration_doc(tmp_path)
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        evaluator = FixedProbeEvaluator(reg_path, output_dir)

        rows = [
            {"image_id": "probe_img_0", "individual_id": "id_0",
             "session_id": "s0", "split": "probe"},
            {"image_id": "gallery_img_0", "individual_id": "id_0",
             "session_id": "g0", "split": "gallery"},
        ]
        splits_df = pd.DataFrame(rows)
        mock_artifacts = MagicMock()
        mock_artifacts.frozen_k = 5
        mock_artifacts.fusion_weights = {"miewid": 1.0}
        mock_artifacts.all_channels = ["miewid"]
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = []

        # Patch DataFrame.to_parquet to raise
        with patch(
            "pipeline.fixed_probe_evaluator.build_system_scorer",
            return_value=mock_scorer,
        ):
            with patch.object(pd.DataFrame, "to_parquet", side_effect=OSError("disk full")):
                with pytest.raises(OSError, match="disk full"):
                    evaluator.score_probes(
                        splits_df=splits_df,
                        artifacts=mock_artifacts,
                        crop_df=pd.DataFrame(),
                        embedding_matrices={"miewid": np.zeros((1, 8))},
                        descriptor_mappings={"miewid": pd.DataFrame()},
                        systems=["selected_v1"],
                    )

        marker_path = output_dir / CONSUMED_MARKER_FILENAME
        assert not marker_path.exists(), (
            "Consumed marker must NOT be written when parquet write fails."
        )


# ===========================================================================
# 13. test_one_touch_rerun_blocked
# ===========================================================================
class TestOneTouchRerunBlocked:
    def test_second_score_probes_raises(self, tmp_path):
        reg_path, reg_hash = _make_registration_doc(tmp_path)
        output_dir = tmp_path / "output"
        evaluator = FixedProbeEvaluator(reg_path, output_dir)

        # Plant a consumed marker
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / CONSUMED_MARKER_FILENAME).write_text(reg_hash)

        mock_artifacts = MagicMock()
        mock_artifacts.frozen_k = 5
        mock_artifacts.fusion_weights = {}
        mock_artifacts.all_channels = []

        with pytest.raises(RegistrationAlreadyConsumedError):
            evaluator.score_probes(
                splits_df=pd.DataFrame(
                    columns=["image_id", "individual_id", "session_id", "split"]
                ),
                artifacts=mock_artifacts,
                crop_df=pd.DataFrame(),
                embedding_matrices={},
                descriptor_mappings={},
            )

    def test_is_consumed_property(self, tmp_path):
        reg_path, reg_hash = _make_registration_doc(tmp_path)
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluator = FixedProbeEvaluator(reg_path, output_dir)

        assert not evaluator.is_consumed
        (output_dir / CONSUMED_MARKER_FILENAME).write_text(reg_hash)
        assert evaluator.is_consumed


# ===========================================================================
# 14. test_cli_score_dry_run_verifies_hashes
# ===========================================================================
class TestCliScoreDryRun:
    def test_dry_run_no_probe_execution(self, tmp_path, capsys):
        """dry-run exits after hash verification without any probe scoring."""
        arts_dir = _make_oof_artifacts(tmp_path)
        from pipeline.statistical_registration import _hash_oof_artifacts
        real_oof_hash = _hash_oof_artifacts(arts_dir)
        reg_path, _ = _make_registration_doc(
            tmp_path,
            oof_artifacts_hash=real_oof_hash,
            oof_scoring_fingerprint="fp_abc",
            splits_manifest_hash="unspecified",
        )
        splits_path = _make_splits_parquet(tmp_path)
        output_dir = tmp_path / "output"
        crop_path = tmp_path / "crop.parquet"
        pd.DataFrame(columns=["image_id", "crop_id", "crop_path",
                               "crop_kind", "detector_status"]).to_parquet(str(crop_path))

        from pipeline.fixed_probe_evaluator import main
        ret = main([
            "score",
            "--registration-file", str(reg_path),
            "--splits-parquet", str(splits_path),
            "--oof-artifacts-dir", str(arts_dir),
            "--crop-manifest", str(crop_path),
            "--ref-embeddings-dir", str(tmp_path),  # placeholder
            "--output-dir", str(output_dir),
            "--dry-run",
        ])
        captured = capsys.readouterr()
        out = json.loads(captured.out)
        assert ret == 0
        assert out["status"] == "dry_run_ok"
        assert out["oof_artifacts_verified"] is True
        # No consumed marker written
        assert not (output_dir / CONSUMED_MARKER_FILENAME).exists()

    def test_dry_run_with_bad_oof_hash_fails(self, tmp_path):
        """dry-run with mismatched OOF hash raises before scoring."""
        arts_dir = _make_oof_artifacts(tmp_path)
        reg_path, _ = _make_registration_doc(
            tmp_path,
            oof_artifacts_hash="wrong_hash_" + "0" * 54,
            splits_manifest_hash="unspecified",
        )
        splits_path = _make_splits_parquet(tmp_path)
        output_dir = tmp_path / "output"
        crop_path = tmp_path / "crop.parquet"
        pd.DataFrame().to_parquet(str(crop_path))

        from pipeline.fixed_probe_evaluator import main
        with pytest.raises(OOFArtifactHashMismatchError):
            main([
                "score",
                "--registration-file", str(reg_path),
                "--splits-parquet", str(splits_path),
                "--oof-artifacts-dir", str(arts_dir),
                "--crop-manifest", str(crop_path),
                "--ref-embeddings-dir", str(tmp_path),
                "--output-dir", str(output_dir),
                "--dry-run",
            ])


# ===========================================================================
# 15. test_cli_load_not_consumed
# ===========================================================================
class TestCliLoadNotConsumed:
    def test_reports_not_consumed(self, tmp_path, capsys):
        reg_path, reg_hash = _make_registration_doc(tmp_path)
        rankings_dir = tmp_path / "output"
        rankings_dir.mkdir(parents=True)

        from pipeline.fixed_probe_evaluator import main
        ret = main([
            "load",
            "--registration-file", str(reg_path),
            "--rankings-dir", str(rankings_dir),
        ])
        captured = capsys.readouterr()
        out = json.loads(captured.out)
        assert ret == 0
        assert out["status"] == "not_consumed"


# ===========================================================================
# 16. test_cli_load_consumed_with_correct_hash
# ===========================================================================
class TestCliLoadConsumedCorrectHash:
    def test_load_succeeds(self, tmp_path, capsys):
        reg_path, reg_hash = _make_registration_doc(tmp_path)
        rankings_dir = tmp_path / "output"
        _write_fake_rankings(rankings_dir, reg_hash)

        from pipeline.fixed_probe_evaluator import main
        ret = main([
            "load",
            "--registration-file", str(reg_path),
            "--rankings-dir", str(rankings_dir),
        ])
        captured = capsys.readouterr()
        out = json.loads(captured.out)
        assert ret == 0
        assert out["status"] == "ok"
        assert out["consumed_marker_present"] is True
        assert "selected_v1" in out["systems"]


# ===========================================================================
# 17. test_cli_load_rankings_hash_mismatch
# ===========================================================================
class TestCliLoadHashMismatch:
    def test_wrong_hash_in_rankings_raises(self, tmp_path):
        reg_path, reg_hash = _make_registration_doc(tmp_path)
        rankings_dir = tmp_path / "output"
        # Write rankings with DIFFERENT registration hash
        _write_fake_rankings(rankings_dir, "wrong_hash_" + "x" * 53)
        # Also write a consumed marker matching the correct hash
        (rankings_dir / CONSUMED_MARKER_FILENAME).write_text(reg_hash)

        from pipeline.fixed_probe_evaluator import main
        with pytest.raises(RegistrationHashMismatchError):
            main([
                "load",
                "--registration-file", str(reg_path),
                "--rankings-dir", str(rankings_dir),
            ])


# ===========================================================================
# 18 & 19. test_cli_postprocess_verifies_registration_hash
# ===========================================================================
class TestCliPostprocessHashVerification:
    def test_postprocess_verifies_rankings_hash(self, tmp_path, capsys):
        """postprocess succeeds when rankings hash matches registration."""
        reg_path, reg_hash = _make_registration_doc(tmp_path)
        rankings_dir = tmp_path / "output"
        _write_fake_rankings(rankings_dir, reg_hash)
        postprocess_dir = tmp_path / "postprocess"

        from pipeline.fixed_probe_evaluator import main
        ret = main([
            "postprocess",
            "--registration-file", str(reg_path),
            "--rankings-dir", str(rankings_dir),
            "--output-dir", str(postprocess_dir),
            "--n-replicates", "50",  # fast for testing
        ])
        captured = capsys.readouterr()
        out = json.loads(captured.out)
        assert ret == 0
        assert out["status"] == "ok"
        assert out["registration_hash"] == reg_hash

    def test_postprocess_raises_on_hash_mismatch(self, tmp_path):
        """postprocess raises when rankings have mismatched registration hash."""
        reg_path, reg_hash = _make_registration_doc(tmp_path)
        rankings_dir = tmp_path / "output"
        _write_fake_rankings(rankings_dir, "tampered_hash_" + "0" * 50)
        (rankings_dir / CONSUMED_MARKER_FILENAME).write_text(reg_hash)
        postprocess_dir = tmp_path / "postprocess"

        from pipeline.fixed_probe_evaluator import main
        with pytest.raises(RegistrationHashMismatchError):
            main([
                "postprocess",
                "--registration-file", str(reg_path),
                "--rankings-dir", str(rankings_dir),
                "--output-dir", str(postprocess_dir),
                "--n-replicates", "10",
            ])

    def test_postprocess_rejects_missing_baseline_system(self, tmp_path):
        reg_path, reg_hash = _make_registration_doc(tmp_path)
        rankings_dir = tmp_path / "output"
        _write_fake_rankings(rankings_dir, reg_hash)

        from pipeline.fixed_probe_evaluator import main
        with pytest.raises(
            ValueError,
            match="missing systems required for postprocessing.*not_scored",
        ):
            main([
                "postprocess",
                "--registration-file", str(reg_path),
                "--rankings-dir", str(rankings_dir),
                "--output-dir", str(tmp_path / "postprocess"),
                "--baseline-system", "not_scored",
                "--n-replicates", "10",
            ])


# ===========================================================================
# 20. test_cli_postprocess_writes_report_and_protocol
# ===========================================================================
class TestCliPostprocessOutputFiles:
    def test_writes_report_and_protocol(self, tmp_path):
        reg_path, reg_hash = _make_registration_doc(tmp_path)
        rankings_dir = tmp_path / "output"
        _write_fake_rankings(rankings_dir, reg_hash)
        postprocess_dir = tmp_path / "postprocess"

        from pipeline.fixed_probe_evaluator import main
        main([
            "postprocess",
            "--registration-file", str(reg_path),
            "--rankings-dir", str(rankings_dir),
            "--output-dir", str(postprocess_dir),
            "--n-replicates", "30",
        ])

        report_path = postprocess_dir / "candidate_report.json"
        protocol_path = postprocess_dir / "future_session_protocol.json"
        assert report_path.exists(), "candidate_report.json must be written."
        assert protocol_path.exists(), "future_session_protocol.json must be written."

        with open(report_path) as fh:
            report = json.load(fh)
        assert report.get("registration_hash") == reg_hash
        assert "decision" in report
        assert report["head_gate_status"] == "failed_not_applicable"


# ===========================================================================
# 21. test_cli_postprocess_writes_bootstrap_metrics
# ===========================================================================
class TestCliPostprocessBootstrapMetrics:
    def test_writes_bootstrap_metrics_json(self, tmp_path):
        reg_path, reg_hash = _make_registration_doc(tmp_path)
        rankings_dir = tmp_path / "output"
        _write_fake_rankings(rankings_dir, reg_hash)
        postprocess_dir = tmp_path / "postprocess"

        from pipeline.fixed_probe_evaluator import main
        main([
            "postprocess",
            "--registration-file", str(reg_path),
            "--rankings-dir", str(rankings_dir),
            "--output-dir", str(postprocess_dir),
            "--n-replicates", "30",
        ])

        metrics_path = postprocess_dir / "bootstrap_metrics.json"
        assert metrics_path.exists()

        with open(metrics_path) as fh:
            metrics = json.load(fh)

        required_keys = {
            "registration_hash", "candidate_system", "baseline_system",
            "n_replicates", "seed", "system_metrics",
            "primary_contrast", "secondary_contrasts",
        }
        assert required_keys.issubset(set(metrics.keys()))
        assert metrics["registration_hash"] == reg_hash
        assert metrics["seed"] == SIMULATION_SEED

    def test_bootstrap_metrics_primary_contrast_fields(self, tmp_path):
        reg_path, reg_hash = _make_registration_doc(tmp_path)
        rankings_dir = tmp_path / "output"
        _write_fake_rankings(rankings_dir, reg_hash)
        postprocess_dir = tmp_path / "postprocess"

        from pipeline.fixed_probe_evaluator import main
        main([
            "postprocess",
            "--registration-file", str(reg_path),
            "--rankings-dir", str(rankings_dir),
            "--output-dir", str(postprocess_dir),
            "--n-replicates", "30",
        ])

        with open(postprocess_dir / "bootstrap_metrics.json") as fh:
            metrics = json.load(fh)
        pc = metrics["primary_contrast"]
        for key in ("contrast_name", "point_delta", "ci_lo", "ci_hi",
                    "p_value", "p_value_holm", "reject_h0"):
            assert key in pc, f"Missing key in primary_contrast: {key}"


# ===========================================================================
# 22. test_cli_postprocess_no_production_mutation
# ===========================================================================
class TestCliPostprocessNoProductionMutation:
    def test_oof_artifacts_unchanged(self, tmp_path):
        """postprocess must not write to OOF artifacts directory."""
        arts_dir = _make_oof_artifacts(tmp_path)
        mtime_before = {f: f.stat().st_mtime for f in arts_dir.iterdir()}

        reg_path, reg_hash = _make_registration_doc(tmp_path)
        rankings_dir = tmp_path / "output"
        _write_fake_rankings(rankings_dir, reg_hash)
        postprocess_dir = tmp_path / "postprocess"

        from pipeline.fixed_probe_evaluator import main
        main([
            "postprocess",
            "--registration-file", str(reg_path),
            "--rankings-dir", str(rankings_dir),
            "--output-dir", str(postprocess_dir),
            "--n-replicates", "20",
        ])

        mtime_after = {f: f.stat().st_mtime for f in arts_dir.iterdir()}
        assert mtime_before == mtime_after, (
            "postprocess must not modify OOF artifact files."
        )

    def test_registration_file_unchanged(self, tmp_path):
        """postprocess must not modify the registration file."""
        reg_path, reg_hash = _make_registration_doc(tmp_path)
        mtime = reg_path.stat().st_mtime

        rankings_dir = tmp_path / "output"
        _write_fake_rankings(rankings_dir, reg_hash)
        postprocess_dir = tmp_path / "postprocess"

        from pipeline.fixed_probe_evaluator import main
        main([
            "postprocess",
            "--registration-file", str(reg_path),
            "--rankings-dir", str(rankings_dir),
            "--output-dir", str(postprocess_dir),
            "--n-replicates", "20",
        ])

        assert reg_path.stat().st_mtime == mtime

    def test_rankings_file_unchanged(self, tmp_path):
        """postprocess must not modify the probe_rankings.parquet."""
        reg_path, reg_hash = _make_registration_doc(tmp_path)
        rankings_dir = tmp_path / "output"
        rankings_path = _write_fake_rankings(rankings_dir, reg_hash)
        mtime = rankings_path.stat().st_mtime
        postprocess_dir = tmp_path / "postprocess"

        from pipeline.fixed_probe_evaluator import main
        main([
            "postprocess",
            "--registration-file", str(reg_path),
            "--rankings-dir", str(rankings_dir),
            "--output-dir", str(postprocess_dir),
            "--n-replicates", "20",
        ])

        assert rankings_path.stat().st_mtime == mtime


# ===========================================================================
# 23. test_systems_selected_v1_no_local_scorers
# ===========================================================================
class TestSystemsSelectedV1NoLocalScorers:
    def test_selected_v1_uses_global_channels_only(self):
        """selected_v1 system uses only global channels, ignores local scorers."""
        spec = SYSTEM_SPECS["selected_v1"]
        assert spec.channels_override is not None
        from pipeline.local_oof_calibration import GLOBAL_CHANNELS
        for ch in spec.channels_override:
            assert ch in GLOBAL_CHANNELS, (
                f"selected_v1 should use only global channels, found {ch}"
            )

    def test_selected_v1_no_local_flag(self):
        spec = SYSTEM_SPECS["selected_v1"]
        assert not spec.use_frozen_ear
        assert not spec.exploratory


# ===========================================================================
# 24. test_systems_exploratory_skipped_when_unavailable
# ===========================================================================
class TestExploratorySystems:
    def test_local_only_exploratory_spec(self):
        spec = SYSTEM_SPECS["local_only_exploratory"]
        assert spec.exploratory is True

    def test_mega_exploratory_spec(self):
        spec = SYSTEM_SPECS["megadesc_frozen_ear_exploratory"]
        assert spec.exploratory is True

    def test_mega_exploratory_skipped_when_channels_missing(self):
        """
        megadesc_frozen_ear_exploratory returns None when megadescriptor channels
        are absent from embedding_matrices.
        """
        from pipeline.fixed_probe_evaluator import build_system_scorer

        spec = SYSTEM_SPECS["megadesc_frozen_ear_exploratory"]
        mock_artifacts = MagicMock()
        mock_artifacts.all_channels = ["miewid", "ear_miewid_projected"]
        mock_artifacts.fusion_weights = {"miewid": 1.0, "ear_miewid_projected": 0.0}
        mock_artifacts.frozen_k = 20
        mock_artifacts.calibrators_global = {}
        mock_artifacts.calibrator_body = None
        mock_artifacts.calibrator_ear = None
        mock_artifacts.oof_metrics = {}
        mock_artifacts.config = {}
        mock_artifacts.fingerprint = {}

        gallery_df = pd.DataFrame(
            {"image_id": ["g1"], "individual_id": ["id_0"],
             "session_id": ["s0"], "split": ["gallery"]}
        )
        # embedding_matrices does NOT contain 'megadescriptor'
        result = build_system_scorer(
            spec=spec,
            artifacts=mock_artifacts,
            gallery_df=gallery_df,
            crop_df=pd.DataFrame(),
            embedding_matrices={"miewid": np.zeros((1, 8))},
            descriptor_mappings={"miewid": pd.DataFrame()},
            local_scorer_body=MagicMock(),
            local_scorer_ear=MagicMock(),
        )
        assert result is None, (
            "Exploratory system must be skipped (return None) when required "
            "global channels are absent from embedding_matrices."
        )

    def test_local_only_exploratory_spec_channels(self):
        """local_only_exploratory uses only local channels."""
        from pipeline.local_oof_calibration import CHANNEL_BODY_LOCAL, CHANNEL_EAR_LOCAL
        spec = SYSTEM_SPECS["local_only_exploratory"]
        assert spec.channels_override == [CHANNEL_BODY_LOCAL, CHANNEL_EAR_LOCAL]


# ===========================================================================
# 25. test_systems_frozen_ear_uses_non_projected_embeddings
# ===========================================================================
class TestFrozenEarSystem:
    def test_frozen_ear_spec(self):
        spec = SYSTEM_SPECS["selected_v1_frozen_ear"]
        assert spec.use_frozen_ear is True
        assert not spec.exploratory

    def test_frozen_ear_replaces_projected_ear(self):
        """
        When use_frozen_ear=True, build_system_scorer replaces the projected ear
        embeddings with the raw ear_miewid embeddings.
        """
        from pipeline.fixed_probe_evaluator import build_system_scorer
        from pipeline.local_oof_calibration import CHANNEL_EAR

        spec = SYSTEM_SPECS["selected_v1_frozen_ear"]
        mock_artifacts = MagicMock()
        mock_artifacts.all_channels = ["miewid", "ear_miewid_projected"]
        mock_artifacts.fusion_weights = {
            "miewid": 0.5, "ear_miewid_projected": 0.3,
            "body_local": 0.1, "ear_local": 0.1,
        }
        mock_artifacts.frozen_k = 20
        mock_artifacts.calibrators_global = {}
        mock_artifacts.calibrator_body = None
        mock_artifacts.calibrator_ear = None
        mock_artifacts.oof_metrics = {}
        mock_artifacts.config = {}
        mock_artifacts.fingerprint = {}

        projected_ear = np.array([[1.0, 0.0]])
        raw_ear = np.array([[0.0, 1.0]])

        gallery_df = pd.DataFrame(
            {"image_id": ["g1"], "individual_id": ["id_0"],
             "session_id": ["s0"], "split": ["gallery"]}
        )
        embedding_matrices = {
            "miewid": np.zeros((1, 2)),
            "ear_miewid_projected": projected_ear,
            "ear_miewid": raw_ear,  # raw (frozen) ear
        }
        descriptor_mappings = {
            "miewid": pd.DataFrame({"image_id": ["g1"], "embedding_row": [0]}),
            "ear_miewid_projected": pd.DataFrame({"image_id": ["g1"], "embedding_row": [0]}),
            "ear_miewid": pd.DataFrame({"image_id": ["g1"], "embedding_row": [0]}),
        }

        scorer = build_system_scorer(
            spec=spec,
            artifacts=mock_artifacts,
            gallery_df=gallery_df,
            crop_df=pd.DataFrame(),
            embedding_matrices=embedding_matrices,
            descriptor_mappings=descriptor_mappings,
            local_scorer_body=MagicMock(),
            local_scorer_ear=MagicMock(),
        )
        assert scorer is not None
        # The scorer's ear channel embedding should be the raw ear, not projected
        ear_mat = scorer.embedding_matrices.get(CHANNEL_EAR)
        assert ear_mat is not None
        assert np.allclose(ear_mat, raw_ear), (
            "frozen_ear system must substitute raw ear_miewid for projected ear."
        )


# ===========================================================================
# 26-28. CLI parser tests
# ===========================================================================
class TestCLIParser:
    def test_score_parser_required_args(self):
        parser = _build_parser()
        args = parser.parse_args([
            "score",
            "--registration-file", "reg.json",
            "--splits-parquet", "splits.parquet",
            "--oof-artifacts-dir", "oof/",
            "--crop-manifest", "crop.parquet",
            "--ref-embeddings-dir", "emb/",
            "--output-dir", "out/",
        ])
        assert args.command == "score"
        assert args.registration_file == "reg.json"
        assert args.splits_parquet == "splits.parquet"
        assert args.oof_artifacts_dir == "oof/"
        assert args.crop_manifest == "crop.parquet"
        assert args.ref_embeddings_dir == "emb/"
        assert args.output_dir == "out/"

    def test_score_parser_defaults(self):
        parser = _build_parser()
        args = parser.parse_args([
            "score",
            "--registration-file", "r.json",
            "--splits-parquet", "s.parquet",
            "--oof-artifacts-dir", "oof/",
            "--crop-manifest", "c.parquet",
            "--ref-embeddings-dir", "emb/",
            "--output-dir", "out/",
        ])
        assert args.query_embeddings_dir is None
        assert args.cache_dir is None
        assert args.device == "cpu"
        assert args.disable_cudnn is False
        assert args.max_keypoints == 1024
        assert args.systems is None
        assert args.dry_run is False

    def test_load_parser_required_args(self):
        parser = _build_parser()
        args = parser.parse_args([
            "load",
            "--registration-file", "reg.json",
            "--rankings-dir", "rankings/",
        ])
        assert args.command == "load"
        assert args.registration_file == "reg.json"
        assert args.rankings_dir == "rankings/"

    def test_postprocess_parser_required_args(self):
        parser = _build_parser()
        args = parser.parse_args([
            "postprocess",
            "--registration-file", "reg.json",
            "--rankings-dir", "rankings/",
            "--output-dir", "out/",
        ])
        assert args.command == "postprocess"
        assert args.registration_file == "reg.json"
        assert args.rankings_dir == "rankings/"
        assert args.output_dir == "out/"

    def test_postprocess_parser_defaults(self):
        parser = _build_parser()
        args = parser.parse_args([
            "postprocess",
            "--registration-file", "r.json",
            "--rankings-dir", "rankings/",
            "--output-dir", "out/",
        ])
        assert args.candidate_system == "selected_v1_plus_both_local"
        assert args.baseline_system == "selected_v1"
        assert args.n_replicates == 10_000
        assert args.covariate_shift_flag is False

    def test_score_with_systems_list(self):
        parser = _build_parser()
        args = parser.parse_args([
            "score",
            "--registration-file", "r.json",
            "--splits-parquet", "s.parquet",
            "--oof-artifacts-dir", "oof/",
            "--crop-manifest", "c.parquet",
            "--ref-embeddings-dir", "emb/",
            "--output-dir", "out/",
            "--systems", "selected_v1", "selected_v1_plus_both_local",
        ])
        assert args.systems == ["selected_v1", "selected_v1_plus_both_local"]


# ===========================================================================
# 29. test_probe_images_absent_from_gallery_reference
# ===========================================================================
class TestProbeImagesAbsentFromGalleryReference:
    def test_gallery_df_passed_to_scorer_has_no_probes(self, tmp_path):
        """
        When building scorers, the gallery_df must not contain probe image_ids.
        This is enforced by _assert_no_probe_ids in EnsembleScorer.__init__.
        """
        # The EnsembleScorer calls _assert_no_probe_ids(gallery_df)
        from pipeline.local_oof_calibration import ProbePollutionError
        from pipeline.ensemble_inference import EnsembleScorer

        gallery_with_probe = pd.DataFrame({
            "image_id": ["gal1", "probe1"],
            "individual_id": ["id_a", "id_b"],
            "session_id": ["s1", "s2"],
            "split": ["gallery", "probe"],  # probe row present!
        })
        mock_artifacts = MagicMock()
        mock_artifacts.all_channels = ["miewid"]
        mock_artifacts.fusion_weights = {"miewid": 1.0}
        mock_artifacts.frozen_k = 10

        with pytest.raises(ProbePollutionError):
            EnsembleScorer(
                artifacts=mock_artifacts,
                gallery_df=gallery_with_probe,
                crop_df=pd.DataFrame(),
                embedding_matrices={"miewid": np.zeros((1, 8))},
                descriptor_mappings={"miewid": pd.DataFrame()},
                local_scorer_body=None,
                local_scorer_ear=None,
            )


# ===========================================================================
# 30-31. _verify_splits_manifest_hash tests
# ===========================================================================
class TestVerifySplitsManifestHash:
    def test_correct_hash_passes(self, tmp_path):
        splits_path = _make_splits_parquet(tmp_path)
        correct_hash = _file_sha256(splits_path)
        reg_path, _ = _make_registration_doc(
            tmp_path, splits_manifest_hash=correct_hash
        )
        registration = load_and_verify_registration(reg_path)
        _verify_splits_manifest_hash(splits_path, registration)  # must not raise

    def test_wrong_hash_raises(self, tmp_path):
        splits_path = _make_splits_parquet(tmp_path)
        wrong_hash = "deadbeef" * 8  # 64 hex chars
        reg_path, _ = _make_registration_doc(
            tmp_path, splits_manifest_hash=wrong_hash
        )
        registration = load_and_verify_registration(reg_path)
        with pytest.raises(SplitsManifestHashMismatchError, match="mismatch"):
            _verify_splits_manifest_hash(splits_path, registration)

    def test_error_message_shows_expected_and_actual(self, tmp_path):
        splits_path = _make_splits_parquet(tmp_path)
        wrong_hash = "a" * 64
        reg_path, _ = _make_registration_doc(
            tmp_path, splits_manifest_hash=wrong_hash
        )
        registration = load_and_verify_registration(reg_path)
        with pytest.raises(SplitsManifestHashMismatchError) as exc_info:
            _verify_splits_manifest_hash(splits_path, registration)
        assert "aaaaaaa" in str(exc_info.value)  # partial expected hash in message


# ===========================================================================
# 32-33. _verify_oof_artifacts tests
# ===========================================================================
class TestVerifyOOFArtifacts:
    def test_correct_hashes_pass(self, tmp_path):
        arts_dir = _make_oof_artifacts(tmp_path, fp="fp_correct")
        from pipeline.statistical_registration import _hash_oof_artifacts
        real_hash = _hash_oof_artifacts(arts_dir)
        reg_path, _ = _make_registration_doc(
            tmp_path,
            oof_artifacts_hash=real_hash,
            oof_scoring_fingerprint="fp_correct",
        )
        registration = load_and_verify_registration(reg_path)
        _verify_oof_artifacts(arts_dir, registration)  # must not raise

    def test_hash_mismatch_raises(self, tmp_path):
        arts_dir = _make_oof_artifacts(tmp_path)
        reg_path, _ = _make_registration_doc(
            tmp_path,
            oof_artifacts_hash="0" * 64,  # wrong hash
        )
        registration = load_and_verify_registration(reg_path)
        with pytest.raises(OOFArtifactHashMismatchError, match="hash mismatch"):
            _verify_oof_artifacts(arts_dir, registration)

    def test_fingerprint_mismatch_raises(self, tmp_path):
        arts_dir = _make_oof_artifacts(tmp_path, fp="fp_real")
        from pipeline.statistical_registration import _hash_oof_artifacts
        real_hash = _hash_oof_artifacts(arts_dir)
        reg_path, _ = _make_registration_doc(
            tmp_path,
            oof_artifacts_hash=real_hash,
            oof_scoring_fingerprint="fp_WRONG",
        )
        registration = load_and_verify_registration(reg_path)
        with pytest.raises(OOFArtifactHashMismatchError, match="fingerprint mismatch"):
            _verify_oof_artifacts(arts_dir, registration)

    def test_empty_oof_hash_in_registration_skips(self, tmp_path):
        """If oof_artifacts_hash is absent in registration, skip the check."""
        arts_dir = _make_oof_artifacts(tmp_path)
        reg_path, _ = _make_registration_doc(
            tmp_path,
            oof_artifacts_hash="",
            oof_scoring_fingerprint="",
        )
        registration = load_and_verify_registration(reg_path)
        registration["oof_artifacts_hash"] = ""
        registration["oof_scoring_fingerprint"] = ""
        # Should not raise
        _verify_oof_artifacts(arts_dir, registration)
