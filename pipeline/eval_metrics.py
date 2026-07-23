#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Endpoint metric helpers for fixed-probe statistical evaluation.

All functions are purely numeric / stateless — no I/O, no model loading.
They operate on lightweight plain-Python or NumPy structures so they are
safe to use in power simulation, bootstrap resampling, and reporting
without any pipeline dependency overhead.

Endpoint glossary
-----------------
reciprocal_rank(ranked_ids, truth_id)
    RR for a single query; truth identity counted once.

identity_macro_mrr(per_identity_rrs)
    Mean RR within each identity → equal-weight mean across identities.
    Primary fixed-probe endpoint.

query_weighted_mrr(rrs)
    Simple mean of per-query RR values.  This is the legacy 'mAP' metric
    (AP = RR when each query has exactly one true identity).  Reproduces
    the known selected-v1 mAP of 0.473 within tolerance.

identity_macro_top1 / identity_macro_top5
    Fraction of identities where the mean top-k hit rate > 0.

query_weighted_top1 / query_weighted_top5
    Query-level mean hit rates.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Tolerance for the selected-v1 legacy regression fixture
# ---------------------------------------------------------------------------

LEGACY_MAP_KNOWN_VALUE: float = 0.473
LEGACY_MAP_TOLERANCE: float = 1e-3   # ± 0.001 acceptable


# ---------------------------------------------------------------------------
# Per-query helpers
# ---------------------------------------------------------------------------

def reciprocal_rank(ranked_ids: Sequence[str], truth_id: str) -> float:
    """
    Compute reciprocal rank for a single query.

    Truth identity is counted exactly once; duplicate appearances are
    ignored.  Returns 0.0 if *truth_id* does not appear in *ranked_ids*.

    Parameters
    ----------
    ranked_ids : ordered candidate list, best first.
    truth_id   : single ground-truth identity for this query.
    """
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid == truth_id:
            return 1.0 / rank
    return 0.0


def top_k_hit(ranked_ids: Sequence[str], truth_id: str, k: int) -> float:
    """Return 1.0 if *truth_id* is in the top-*k* ranked identities."""
    return float(truth_id in ranked_ids[:k])


# ---------------------------------------------------------------------------
# Identity-macro metrics (primary endpoint)
# ---------------------------------------------------------------------------

def identity_macro_mrr(per_identity_rrs: Dict[str, List[float]]) -> float:
    """
    Identity-macro MRR.

    For each identity: compute mean RR across its queries.
    Return the equal-weight mean of those per-identity means.

    Parameters
    ----------
    per_identity_rrs : mapping from individual_id → list of per-query RR values.
    """
    if not per_identity_rrs:
        return 0.0
    means = [float(np.mean(rrs)) for rrs in per_identity_rrs.values() if rrs]
    if not means:
        return 0.0
    return float(np.mean(means))


def identity_macro_top_k(per_identity_hits: Dict[str, List[float]]) -> float:
    """
    Identity-macro top-k accuracy.

    Per-identity mean hit rate → equal-weight mean across identities.
    *per_identity_hits* is a dict of individual_id → list of 0/1 hit values.
    """
    if not per_identity_hits:
        return 0.0
    means = [float(np.mean(hits)) for hits in per_identity_hits.values() if hits]
    if not means:
        return 0.0
    return float(np.mean(means))


def identity_macro_top1(per_identity_hits: Dict[str, List[float]]) -> float:
    """Identity-macro top-1 accuracy (convenience wrapper)."""
    return identity_macro_top_k(per_identity_hits)


def identity_macro_top5(per_identity_hits: Dict[str, List[float]]) -> float:
    """Identity-macro top-5 accuracy (convenience wrapper)."""
    return identity_macro_top_k(per_identity_hits)


# ---------------------------------------------------------------------------
# Query-weighted metrics (secondary / legacy)
# ---------------------------------------------------------------------------

def query_weighted_mrr(rrs: Sequence[float]) -> float:
    """
    Query-weighted MRR (legacy 'mAP').

    When each query has exactly one true identity, average-precision collapses
    to reciprocal rank.  This reproduces the selected-v1 known_mAP = 0.473
    within LEGACY_MAP_TOLERANCE.

    Parameters
    ----------
    rrs : per-query reciprocal-rank values.
    """
    if not rrs:
        return 0.0
    return float(np.mean(rrs))


def query_weighted_top_k(hits: Sequence[float]) -> float:
    """Query-weighted top-k accuracy (simple mean of 0/1 hit values)."""
    if not hits:
        return 0.0
    return float(np.mean(hits))


def query_weighted_top1(hits: Sequence[float]) -> float:
    """Query-weighted top-1 accuracy."""
    return query_weighted_top_k(hits)


def query_weighted_top5(hits: Sequence[float]) -> float:
    """Query-weighted top-5 accuracy."""
    return query_weighted_top_k(hits)


# ---------------------------------------------------------------------------
# Aggregation helpers: build per-identity dicts from flat query records
# ---------------------------------------------------------------------------

def aggregate_per_identity_rrs(
    query_records: List[dict],
) -> Dict[str, List[float]]:
    """
    Group per-query RR values by individual identity.

    Each record must have keys:
      ``truth_individual_id``  – ground-truth identity (str or None)
      ``ranked_ids``           – ordered list of candidate identity strings
    Records where truth_individual_id is None are skipped.
    """
    per_id: Dict[str, List[float]] = {}
    for rec in query_records:
        truth = rec.get("truth_individual_id")
        if truth is None:
            continue
        rr = reciprocal_rank(rec["ranked_ids"], truth)
        per_id.setdefault(str(truth), []).append(rr)
    return per_id


def aggregate_per_identity_top_k(
    query_records: List[dict],
    k: int,
) -> Dict[str, List[float]]:
    """
    Group per-query top-k hits by individual identity.

    Same record format as aggregate_per_identity_rrs.
    """
    per_id: Dict[str, List[float]] = {}
    for rec in query_records:
        truth = rec.get("truth_individual_id")
        if truth is None:
            continue
        hit = top_k_hit(rec["ranked_ids"], truth, k)
        per_id.setdefault(str(truth), []).append(hit)
    return per_id


def compute_all_metrics(
    query_records: List[dict],
) -> dict:
    """
    Compute all fixed-probe endpoint metrics from a list of query records.

    Returns a dict with keys:
      identity_macro_mrr, query_weighted_mrr,
      identity_macro_top1, identity_macro_top5,
      query_weighted_top1, query_weighted_top5,
      n_queries, n_identities
    """
    per_id_rrs = aggregate_per_identity_rrs(query_records)
    per_id_t1 = aggregate_per_identity_top_k(query_records, k=1)
    per_id_t5 = aggregate_per_identity_top_k(query_records, k=5)

    rrs_flat = [rr for rrs in per_id_rrs.values() for rr in rrs]
    t1_flat = [h for hits in per_id_t1.values() for h in hits]
    t5_flat = [h for hits in per_id_t5.values() for h in hits]

    return {
        "identity_macro_mrr": identity_macro_mrr(per_id_rrs),
        "query_weighted_mrr": query_weighted_mrr(rrs_flat),
        "identity_macro_top1": identity_macro_top1(per_id_t1),
        "identity_macro_top5": identity_macro_top5(per_id_t5),
        "query_weighted_top1": query_weighted_top1(t1_flat),
        "query_weighted_top5": query_weighted_top5(t5_flat),
        "n_queries": len(query_records),
        "n_identities": len(per_id_rrs),
    }


def verify_legacy_map_regression(
    computed_mrr: float,
    tolerance: float = LEGACY_MAP_TOLERANCE,
) -> bool:
    """
    Return True if *computed_mrr* is within *tolerance* of the known
    selected-v1 query-weighted MRR (legacy mAP) value of 0.473.

    Raises ValueError with diagnostic if outside tolerance.
    """
    delta = abs(computed_mrr - LEGACY_MAP_KNOWN_VALUE)
    if delta > tolerance:
        raise ValueError(
            f"query_weighted_mrr regression failed: "
            f"got {computed_mrr:.4f}, expected {LEGACY_MAP_KNOWN_VALUE} "
            f"± {tolerance} (delta={delta:.4f})"
        )
    return True
