# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
import logging
import pstats
import cProfile
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_elephant import GT_COL, ID_COL, VIEWPOINT_COL, MATCH_ACCEPT_THRESHOLD
from utils.helpers_matching import print_memory_usage, log_to_file, restore_stdout
from utils.helpers_matching import load_data_dirs, load_metadata_file
from utils.helpers_matching import save_merged_accuracy_results

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_query_out_of_sample_records(metadata_table_ref, metadata_table_query):
    reference_ids = set(metadata_table_ref[GT_COL])

    for idx, row in metadata_table_query.iterrows():
        if row[GT_COL] in reference_ids:
            metadata_table_query.loc[idx, 'out_of_sample'] = 'False'
        else:
            metadata_table_query.loc[idx, 'out_of_sample'] = 'True'

    FP_table = metadata_table_query[(metadata_table_query['out_of_sample'] == 'True') & (metadata_table_query['matching_status'] == 'matched')]
    TN_table = metadata_table_query[(metadata_table_query['out_of_sample'] == 'True') & (metadata_table_query['matching_status'] == 'not_matched')]
    TP_table = metadata_table_query[(metadata_table_query['out_of_sample'] == 'False') & (metadata_table_query['matching_status'] == 'matched')]
    FN_table = metadata_table_query[(metadata_table_query['out_of_sample'] == 'False') & (metadata_table_query['matching_status'] == 'not_matched')]
    P_table  = metadata_table_query[metadata_table_query['matching_status'] == 'matched']
    N_table  = metadata_table_query[metadata_table_query['matching_status'] == 'not_matched']

    return metadata_table_query, P_table, N_table, FP_table, TN_table, TP_table, FN_table


def evaluate_accuracy_high_level(FP_table, TN_table, TP_table, FN_table):
    FP = len(FP_table)
    TN = len(TN_table)
    TP = len(TP_table)
    FN = len(FN_table)

    precision_pos = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    recall_pos    = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    f1_pos        = (2 * precision_pos * recall_pos) / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else np.nan
    accuracy      = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else np.nan

    precision_neg = TN / (TN + FN) if (TN + FN) > 0 else np.nan
    recall_neg    = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    f1_neg        = (2 * precision_neg * recall_neg) / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else np.nan

    metrics = {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'precision_positive': precision_pos,
        'recall_positive': recall_pos,
        'f1_score_positive': f1_pos,
        'precision_negative': precision_neg,
        'recall_negative': recall_neg,
        'f1_score_negative': f1_neg,
        'overall_accuracy': accuracy,
    }
    return metrics


def evaluate_accuracy_for_re_identified_items(TP_table):
    TP_table = TP_table.copy()
    TP_table['correctness'] = (TP_table[GT_COL] == TP_table['assigned_individual_id'])
    accuracy_matched = TP_table['correctness'].sum() / TP_table.shape[0] if TP_table.shape[0] > 0 else np.nan
    return accuracy_matched


def evaluate_accuracy_partitioning_new_items(N_table):
    a = N_table['assigned_individual_id']
    b = N_table[GT_COL]

    print("\n--- Info about Assigned IDs and Ground Truth Data ---")
    print(f"Total assigned IDs: {len(a)}")
    print(f"Total ground truth entries: {len(b)}")
    print("Assigned IDs:", a.values if len(a) <= 10 else f"{a.values[:10]} ... (truncated)")
    print("Ground Truth IDs:", b.values if len(b) <= 10 else f"{b.values[:10]} ... (truncated)")

    if not a.empty and not b.empty:
        ari_score = adjusted_rand_score(a, b)
        print("Adjusted Rand Index:", ari_score)
        return ari_score

    print("No data to compute Adjusted Rand Index.")
    return None


def compute_topk_accuracy(query_metadata, gt_col, pred_cols, k):
    """
    Returns fraction of rows where the gt value appears in any of the top-k pred columns.
    pred_cols should be an ordered list of column names (rank 1 first).
    """
    cols = pred_cols[:k]
    available = [c for c in cols if c in query_metadata.columns]
    if not available:
        return np.nan

    gt = query_metadata[gt_col]
    hit = pd.Series(False, index=query_metadata.index)
    for col in available:
        hit = hit | (query_metadata[col] == gt)

    return float(hit.sum()) / len(query_metadata) if len(query_metadata) > 0 else np.nan


def compute_map(query_metadata, gt_col, sim_col_prefix, k=3):
    """
    Mean Average Precision over all rows.
    For each query, builds a ranked list from columns '{sim_col_prefix}1' .. k
    and computes the AP using precision-at-rank where prediction is correct.
    """
    pred_id_cols  = [f"match_individual_{i}" for i in range(1, k + 1)]
    pred_sim_cols = [f"{sim_col_prefix}{i}" for i in range(1, k + 1)]

    available_id  = [c for c in pred_id_cols  if c in query_metadata.columns]
    if not available_id:
        return np.nan

    aps = []
    for _, row in query_metadata.iterrows():
        gt_val = row.get(gt_col)
        if pd.isna(gt_val):
            continue
        num_hits = 0
        ap = 0.0
        for rank_idx, col in enumerate(available_id, start=1):
            pred = row.get(col)
            if pred == gt_val:
                num_hits += 1
                ap += num_hits / rank_idx
        # normalise by 1 (only one correct individual per query)
        aps.append(ap)

    return float(np.mean(aps)) if aps else np.nan


def compute_ece(query_metadata, fused_sim_col, gt_col, n_bins=10):
    """
    Expected Calibration Error.
    Treats fused_sim as confidence that the top-1 match is correct.
    Lower ECE means the model's confidence is better calibrated.
    """
    if fused_sim_col not in query_metadata.columns:
        logger.warning("Column %s not found — skipping ECE computation.", fused_sim_col)
        return np.nan
    if 'match_individual_1' not in query_metadata.columns:
        return np.nan

    df = query_metadata[[fused_sim_col, gt_col, 'match_individual_1']].dropna()
    if df.empty:
        return np.nan

    confidences = df[fused_sim_col].astype(float).values
    corrects    = (df['match_individual_1'] == df[gt_col]).astype(float).values

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(confidences)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if not mask.any():
            continue
        bin_conf = confidences[mask].mean()
        bin_acc  = corrects[mask].mean()
        ece     += mask.sum() / n * abs(bin_conf - bin_acc)

    return float(ece)


def evaluate_by_viewpoint(query_metadata, gt_col, pred_col, viewpoint_col, match_viewpoint_col):
    """
    Splits pairs into same-view and cross-view based on query vs candidate viewpoint.
    Logs a warning if cross-view top-1 accuracy falls below 0.70.
    """
    result = {
        "same_view":  {"top1": np.nan, "n": 0},
        "cross_view": {"top1": np.nan, "n": 0},
    }

    required = [gt_col, pred_col, viewpoint_col, match_viewpoint_col]
    missing  = [c for c in required if c not in query_metadata.columns]
    if missing:
        logger.warning("Viewpoint evaluation skipped — missing columns: %s", missing)
        return result

    df = query_metadata[required].dropna()
    same  = df[df[viewpoint_col] == df[match_viewpoint_col]]
    cross = df[df[viewpoint_col] != df[match_viewpoint_col]]

    if len(same) > 0:
        result["same_view"]["top1"] = float((same[pred_col] == same[gt_col]).mean())
        result["same_view"]["n"]    = len(same)

    if len(cross) > 0:
        cross_top1 = float((cross[pred_col] == cross[gt_col]).mean())
        result["cross_view"]["top1"] = cross_top1
        result["cross_view"]["n"]    = len(cross)
        # Signal for model owners: cross-view accuracy below threshold needs investigation
        if cross_top1 < 0.70:
            logger.warning(
                "Cross-view top-1 accuracy is %.3f (< 0.70). "
                "Consider viewpoint-aware re-ranking or additional training data.",
                cross_top1,
            )

    return result


def ablation_summary(results_per_config, output_path):
    """
    Writes a comparison CSV with one row per config (e.g. global_only,
    global+local, full_fusion) and one column per metric.
    """
    rows = []
    for config_name, metrics in results_per_config.items():
        row = {"config": config_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Ablation summary saved to {output_path}")
    return df


def collect_accuracy_info(N_table, FP_table, TN_table, TP_table, FN_table, ref_data_size):
    accuracy_results = {
        'accept_threshold':              MATCH_ACCEPT_THRESHOLD,
        'total_items_in_ref':            ref_data_size,
        'total_number_of_queries':       len(FP_table) + len(TN_table) + len(TP_table) + len(FN_table),
        'total_number_of_pos_queries':   len(FN_table) + len(TP_table),
        'total_number_of_neg_queries':   len(TN_table) + len(FP_table),
    }

    accuracy_results.update(evaluate_accuracy_high_level(FP_table, TN_table, TP_table, FN_table))
    accuracy_results['accuracy_re_identified_items']      = evaluate_accuracy_for_re_identified_items(TP_table)
    accuracy_results['adjusted_rand_index_partitioning']  = evaluate_accuracy_partitioning_new_items(N_table)

    accuracy_results = pd.DataFrame(list(accuracy_results.items()), columns=['Metric', 'Value'])
    accuracy_results['Value'] = accuracy_results['Value'].apply(lambda x: round(x, 4) if isinstance(x, float) else x)
    print(accuracy_results)
    return accuracy_results


def main():
    """IMPORTANT: This code makes sense if we have accepted all results of the model in initial matching round."""
    profiler = cProfile.Profile()
    profiler.enable()

    root_dir, _ = load_data_dirs()

    log_file_std_output, log_file_err_output = log_to_file(root_dir, 'compute_accuracy')

    metadata_filepath_query = os.path.join(root_dir, 'query_dir', 'metadata_query.csv')
    query_metadata = load_metadata_file(metadata_filepath_query)
    query_metadata.drop(columns=['out_of_sample'], inplace=True, errors='ignore')

    metadata_filepath_ref = os.path.join(root_dir, 'reference_dir', 'metadata_reference.csv')
    ref_metadata   = load_metadata_file(metadata_filepath_ref)
    ref_data_size  = int(ref_metadata.shape[0])

    all_metrics = {}

    if GT_COL in ref_metadata.columns:
        if GT_COL in query_metadata.columns and 'assigned_individual_id' in query_metadata.columns:
            filtered = query_metadata[
                (query_metadata['assigned_individual_id'].notna()) &
                (query_metadata[GT_COL].notna()) &
                (query_metadata[GT_COL] != -1)
            ].copy()

            if len(filtered) != 0:
                metadata_table_query, _, N_table, FP_table, TN_table, TP_table, FN_table = \
                    find_query_out_of_sample_records(ref_metadata, filtered)
                metadata_table_query.to_csv(
                    os.path.join(root_dir, 'query_dir', 'metadata_query.csv'), index=False
                )

                accuracy_results = collect_accuracy_info(
                    N_table, FP_table, TN_table, TP_table, FN_table, ref_data_size
                )
                save_merged_accuracy_results(accuracy_results, root_dir, True)

                # Flatten high-level metrics into all_metrics dict for final CSV
                for _, row in accuracy_results.iterrows():
                    all_metrics[row['Metric']] = row['Value']
        else:
            print(f"Warning: {GT_COL} and/or assigned_individual_id columns not found in {metadata_filepath_query}.")
    else:
        print(f"Warning: {GT_COL} column not found in '{metadata_filepath_ref}'.")

    # -----------------------------------------------------------------------
    # New WildFusion-specific metrics
    # -----------------------------------------------------------------------
    pred_cols_all = [f"match_individual_{i}" for i in range(1, 4)]

    if GT_COL in query_metadata.columns:
        matched_rows = query_metadata[query_metadata['matching_status'] == 'matched'].copy() \
            if 'matching_status' in query_metadata.columns else query_metadata.copy()

        top1 = compute_topk_accuracy(matched_rows, GT_COL, pred_cols_all, k=1)
        top3 = compute_topk_accuracy(matched_rows, GT_COL, pred_cols_all, k=3)
        map3 = compute_map(matched_rows, GT_COL, sim_col_prefix='match_global_sim_', k=3)
        ece  = compute_ece(matched_rows, 'match_fused_sim_1', GT_COL)

        vp_result = evaluate_by_viewpoint(
            matched_rows, GT_COL,
            pred_col='match_individual_1',
            viewpoint_col=VIEWPOINT_COL,
            match_viewpoint_col='match_viewpoint_1',
        )

        all_metrics['top1_accuracy']          = round(top1, 4) if not np.isnan(top1) else top1
        all_metrics['top3_accuracy']          = round(top3, 4) if not np.isnan(top3) else top3
        all_metrics['mAP_at_3']               = round(map3, 4) if not np.isnan(map3) else map3
        all_metrics['ece_fused_sim_1']        = round(ece, 4)  if not np.isnan(ece)  else ece
        all_metrics['same_view_top1']         = vp_result['same_view']['top1']
        all_metrics['same_view_n']            = vp_result['same_view']['n']
        all_metrics['cross_view_top1']        = vp_result['cross_view']['top1']
        all_metrics['cross_view_n']           = vp_result['cross_view']['n']

        print("\n--- WildFusion Evaluation Metrics ---")
        print(f"  Top-1 accuracy:        {top1:.4f}" if not np.isnan(top1) else "  Top-1 accuracy: N/A")
        print(f"  Top-3 accuracy:        {top3:.4f}" if not np.isnan(top3) else "  Top-3 accuracy: N/A")
        print(f"  mAP@3:                 {map3:.4f}" if not np.isnan(map3) else "  mAP@3: N/A")
        print(f"  ECE (fused_sim_1):     {ece:.4f}"  if not np.isnan(ece)  else "  ECE: N/A")
        print(f"  Same-view top-1:       {vp_result['same_view']['top1']}  (n={vp_result['same_view']['n']})")
        print(f"  Cross-view top-1:      {vp_result['cross_view']['top1']}  (n={vp_result['cross_view']['n']})")

        cross_top1 = vp_result['cross_view']['top1']
        if not (isinstance(cross_top1, float) and np.isnan(cross_top1)) and cross_top1 < 0.70:
            print("WARNING: Cross-view top-1 accuracy is below 0.70 — review viewpoint handling.")

    # Save all metrics to a single results CSV
    if all_metrics:
        metrics_df = pd.DataFrame(list(all_metrics.items()), columns=['Metric', 'Value'])
        out_path = os.path.join(root_dir, 'query_dir', 'accuracy_results_full.csv')
        metrics_df.to_csv(out_path, index=False)
        print(f"\nAll metrics saved to {out_path}")

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

    print_memory_usage()

    restore_stdout(log_file_std_output, log_file_err_output)


if __name__ == "__main__":
    main()
