"""
Evaluation Script — InsiderTransformerAE
=========================================
Evaluate the trained model using reconstruction error (standalone ML).

Computes:
  1. Single-pass anomaly scores on test set
  2. Threshold-independent metrics (AUPRC, AUROC, score separation)
  3. Multiple threshold methods with per-threshold P/R/F1
  4. Per-user detection breakdown + aggregate user-level metrics
  5. Per-scenario breakdown (from DuckDB insiders table)
  6. Detection latency (sequence-based and time-based)
  7. SOC alert report (outputs/soc_report.json)
  8. Saves scores + metrics for plotting (05_plot.py)

Usage:
    python scripts/04_evaluate.py
    python scripts/04_evaluate.py --threshold best_f1
    python scripts/04_evaluate.py --load-scores              # skip scoring, load cached
    python scripts/04_evaluate.py --exclude-scenarios 3      # exclude scenario 3
    python scripts/04_evaluate.py --exclude-scenarios 2 3    # exclude scenarios 2 and 3
    python scripts/04_evaluate.py --dry-run                  # evaluate using dry run checkpoints
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import load_config
from evaluation.scoring import score_dataset
from evaluation.helpers import (
    load_model as _load_model,
    compute_calibration_scores,
    get_threshold,
    compute_threshold_metrics,
    compute_score_separation,
    per_user_breakdown,
    compute_latency_summary,
    user_level_metrics,
    load_user_scenarios,
    load_insider_metadata,
    per_scenario_breakdown,
    generate_soc_report,
    load_sessions_from_duckdb,
    compute_session_metrics,
)


def load_model(output_dir, device, config):
    """Load best trained model (thin wrapper for backward compatibility)."""
    model, n_continuous, _checkpoint = _load_model(output_dir, device, config)
    return model, n_continuous


# =========================================================================
# Main
# =========================================================================

class StepCounter:
    """Auto-incrementing step counter for clean output."""
    def __init__(self):
        self._n = 0
    def next(self, label):
        self._n += 1
        return f"[{self._n}] {label}"


def main():
    parser = argparse.ArgumentParser(description='Evaluate InsiderTransformerAE')
    parser.add_argument('--threshold', type=str, default='all',
                        choices=['all', 'best_f1', 'percentile_99', 'percentile_95',
                                 'mean_plus_3std', 'fixed_recall_95', 'fixed_recall_99'])
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--load-scores', action='store_true',
                        help='Load cached scores instead of re-scoring')
    parser.add_argument('--exclude-scenarios', type=int, nargs='+', default=None,
                        help='Scenario numbers to exclude (e.g. --exclude-scenarios 3)')
    parser.add_argument('--use-augmented', action='store_true',
                        help='Use augmented test set instead of original')
    parser.add_argument('--dry-run', action='store_true',
                        help='Evaluate using dry run checkpoints (outputs/dry_run/)')
    args = parser.parse_args()

    config = load_config()
    project_root = Path(__file__).parent.parent
    data_dir = project_root / config['data']['processed_dir']
    output_dir = project_root / 'outputs'
    if args.dry_run:
        output_dir = output_dir / 'dry_run'
    output_dir.mkdir(parents=True, exist_ok=True)

    step = StepCounter()

    print("=" * 60)
    print("INSIDER TRANSFORMER AE — EVALUATION")
    print("=" * 60)
    if args.dry_run:
        print(f"  DRY RUN: Using checkpoints from {output_dir}")
    if args.exclude_scenarios:
        print(f"  Excluding scenarios: {args.exclude_scenarios}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"  Device: {device}")
    if use_amp:
        print(f"  Mixed precision (AMP): enabled for inference")
    t_start = time.time()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"\n{step.next('Loading model...')}")
    model, n_continuous = load_model(output_dir, device, config)

    # ------------------------------------------------------------------
    # Load test data + scenario mapping
    # ------------------------------------------------------------------
    print(f"\n{step.next('Loading test data...')}")
    if args.use_augmented:
        print("  Using augmented test set")
        X_test_cont = np.load(data_dir / 'X_test_continuous_aug.npy')
        X_test_cat = np.load(data_dir / 'X_test_categorical_aug.npy')
        y_test = np.load(data_dir / 'y_test_aug.npy')
        user_ids_test = np.load(data_dir / 'user_ids_test.npy', allow_pickle=True)
    else:
        X_test_cont = np.load(data_dir / 'X_test_continuous.npy')
        X_test_cat = np.load(data_dir / 'X_test_categorical.npy')
        y_test = np.load(data_dir / 'y_test.npy')
        user_ids_test = np.load(data_dir / 'user_ids_test.npy', allow_pickle=True)

    user_scenario = load_user_scenarios(config)
    insider_meta = load_insider_metadata(config)

    # Load session data for session-level evaluation
    session_data = load_sessions_from_duckdb(config)
    if session_data is not None:
        print(f"  Loaded {len(session_data):,} sessions for session-level evaluation")
    else:
        print("  Session data not available (session-level metrics will be skipped)")

    # Load sequence-level timestamps (optional — generated by sequence_creation)
    ts_name = 'dates_test.npy'
    if args.use_augmented:
        ts_name = 'dates_test_aug.npy'
    ts_path = data_dir / ts_name
    timestamps_test = None
    if ts_path.exists():
        timestamps_test = np.load(ts_path, allow_pickle=True)
        print(f"  Loaded {ts_name} ({len(timestamps_test):,} timestamps)")
    else:
        print(f"  [info] {ts_name} not found — time-based latency will be skipped")

    # ------------------------------------------------------------------
    # Compute or load anomaly scores (before filtering so indices align)
    # ------------------------------------------------------------------
    test_scores_path = output_dir / 'test_anomaly_scores.npy'
    cal_scores_path = output_dir / 'calibration_scores.npy'

    if args.load_scores and test_scores_path.exists() and cal_scores_path.exists():
        print(f"\n{step.next('Loading cached scores...')}")
        test_scores = np.load(test_scores_path)
        cal_scores = np.load(cal_scores_path)
        print(f"  Loaded {len(test_scores):,} test + {len(cal_scores):,} calibration scores")
    else:
        print(f"\n{step.next('Computing anomaly scores (single-pass)...')}")
        test_dataset = TensorDataset(
            torch.tensor(X_test_cont, dtype=torch.float32),
            torch.tensor(X_test_cat, dtype=torch.long),
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        test_scores = score_dataset(model, test_loader, device, desc="Test", use_amp=use_amp)

        print(f"\n{step.next('Computing calibration scores...')}")
        cal_scores = compute_calibration_scores(model, data_dir, device, args.batch_size, use_amp=use_amp)

    # Cache unfiltered scores (before any exclusion filtering)
    np.save(output_dir / 'test_anomaly_scores.npy', test_scores)
    np.save(output_dir / 'calibration_scores.npy', cal_scores)

    # ------------------------------------------------------------------
    # Exclude scenarios (after scoring so cached scores stay aligned)
    # ------------------------------------------------------------------
    if args.exclude_scenarios and user_scenario is not None:
        excluded_users = {uid for uid, scen in user_scenario.items()
                         if scen in args.exclude_scenarios}
        keep_mask = np.array([str(u) not in excluded_users and u not in excluded_users
                              for u in user_ids_test])
        n_excluded = (~keep_mask).sum()
        print(f"  Excluding {n_excluded:,} sequences from scenarios {args.exclude_scenarios}")
        X_test_cont = X_test_cont[keep_mask]
        X_test_cat = X_test_cat[keep_mask]
        y_test = y_test[keep_mask]
        user_ids_test = user_ids_test[keep_mask]
        test_scores = test_scores[keep_mask]
        if timestamps_test is not None:
            timestamps_test = timestamps_test[keep_mask]

    n_insider = int(y_test.sum())
    n_normal = int((y_test == 0).sum())
    print(f"  Test: {len(y_test):,} sequences")
    print(f"  Insider: {n_insider:,} ({n_insider/len(y_test):.2%})")
    print(f"  Normal:  {n_normal:,} ({n_normal/len(y_test):.2%})")

    print(f"\n  Test  scores: mean={test_scores.mean():.6f}, std={test_scores.std():.6f}, "
          f"range=[{test_scores.min():.6f}, {test_scores.max():.6f}]")
    print(f"  Cal   scores: mean={cal_scores.mean():.6f}, std={cal_scores.std():.6f}, "
          f"range=[{cal_scores.min():.6f}, {cal_scores.max():.6f}]")

    # ------------------------------------------------------------------
    # Threshold-independent metrics (computed once)
    # ------------------------------------------------------------------
    print(f"\n{step.next('Threshold-independent metrics:')}")
    print("-" * 60)

    auprc = float(average_precision_score(y_test, test_scores))
    auroc = float(roc_auc_score(y_test, test_scores))
    print(f"  AUPRC  = {auprc:.4f}")
    print(f"  AUROC  = {auroc:.4f}")

    sep_stats = compute_score_separation(test_scores, y_test)
    print(f"\n  Score separation:")
    print(f"    Normal  — mean={sep_stats['normal_mean']:.6f}, "
          f"median={sep_stats['normal_median']:.6f}, std={sep_stats['normal_std']:.6f}")
    print(f"    Insider — mean={sep_stats['insider_mean']:.6f}, "
          f"median={sep_stats['insider_median']:.6f}, std={sep_stats['insider_std']:.6f}")
    print(f"    Mean ratio (insider/normal):   {sep_stats['mean_ratio']:.2f}x")
    print(f"    Median ratio (insider/normal): {sep_stats['median_ratio']:.2f}x")
    print(f"    Insider overlap below normal P95: {sep_stats['overlap_below_normal_p95']:.1%}")

    # ------------------------------------------------------------------
    # [6] Per-threshold metrics
    # ------------------------------------------------------------------
    oracle_methods = {'best_f1', 'fixed_recall_95', 'fixed_recall_99'}

    if args.threshold == 'all':
        methods = config.get('evaluation', {}).get('threshold_methods',
                    ['percentile_99', 'percentile_95', 'mean_plus_3std',
                     'best_f1', 'fixed_recall_95', 'fixed_recall_99'])
    else:
        methods = [args.threshold]

    print(f"\n{step.next('Per-threshold evaluation:')}")
    print("-" * 60)
    all_results = {}

    for method in methods:
        is_oracle = method in oracle_methods
        threshold = get_threshold(method, cal_scores, y_test, test_scores)
        y_pred = (test_scores >= threshold).astype(int)
        metrics = compute_threshold_metrics(y_test, y_pred)
        metrics['threshold'] = float(threshold)
        metrics['method'] = method
        metrics['is_oracle'] = is_oracle
        all_results[method] = metrics

        oracle_tag = " [oracle]" if is_oracle else ""
        print(f"\n  {method}{oracle_tag} (threshold={threshold:.6f}):")
        print(f"    F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  "
              f"R={metrics['recall']:.4f}")
        print(f"    FPR={metrics['fpr']:.4f}  FNR={metrics['fnr']:.4f}  "
              f"Spec={metrics['specificity']:.4f}")
        print(f"    TP={metrics['tp']}  FP={metrics['fp']}  "
              f"FN={metrics['fn']}  TN={metrics['tn']}")

    # ------------------------------------------------------------------
    # Per-user breakdown + user-level metrics
    # ------------------------------------------------------------------
    best_method = 'best_f1' if 'best_f1' in all_results else methods[0]
    best_threshold = all_results[best_method]['threshold']

    # Compute y_pred once for all breakdowns
    y_pred_best = (test_scores >= best_threshold).astype(int)

    print(f"\n{step.next(f'Per-user insider detection (threshold={best_method}):')}")
    print("-" * 60)
    user_results = per_user_breakdown(
        test_scores, y_test, y_pred_best, user_ids_test,
        timestamps=timestamps_test, insider_meta=insider_meta,
    )

    for ur in user_results:
        detected_str = f"{ur['detected']}/{ur['insider_sequences']}"
        if ur['latency_sequences'] >= 0:
            latency_str = f"latency={ur['latency_sequences']} seqs"
            if 'latency_hours' in ur and ur['latency_hours'] >= 0:
                latency_str += f" (~{ur['latency_hours']:.1f}h)"
        else:
            latency_str = "MISSED"
        print(f"  {ur['user']}: {detected_str} detected "
              f"(rate={ur['detection_rate']:.1%}), "
              f"max_score={ur['max_score']:.4f}, {latency_str}")

    n_caught = sum(1 for ur in user_results if ur['detected'] > 0)
    print(f"\n  Caught {n_caught}/{len(user_results)} insider users")

    latency_summary = compute_latency_summary(user_results)
    if 'mean_latency_hours' in latency_summary:
        print(f"  Mean latency: {latency_summary['mean_latency_hours']}h | "
              f"Median latency: {latency_summary['median_latency_hours']}h "
              f"(N={latency_summary['caught_users']} caught)")
    elif 'mean_latency_sequences' in latency_summary:
        print(f"  Mean latency: {latency_summary['mean_latency_sequences']:.1f} seqs "
              f"(N={latency_summary['caught_users']} caught)")

    # User-level aggregate metrics
    user_metrics_result = user_level_metrics(y_test, y_pred_best, user_ids_test)
    print(f"\n  User-level aggregate ({user_metrics_result['total_users']} users, "
          f"{user_metrics_result['insider_users']} insiders):")
    print(f"    P={user_metrics_result['precision']:.4f}  R={user_metrics_result['recall']:.4f}  "
          f"F1={user_metrics_result['f1']:.4f}")
    print(f"    TP={user_metrics_result['tp']}  FP={user_metrics_result['fp']}  "
          f"FN={user_metrics_result['fn']}  TN={user_metrics_result['tn']}")

    # ------------------------------------------------------------------
    # Per-scenario breakdown
    # ------------------------------------------------------------------
    print(f"\n{step.next('Per-scenario breakdown:')}")
    print("-" * 60)
    scenario_results = per_scenario_breakdown(y_test, y_pred_best, user_ids_test, user_scenario)
    if scenario_results:
        for scen_name, sr in scenario_results.items():
            print(f"  {scen_name}: "
                  f"sequences={sr['detected_sequences']}/{sr['insider_sequences']} "
                  f"({sr['sequence_recall']:.1%}), "
                  f"users={sr['users_detected']}/{sr['insider_users']} "
                  f"({sr['user_recall']:.1%})")
    else:
        print("  (not available)")

    # ------------------------------------------------------------------
    # Session-level metrics
    # ------------------------------------------------------------------
    session_metrics_result = None
    if session_data is not None:
        print(f"\n{step.next('Session-level evaluation:')}")
        print("-" * 60)
        session_metrics_result = compute_session_metrics(
            test_scores, y_test, y_pred_best, user_ids_test,
            timestamps_test, insider_meta, session_data, best_threshold
        )
        if session_metrics_result:
            sm = session_metrics_result
            print(f"  Session Precision: {sm['session_precision']:.4f}")
            print(f"  Session Recall:    {sm['session_recall']:.4f}")
            print(f"  Session F1:        {sm['session_f1']:.4f}")
            print(f"  Localization Acc:  {sm['session_localization_accuracy']:.4f}")
            print(f"  Insider sessions flagged: {sm['flagged_insider_sessions']}/{sm['total_insider_sessions']}")
            print(f"  Total flagged sessions: {sm['total_flagged_sessions']}")
        else:
            print("  Session metrics could not be computed (missing data)")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print(f"\n{step.next(f'Saving results to {output_dir}/ ... (total time: {elapsed:.1f}s)')}")

    eval_output = {
        'global_metrics': {
            'auprc': auprc,
            'auroc': auroc,
        },
        'score_separation': sep_stats,
        'threshold_results': all_results,
        'per_user_results': user_results,
        'user_level_metrics': user_metrics_result,
        'per_scenario_results': scenario_results,
        'detection_latency_summary': latency_summary,
        'session_level_metrics': session_metrics_result,
        'excluded_scenarios': args.exclude_scenarios,
        'use_augmented': args.use_augmented,
        'evaluation_time_seconds': round(elapsed, 1),
    }

    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(eval_output, f, indent=2)

    print("  Saved: test_anomaly_scores.npy")
    print("  Saved: calibration_scores.npy")
    print("  Saved: evaluation_results.json")

    # ------------------------------------------------------------------
    # SOC Alert Report
    # ------------------------------------------------------------------
    print(f"\n{step.next('Generating SOC alert report...')}")
    soc_report = generate_soc_report(
        user_results, test_scores, y_test, user_ids_test,
        timestamps_test, insider_meta, best_threshold, best_method,
        model, X_test_cont, X_test_cat, config, device, output_dir,
        session_data=session_data,
    )
    n_alerts = soc_report['summary']['total_alerts']
    sev = soc_report['summary']['by_severity']
    sev_str = ", ".join(f"{k}={v}" for k, v in sorted(sev.items()))
    print(f"  {n_alerts} alerts generated ({sev_str})")
    if soc_report['summary']['users_missed']:
        print(f"  Missed users: {soc_report['summary']['users_missed']}")
    print("  Saved: soc_report.json")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    bm = all_results[best_method]
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  AUPRC = {auprc:.4f}    AUROC = {auroc:.4f}")
    print(f"  Score ratio (insider/normal): {sep_stats['mean_ratio']:.2f}x")
    print(f"  Best threshold ({best_method}): {best_threshold:.6f}")
    print(f"  F1={bm['f1']:.4f}  P={bm['precision']:.4f}  R={bm['recall']:.4f}")
    print(f"  Users caught: {n_caught}/{len(user_results)}")
    if 'mean_latency_hours' in latency_summary:
        print(f"  Detection latency: {latency_summary['mean_latency_hours']}h mean, "
              f"{latency_summary['median_latency_hours']}h median")
    print(f"  SOC alerts: {n_alerts} ({sev_str})")
    if session_metrics_result:
        sm = session_metrics_result
        print(f"  Session F1={sm['session_f1']:.4f}  "
              f"Localization={sm['session_localization_accuracy']:.1%}")
    print(f"{'=' * 60}")
    print(f"\nEvaluation complete ({elapsed:.1f}s)! Next: python scripts/05_plot.py")



if __name__ == "__main__":
    main()
