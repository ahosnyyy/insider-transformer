"""
Inference Script — InsiderTransformerAE
=======================================
Score users for insider threat risk using a trained model.

Runs the full pipeline: feature engineering → sequencing → scoring → report.
Uses saved preprocessing artifacts (scaler, label mappings) from training.

Usage:
    conda activate insider-threat
    python scripts/06_inference.py                          # score all users
    python scripts/06_inference.py --user-id ACME/user123   # score one user
    python scripts/06_inference.py --threshold 0.85         # custom threshold
    python scripts/06_inference.py --top-k 20               # show top 20 riskiest
"""

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Path setup — allow imports from src/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.feature_engineering import run_inference_feature_engineering
from data.sequence_creation import create_sequences_for_user
from evaluation.helpers import load_model, identify_risk_indicators
from evaluation.scoring import score_dataset
from utils.common import load_config


# =========================================================================
# Helpers
# =========================================================================

def load_artifacts(data_dir):
    """Load preprocessing artifacts saved during training."""
    artifacts_path = data_dir / 'preprocessing_artifacts.pkl'
    if not artifacts_path.exists():
        raise FileNotFoundError(
            f"Preprocessing artifacts not found at {artifacts_path}. "
            f"Run the training pipeline first (01 → 02 → 03)."
        )
    with open(artifacts_path, 'rb') as f:
        return pickle.load(f)


def load_threshold(output_dir, method='percentile_99'):
    """Load threshold from evaluation results, or compute from calibration scores.

    Priority:
      1. evaluation_results.json → thresholds → <method>
      2. calibration_scores.npy → compute percentile_99 / mean+3std
    """
    eval_path = output_dir / 'evaluation_results.json'
    if eval_path.exists():
        with open(eval_path, 'r') as f:
            results = json.load(f)
        thresholds = results.get('thresholds', {})
        if method in thresholds:
            return thresholds[method]['value']
        # Fallback: try any available threshold
        for m in ['best_f1', 'percentile_99', 'mean_plus_3std', 'percentile_95']:
            if m in thresholds:
                print(f"  [info] Requested '{method}' not found, using '{m}'")
                return thresholds[m]['value']

    # Fallback: compute from calibration scores
    cal_path = output_dir / 'calibration_scores.npy'
    if cal_path.exists():
        cal_scores = np.load(cal_path)
        if method == 'mean_plus_3std':
            threshold = float(cal_scores.mean() + 3 * cal_scores.std())
        else:
            threshold = float(np.percentile(cal_scores, 99))
        print(f"  Computed threshold from calibration scores: {threshold:.6f}")
        return threshold

    raise FileNotFoundError(
        f"No threshold source found. Run evaluation (04_evaluate.py) first, "
        f"or provide --threshold manually."
    )


def build_sequences_for_inference(X_cont, X_cat, user_ids, dates, lookback):
    """Build sliding-window sequences per user for inference.

    Uses stride=1 for maximum coverage (every possible window).

    Returns:
        X_seq_cont: (n_sequences, lookback, n_continuous)
        X_seq_cat: (n_sequences, lookback, n_categorical)
        seq_user_ids: (n_sequences,) user ID per sequence
        seq_dates: (n_sequences,) end-date per sequence
    """
    stride = 1  # dense coverage for inference
    unique_users = np.unique(user_ids)

    all_cont, all_cat, all_uids, all_dates = [], [], [], []

    for user in unique_users:
        mask = user_ids == user
        u_cont = X_cont[mask]
        u_cat = X_cat[mask]
        u_dates = dates[mask]
        # Dummy labels (not used for inference)
        u_labels = np.zeros(len(u_cont), dtype=np.int64)

        result = create_sequences_for_user(
            u_cont, u_cat, u_labels, lookback, stride,
            user_dates=u_dates,
        )
        sc, scat, _sy, sdates = result

        if sc is None:
            continue

        all_cont.append(sc)
        all_cat.append(scat)
        all_uids.extend([user] * len(sc))
        all_dates.append(sdates)

    if not all_cont:
        raise ValueError("No sequences could be created. Check data coverage.")

    return (
        np.concatenate(all_cont),
        np.concatenate(all_cat),
        np.array(all_uids),
        np.concatenate(all_dates),
    )


def build_user_report(user_id, user_scores, user_dates, threshold,
                      model, user_cont, user_cat, config, device):
    """Build a per-user risk report entry."""
    n_total = len(user_scores)
    n_flagged = int((user_scores >= threshold).sum())
    max_score = float(user_scores.max())
    mean_score = float(user_scores.mean())
    detection_rate = n_flagged / max(n_total, 1)

    # Severity classification
    if detection_rate > 0.5 or max_score > 2 * threshold:
        severity = "HIGH"
    elif detection_rate > 0.25:
        severity = "MEDIUM"
    elif n_flagged > 0:
        severity = "LOW"
    else:
        severity = "NORMAL"

    entry = {
        'user_id': user_id,
        'severity': severity,
        'anomaly_score_max': round(max_score, 6),
        'anomaly_score_mean': round(mean_score, 6),
        'total_sequences': n_total,
        'flagged_sequences': n_flagged,
        'detection_rate': round(detection_rate, 4),
    }

    # Top anomalous windows
    top_n = min(5, n_total)
    top_idx = np.argsort(user_scores)[::-1][:top_n]
    entry['top_anomalous_windows'] = [
        {
            'score': round(float(user_scores[i]), 6),
            'date': str(user_dates[i]),
        }
        for i in top_idx
    ]

    # Risk indicators (from anomalous sequences)
    if n_flagged > 0:
        anomalous_mask = user_scores >= threshold
        sample_cont = user_cont[anomalous_mask][:min(5, n_flagged)]
        sample_cat = user_cat[anomalous_mask][:min(5, n_flagged)]
        entry['risk_indicators'] = identify_risk_indicators(
            model, sample_cont, sample_cat, config, device
        )
    else:
        entry['risk_indicators'] = []

    return entry


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Insider Threat Inference — Score users with trained model'
    )
    parser.add_argument('--user-id', type=str, default=None,
                        help='Score a specific user (e.g., ACME/user123)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date for scoring window (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for scoring window (YYYY-MM-DD)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override anomaly threshold (default: load from eval results)')
    parser.add_argument('--threshold-method', type=str, default='best_f1',
                        choices=['best_f1', 'percentile_99', 'percentile_95', 'mean_plus_3std'],
                        help='Threshold method to load (default: best_f1)')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Show top K riskiest users in console output')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for scoring')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: outputs/inference_report.json)')
    args = parser.parse_args()

    config = load_config()
    data_dir = PROJECT_ROOT / config['data']['processed_dir']
    output_dir = PROJECT_ROOT / 'outputs'
    db_path = PROJECT_ROOT / config['data']['duckdb_path']

    print("=" * 60)
    print("INSIDER TRANSFORMER AE — INFERENCE")
    print("=" * 60)

    # ------------------------------------------------------------------
    # [1] Load model + artifacts + threshold
    # ------------------------------------------------------------------
    print("\n[1] Loading model and preprocessing artifacts...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model, n_continuous, _ckpt = load_model(output_dir, device, config)
    artifacts = load_artifacts(data_dir)

    if args.threshold is not None:
        threshold = args.threshold
        print(f"  Threshold (manual): {threshold:.6f}")
    else:
        threshold = load_threshold(output_dir, args.threshold_method)
        print(f"  Threshold ({args.threshold_method}): {threshold:.6f}")

    # ------------------------------------------------------------------
    # [2] Feature engineering (reuse SQL pipeline with saved scaler)
    # ------------------------------------------------------------------
    user_filter = [args.user_id] if args.user_id else None
    date_range = None
    if args.start_date and args.end_date:
        date_range = (args.start_date, args.end_date)
    elif args.start_date or args.end_date:
        parser.error("Both --start-date and --end-date are required together")

    filter_parts = []
    if args.user_id:
        filter_parts.append(f"user '{args.user_id}'")
    if date_range:
        filter_parts.append(f"{date_range[0]} to {date_range[1]}")
    filter_desc = f" for {', '.join(filter_parts)}" if filter_parts else " for all users"

    print(f"\n[2] Running feature engineering{filter_desc}...")
    t0 = time.time()

    result = run_inference_feature_engineering(
        db_path, config, artifacts,
        user_filter=user_filter, date_range=date_range,
    )
    X_cont = result['X_continuous']
    X_cat = result['X_categorical']
    user_ids = result['user_ids']
    dates = result['dates']

    n_users = len(np.unique(user_ids))
    print(f"  Feature engineering: {len(X_cont):,} days, {n_users} users "
          f"({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # [3] Build sequences
    # ------------------------------------------------------------------
    lookback = config['model']['lookback']
    print(f"\n[3] Building sequences (lookback={lookback}, stride=1)...")
    t0 = time.time()

    X_seq_cont, X_seq_cat, seq_uids, seq_dates = build_sequences_for_inference(
        X_cont, X_cat, user_ids, dates, lookback,
    )
    print(f"  Sequences: {len(X_seq_cont):,} ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # [4] Score
    # ------------------------------------------------------------------
    print(f"\n[4] Scoring...")
    use_amp = torch.cuda.is_available() and not args.no_amp

    dataset = TensorDataset(
        torch.tensor(X_seq_cont, dtype=torch.float32),
        torch.tensor(X_seq_cat, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    scores = score_dataset(model, loader, device, desc="Inference", use_amp=use_amp)

    n_flagged_total = int((scores >= threshold).sum())
    print(f"  Flagged sequences: {n_flagged_total:,}/{len(scores):,} "
          f"({n_flagged_total/max(len(scores),1):.1%})")

    # ------------------------------------------------------------------
    # [5] Build per-user reports
    # ------------------------------------------------------------------
    print(f"\n[5] Building user reports...")
    unique_users = np.unique(seq_uids)
    user_reports = []

    for uid in unique_users:
        mask = seq_uids == uid
        u_scores = scores[mask]
        u_dates = seq_dates[mask]
        u_cont = X_seq_cont[mask]
        u_cat = X_seq_cat[mask]

        report = build_user_report(
            uid, u_scores, u_dates, threshold,
            model, u_cont, u_cat, config, device,
        )
        user_reports.append(report)

    # Sort by max anomaly score descending
    user_reports.sort(key=lambda r: r['anomaly_score_max'], reverse=True)

    # ------------------------------------------------------------------
    # [6] Output
    # ------------------------------------------------------------------
    flagged_users = [r for r in user_reports if r['severity'] != 'NORMAL']
    severity_counts = defaultdict(int)
    for r in flagged_users:
        severity_counts[r['severity']] += 1

    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Users scored:   {len(user_reports)}")
    print(f"  Users flagged:  {len(flagged_users)}")
    for sev in ['HIGH', 'MEDIUM', 'LOW']:
        if severity_counts[sev] > 0:
            print(f"    {sev}: {severity_counts[sev]}")
    print(f"  Threshold:      {threshold:.6f}")

    # Top-K riskiest users
    top_k = min(args.top_k, len(user_reports))
    print(f"\n  Top {top_k} riskiest users:")
    print(f"  {'Rank':<6} {'User':<25} {'Severity':<10} {'Max Score':<12} {'Flagged':<10}")
    print(f"  {'-'*63}")
    for i, r in enumerate(user_reports[:top_k]):
        print(f"  {i+1:<6} {r['user_id']:<25} {r['severity']:<10} "
              f"{r['anomaly_score_max']:<12.6f} {r['flagged_sequences']}/{r['total_sequences']}")

    # Save full report
    report_output = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'model': 'InsiderTransformerAE',
            'threshold': round(float(threshold), 6),
            'threshold_method': args.threshold_method if args.threshold is None else 'manual',
            'total_users': len(user_reports),
            'flagged_users': len(flagged_users),
            'total_sequences': int(len(scores)),
            'flagged_sequences': n_flagged_total,
        },
        'summary': {
            'by_severity': dict(severity_counts),
        },
        'users': user_reports,
    }

    output_path = Path(args.output) if args.output else output_dir / 'inference_report.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report_output, f, indent=2, default=str)
    print(f"\n  Report saved: {output_path}")

    # Also save raw scores for further analysis
    scores_path = output_path.parent / 'inference_scores.npz'
    np.savez(scores_path,
             scores=scores,
             user_ids=seq_uids,
             dates=seq_dates.astype(str))
    print(f"  Scores saved: {scores_path}")

    print(f"\nInference complete.")


if __name__ == "__main__":
    main()
