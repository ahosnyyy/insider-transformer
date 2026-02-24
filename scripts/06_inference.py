"""
06_inference.py — Score users with InsiderTransformerAE
========================================================
Thin CLI wrapper around src/inference/runner.py.

Usage:
    python scripts/06_inference.py                                # Score all users
    python scripts/06_inference.py --amp                          # Enable AMP
    python scripts/06_inference.py --user-id ACME/user123         # Score one user
    python scripts/06_inference.py --threshold 0.85               # Custom threshold
    python scripts/06_inference.py --top-k 20                     # Top 20 riskiest
    python scripts/06_inference.py --start-date 2011-03-01 --end-date 2011-04-01
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.common import load_config
from inference.runner import InferenceRunner


def parse_args():
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
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision (AMP) on GPU (disabled by default)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: outputs/inference_report.json)')
    args = parser.parse_args()

    if bool(args.start_date) != bool(args.end_date):
        parser.error("Both --start-date and --end-date are required together")

    return args


def main():
    args = parse_args()
    config = load_config()

    runner = InferenceRunner(
        config=config,
        user_id=args.user_id,
        start_date=args.start_date,
        end_date=args.end_date,
        threshold=args.threshold,
        threshold_method=args.threshold_method,
        top_k=args.top_k,
        batch_size=args.batch_size,
        use_amp=args.amp,
        output_path=args.output,
        project_root=PROJECT_ROOT,
    )
    runner.run()

    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE!")
    print("=" * 70)
    print("\nNext step: Review outputs/inference_report.json for risk scores and alerts")


if __name__ == "__main__":
    main()
