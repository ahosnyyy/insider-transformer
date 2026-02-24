"""
04_evaluate.py — Evaluate InsiderTransformerAE
================================================
Thin CLI wrapper around src/evaluation/evaluator.py.

Usage:
    python scripts/04_evaluate.py                              # Default
    python scripts/04_evaluate.py --amp                        # Enable AMP
    python scripts/04_evaluate.py --threshold best_f1
    python scripts/04_evaluate.py --load-scores                # Skip re-scoring
    python scripts/04_evaluate.py --use-augmented              # Augmented test set
    python scripts/04_evaluate.py --exclude-scenarios 3        # Exclude scenario
    python scripts/04_evaluate.py --dry-run                    # Dry-run checkpoints
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import load_config
from evaluation.evaluator import Evaluator


def parse_args():
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
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision (AMP) on GPU (disabled by default)')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()

    evaluator = Evaluator(
        config=config,
        batch_size=args.batch_size,
        threshold_method=args.threshold,
        load_scores=args.load_scores,
        use_augmented=args.use_augmented,
        exclude_scenarios=args.exclude_scenarios,
        use_amp=args.amp,
        dry_run=args.dry_run,
    )
    evaluator.run()

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print("\nNext step: Run python scripts/05_plot.py to generate plots")


if __name__ == "__main__":
    main()
