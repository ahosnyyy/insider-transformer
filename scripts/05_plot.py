"""
05_plot.py — Generate InsiderTransformerAE plots
==================================================
Thin CLI wrapper around src/visualization/plotter.py.

Usage:
    python scripts/05_plot.py                        # Default: augmented test labels
    python scripts/05_plot.py --no-augment            # Original test labels
    python scripts/05_plot.py --dry-run               # Plot from dry-run outputs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import load_config
from visualization.plotter import Plotter


def parse_args():
    parser = argparse.ArgumentParser(description='Generate InsiderTransformerAE plots')
    parser.add_argument('--no-augment', action='store_true',
                        help='Use original test labels instead of augmented (default: use augmented)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Plot from dry run outputs (outputs/dry_run/)')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()

    plotter = Plotter(
        config=config,
        no_augment=args.no_augment,
        dry_run=args.dry_run,
    )
    plotter.run()

    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE!")
    print("=" * 70)
    print("\nNext step: Run python scripts/06_inference.py to score users with the trained model")


if __name__ == "__main__":
    main()
