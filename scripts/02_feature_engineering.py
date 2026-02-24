#!/usr/bin/env python
"""
02_feature_engineering.py — Session-Aware Feature Engineering Pipeline

Step 1: Extract session-aware daily features from DuckDB events
        - Daily activity counts (18 features)
        - Session detection and per-session statistics (18 features)
        - Interaction, cyclical, and binary features (11 features)
        - Personality traits and context features (5 features)
        - Total: ~52 continuous features + 5 categorical embeddings

Step 2: Create 60-day sliding window sequences
        - User-based splitting (train/val/test = 70/15/15)
        - Stride of 5 days between sequences
        - Zero-padding for users with <60 active days

Step 3: Augment insider test sequences (optional)
        - Jittering: Gaussian noise on continuous features
        - Feature dropout: Randomly zero 10% of features
        - Scaling: Multiply by random factor [0.9, 1.1]
        - Creates 5 augmented copies per insider sequence

Output: models/<timestamp>/ with feature arrays and preprocessing artifacts
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.feature_engineering import run_feature_engineering
from src.data.sequence_creation import run_sequence_creation
from src.data.augmentation import run_augmentation


def parse_args():
    parser = argparse.ArgumentParser(
        description='Daily Feature Engineering for Insider Threat Detection'
    )
    parser.add_argument(
        '--no-augment', action='store_true',
        help='Skip test set augmentation'
    )
    parser.add_argument(
        '--skip-sequences', action='store_true',
        help='Only run feature extraction (skip sequence creation + augmentation)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    t_pipeline = time.time()

    print("=" * 70)
    print("INSIDER THREAT DETECTION — DAILY FEATURE ENGINEERING")
    print("=" * 70)

    # Step 1: Session-aware feature extraction
    print("\n[STEP 1/3] Session-aware daily feature extraction...")
    print("  Computing: daily counts + session features + interactions + context")
    t0 = time.time()
    try:
        output_dir = run_feature_engineering()
    except Exception as e:
        print(f"\n[ERROR] Feature extraction failed: {e}")
        sys.exit(1)
    print(f"  Step 1 done in {time.time() - t0:.1f}s")

    if args.skip_sequences:
        print("\n--skip-sequences flag set, stopping here.")
        return

    # Step 2: Sequence creation with user-based splitting
    print("\n[STEP 2/3] Sequence creation (60-day windows, stride=5)...")
    print("  Splitting: train/val/test = 70/15/15 (user-based)")
    t0 = time.time()
    try:
        run_sequence_creation()
    except Exception as e:
        print(f"\n[ERROR] Sequence creation failed: {e}")
        sys.exit(1)
    print(f"  Step 2 done in {time.time() - t0:.1f}s")

    # Step 3: Insider sequence augmentation
    if not args.no_augment:
        print("\n[STEP 3/3] Test set augmentation (insider sequences only)...")
        print("  Strategies: jittering, feature_dropout, scaling")
        t0 = time.time()
        try:
            from src.utils.common import load_config
            config = load_config()
            run_augmentation(output_dir, config)
        except Exception as e:
            print(f"\n[ERROR] Augmentation failed: {e}")
            sys.exit(1)
        print(f"  Step 3 done in {time.time() - t0:.1f}s")
    else:
        print("\n[STEP 3/3] Augmentation skipped (--no-augment)")

    total = time.time() - t_pipeline
    print("\n" + "=" * 70)
    print(f"DAILY FEATURE ENGINEERING COMPLETE! (total: {total:.1f}s / {total/60:.1f}m)")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("Next step: Run scripts/03_train.py to train the Transformer model")


if __name__ == "__main__":
    main()
