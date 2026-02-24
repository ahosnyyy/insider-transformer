"""
03_train.py — Train InsiderTransformerAE
=========================================
Thin CLI wrapper around src/training/trainer.py.

Usage:
    python scripts/03_train.py                               # Default config
    python scripts/03_train.py --epochs 100 --batch-size 128 # Override
    python scripts/03_train.py --amp                         # Enable AMP on GPU
    python scripts/03_train.py --resume                      # Resume training
    python scripts/03_train.py --dry-run                     # Quick pipeline test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import load_config
from training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train InsiderTransformerAE')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from best_model.pt checkpoint')
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision (AMP) on GPU (disabled by default)')
    parser.add_argument('--eval-interval', type=int, default=10,
                        help='Run evaluation on test set every N epochs')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='Save latest_model.pt every N epochs (default: 10, 0 to disable)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run one short epoch to test pipeline; outputs under outputs/dry_run/')
    parser.add_argument('--dry-run-steps', type=int, default=5,
                        help='Max train/val batches per phase in dry run')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()

    epochs = args.epochs or config['training']['epochs']
    batch_size = args.batch_size or config['training']['batch_size']
    lr = args.lr or config['training']['learning_rate']

    trainer = Trainer(
        config=config,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=args.patience,
        use_amp=args.amp,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        dry_run=args.dry_run,
        dry_run_steps=args.dry_run_steps,
        resume=args.resume,
    )
    trainer.run()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nNext step: Run python scripts/04_evaluate.py to evaluate the trained model")


if __name__ == "__main__":
    main()
