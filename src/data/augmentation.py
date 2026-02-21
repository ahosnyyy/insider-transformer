"""
Data Augmentation Module — Daily Version
======================================
Applies augmentation strategies to insider sequences in the test set.

Strategies (adapted for daily features):
  - jittering: Gaussian noise on continuous features
  - feature_dropout: Randomly zero out 10% of continuous features
  - scaling: Multiply continuous features by random factor [0.9, 1.1]
  - time_warping: Not applicable for daily (no time_since_last_event)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset


def jittering(X: np.ndarray, noise_std: float = 0.05, rng: np.random.Generator = None) -> np.ndarray:
    """Add Gaussian noise to continuous features.

    Args:
        X: (n_sequences, seq_len, n_features)
        noise_std: Noise std relative to feature std
        rng: numpy random generator
    Returns:
        Augmented X
    """
    if rng is None:
        rng = np.random.default_rng(42)
    noise = rng.normal(0, noise_std * X.std(axis=(0, 1), keepdims=True), X.shape)
    return X + noise


def feature_dropout(X: np.ndarray, dropout_rate: float = 0.1, rng: np.random.Generator = None) -> np.ndarray:
    """Randomly zero out continuous features.

    Args:
        X: (n_sequences, seq_len, n_features)
        dropout_rate: Fraction of features to zero out
        rng: numpy random generator
    Returns:
        Augmented X
    """
    if rng is None:
        rng = np.random.default_rng(42)
    mask = rng.random((1, 1, X.shape[2])) > dropout_rate
    return X * mask


def scaling(X: np.ndarray, scale_range: List[float] = [0.9, 1.1], rng: np.random.Generator = None) -> np.ndarray:
    """Multiply continuous features by random factor.

    Args:
        X: (n_sequences, seq_len, n_features)
        scale_range: [min, max] for scaling factor
        rng: numpy random generator
    Returns:
        Augmented X
    """
    if rng is None:
        rng = np.random.default_rng(42)
    scale = rng.uniform(scale_range[0], scale_range[1])
    return X * scale


def augment_sequences(X_cont: np.ndarray, X_cat: np.ndarray, y: np.ndarray,
                    config: dict, rng: Optional[np.random.Generator] = None) -> tuple:
    """Apply enabled augmentation strategies to insider sequences.

    Args:
        X_cont: (n_sequences, seq_len, n_continuous)
        X_cat: (n_sequences, seq_len, n_categorical)
        y: (n_sequences,) labels
        config: config dict with augmentation settings
        rng: numpy random generator
    Returns:
        (X_cont_aug, X_cat_aug, y_aug) augmented arrays
    """
    if rng is None:
        rng = np.random.default_rng(42)

    aug_cfg = config.get('augmentation', {})
    if not aug_cfg.get('enabled', False):
        return X_cont, X_cat, y

    insider_mask = y == 1
    n_insider = insider_mask.sum()
    if n_insider == 0:
        return X_cont, X_cat, y

    multiplier = aug_cfg.get('insider_multiplier', 5)
    strategies = aug_cfg.get('strategies', {})

    # Create augmented copies
    X_cont_insider = X_cont[insider_mask]
    X_cat_insider = X_cat[insider_mask]
    y_insider = y[insider_mask]

    X_cont_aug_list = [X_cont_insider.copy()]
    X_cat_aug_list = [X_cat_insider.copy()]
    y_aug_list = [y_insider.copy()]

    for i in range(multiplier):
        aug_cont = X_cont_insider.copy()
        aug_cat = X_cat_insider.copy()

        # Jittering
        if strategies.get('jittering', {}).get('enabled', False):
            aug_cont = jittering(aug_cont,
                                noise_std=strategies['jittering'].get('noise_std', 0.05),
                                rng=rng)

        # Feature dropout
        if strategies.get('feature_dropout', {}).get('enabled', False):
            aug_cont = feature_dropout(aug_cont,
                                      dropout_rate=strategies['feature_dropout'].get('dropout_rate', 0.1),
                                      rng=rng)

        # Scaling
        if strategies.get('scaling', {}).get('enabled', False):
            aug_cont = scaling(aug_cont,
                             scale_range=strategies['scaling'].get('scale_range', [0.9, 1.1]),
                             rng=rng)

        X_cont_aug_list.append(aug_cont)
        X_cat_aug_list.append(aug_cat)
        y_aug_list.append(y_insider.copy())

    # Concatenate all augmented copies
    X_cont_aug = np.vstack(X_cont_aug_list)
    X_cat_aug = np.vstack(X_cat_aug_list)
    y_aug = np.concatenate(y_aug_list)

    # Concatenate with normal sequences
    X_cont_normal = X_cont[~insider_mask]
    X_cat_normal = X_cat[~insider_mask]
    y_normal = y[~insider_mask]

    X_cont_final = np.vstack([X_cont_normal, X_cont_aug])
    X_cat_final = np.vstack([X_cat_normal, X_cat_aug])
    y_final = np.concatenate([y_normal, y_aug])

    return X_cont_final, X_cat_final, y_final


def run_augmentation(data_dir: Path, config: dict) -> Path:
    """Run augmentation pipeline and save augmented test set.

    Args:
        data_dir: processed data directory
        config: config dict
    Returns:
        data_dir (same as input)
    """
    print("\n" + "=" * 60)
    print("DATA AUGMENTATION (Daily)")
    print("=" * 60)

    X_test_cont = np.load(data_dir / 'X_test_continuous.npy')
    X_test_cat = np.load(data_dir / 'X_test_categorical.npy')
    y_test = np.load(data_dir / 'y_test.npy')

    print(f"  Original test set: {len(y_test):,} sequences, {y_test.sum():,} insider")

    X_cont_aug, X_cat_aug, y_aug = augment_sequences(X_test_cont, X_test_cat, y_test, config)

    print(f"  Augmented test set: {len(y_aug):,} sequences, {y_aug.sum():,} insider")

    np.save(data_dir / 'X_test_continuous_aug.npy', X_cont_aug)
    np.save(data_dir / 'X_test_categorical_aug.npy', X_cat_aug)
    np.save(data_dir / 'y_test_aug.npy', y_aug)
    print(f"  Saved augmented files with '_aug' suffix")

    return data_dir


if __name__ == "__main__":
    from utils.common import load_config

    config = load_config()
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / config['data']['processed_dir']
    run_augmentation(data_dir, config)
