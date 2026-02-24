"""
Sequence Creation Module — Session-Aware Daily Features
=======================================================
Creates 60-day sliding window sequences from session-aware daily features.

User-based splitting ensures no data leakage:
- Normal users: 70% train, 15% val, 15% test
- Insider users: 100% test (never seen during training)

Produces per split:
  - X_{split}_continuous.npy: (n_sequences, 60, ~52 features)
    Daily activity counts + session-derived features + interactions
  - X_{split}_categorical.npy: (n_sequences, 60, 5 features)
    User, PC, role, department, functional_unit embeddings
  - y_{split}.npy: (n_sequences,) binary labels (0=normal, 1=insider)
  - dates_{split}.npy: (n_sequences,) datetime64 — last date per sequence
  - user_ids_{split}.npy: (n_sequences,) user ID per sequence
"""

import gc
import pickle
from pathlib import Path

import numpy as np
import yaml

try:
    from ..utils.common import load_config
except (ImportError, ValueError):
    def load_config() -> dict:
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)


def split_users(user_ids, labels, config):
    """
    Deterministically split users into train/val/test sets.
    
    Splitting Strategy:
    - Normal users: Randomly shuffled (seed=42) then split 70/15/15
    - Insider users: ALL go to test set (never seen during training)
    - Ensures no data leakage between splits
    
    Args:
        user_ids: Array of user IDs per day
        labels: Array of binary labels per day (0=normal, 1=insider)
        config: Config dict with split ratios
        
    Returns:
        Tuple of sets: (train_users, val_users, test_normal_users, insider_users)
    """
    unique_users = np.unique(user_ids)
    insider_users = set(np.unique(user_ids[labels == 1]))
    normal_users = np.array([u for u in unique_users if u not in insider_users])

    rng = np.random.RandomState(42)
    rng.shuffle(normal_users)

    n_normal = len(normal_users)
    n_train = int(n_normal * config['split']['train'])
    n_val = int(n_normal * config['split']['val'])

    train_users = set(normal_users[:n_train])
    val_users = set(normal_users[n_train:n_train + n_val])
    test_normal_users = set(normal_users[n_train + n_val:])

    return train_users, val_users, test_normal_users, insider_users


def create_sequences_for_user(user_cont, user_cat, user_labels,
                               lookback, stride, user_dates=None):
    """
    Create sliding window sequences for a single user.
    
    Sequence Creation:
    - Sliding windows of length 'lookback' (default: 60 days)
    - Stride of 5 days between windows (overlap allowed)
    - Zero-padding for users with <60 active days
    - Binary label: 1 if any day in window is insider, else 0
    
    Args:
        user_cont: (n_days, n_continuous) continuous features
        user_cat: (n_days, n_categorical) categorical features
        user_labels: (n_days,) binary labels per day
        lookback: Window length in days (default: 60)
        stride: Step between windows (default: 5)
        user_dates: optional (n_days,) datetime64 per day
        
    Returns:
        Tuple of arrays:
        - sequences_cont: (n_seqs, lookback, n_continuous)
        - sequences_cat: (n_seqs, lookback, n_categorical)
        - sequence_labels: (n_seqs,) binary labels
        - sequence_dates: (n_seqs,) end dates (if user_dates provided)
    """
    n = len(user_cont)
    has_dates = user_dates is not None

    if n < lookback:
        # Zero-pad from the left for users with fewer active days
        pad_c = np.zeros((lookback - n, user_cont.shape[1]), dtype=np.float32)
        pad_cat = np.zeros((lookback - n, user_cat.shape[1]), dtype=np.int64)
        user_cont = np.vstack([pad_c, user_cont])
        user_cat = np.vstack([pad_cat, user_cat])
        user_labels = np.concatenate([np.zeros(lookback - n, dtype=np.int64), user_labels])
        if has_dates:
            nat_pad = np.full(lookback - n, np.datetime64('NaT'), dtype=user_dates.dtype)
            user_dates = np.concatenate([nat_pad, user_dates])
        n = lookback

    seqs_c, seqs_cat, seqs_y, seqs_dates = [], [], [], []
    for start in range(0, n - lookback + 1, stride):
        end = start + lookback
        seqs_c.append(user_cont[start:end])
        seqs_cat.append(user_cat[start:end])
        seqs_y.append(1 if user_labels[start:end].any() else 0)
        if has_dates:
            seqs_dates.append(user_dates[end - 1])

    if not seqs_c:
        if has_dates:
            return None, None, None, None
        return None, None, None

    result = (
        np.array(seqs_c, dtype=np.float32),
        np.array(seqs_cat, dtype=np.int64),
        np.array(seqs_y, dtype=np.int64),
    )
    if has_dates:
        return result + (np.array(seqs_dates),)
    return result


def run_sequence_creation() -> Path:
    """Run the full sequence creation pipeline."""
    config = load_config()
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / config['data']['processed_dir']

    print("\n" + "=" * 60)
    print("SEQUENCE CREATION (Daily)")
    print("=" * 60)

    # Load features
    print("\nLoading features...")
    X_cont = np.load(data_dir / 'features_continuous.npy')
    X_cat = np.load(data_dir / 'features_categorical.npy')
    labels = np.load(data_dir / 'labels.npy')
    user_ids = np.load(data_dir / 'user_ids.npy', allow_pickle=True)

    dates_path = data_dir / 'dates.npy'
    dates = None
    if dates_path.exists():
        dates = np.load(dates_path, allow_pickle=True)
        print(f"  Dates: {dates.shape} (dtype={dates.dtype})")
    else:
        print("  [warn] dates.npy not found — date propagation skipped")

    print(f"  Continuous: {X_cont.shape}")
    print(f"  Categorical: {X_cat.shape}")
    print(f"  Labels: {labels.shape} (insider={labels.sum():,})")

    lookback = config['model']['lookback']  # 60 days
    stride = config['processing']['sequence_stride']  # 5 days

    unique_users = np.unique(user_ids)
    train_users, val_users, test_normal_users, insider_users = split_users(
        user_ids, labels, config
    )

    print(f"\nSplit: {len(train_users)} train, {len(val_users)} val, "
          f"{len(test_normal_users)} test-normal, {len(insider_users)} insider users")

    # Process per-user and assign to splits
    has_dates = dates is not None
    split_keys = ['cont', 'cat', 'labels', 'uids']
    if has_dates:
        split_keys.append('dates')
    splits = {s: {k: [] for k in split_keys} for s in ['train', 'val', 'test']}

    print(f"Creating sequences (lookback={lookback}, stride={stride})...")
    for i, user in enumerate(unique_users):
        mask = user_ids == user
        u_cont = X_cont[mask]
        u_cat = X_cat[mask]
        u_labels = labels[mask]
        u_dates = dates[mask] if has_dates else None

        seq_result = create_sequences_for_user(
            u_cont, u_cat, u_labels, lookback, stride,
            user_dates=u_dates,
        )
        if has_dates:
            sc, scat, sy, sdates = seq_result
        else:
            sc, scat, sy = seq_result
            sdates = None

        if sc is None:
            continue

        # Determine split
        if user in insider_users:
            split = 'test'
        elif user in train_users:
            split = 'train'
        elif user in val_users:
            split = 'val'
        else:
            split = 'test'

        splits[split]['cont'].append(sc)
        splits[split]['cat'].append(scat)
        splits[split]['labels'].append(sy)
        splits[split]['uids'].extend([user] * len(sy))
        if has_dates and sdates is not None:
            splits[split]['dates'].append(sdates)

        if (i + 1) % 200 == 0:
            print(f"    Processed {i+1}/{len(unique_users)} users")

    # Free original arrays
    del X_cont, X_cat, labels
    gc.collect()

    # Concatenate and save splits
    print(f"\nSaving splits to {data_dir}/...")
    for name in ['train', 'val', 'test']:
        if splits[name]['cont']:
            sc = np.concatenate(splits[name]['cont'])
            scat = np.concatenate(splits[name]['cat'])
            sy = np.concatenate(splits[name]['labels'])
            su = np.array(splits[name]['uids'])
        else:
            sc = np.empty((0, lookback, X_cont.shape[1] if 'X_cont' in dir() else 0), dtype=np.float32)
            scat = np.empty((0, lookback, 0), dtype=np.int64)
            sy = np.empty(0, dtype=np.int64)
            su = np.empty(0)

        np.save(data_dir / f'X_{name}_continuous.npy', sc)
        np.save(data_dir / f'X_{name}_categorical.npy', scat)
        np.save(data_dir / f'y_{name}.npy', sy)
        np.save(data_dir / f'user_ids_{name}.npy', su)

        if has_dates and splits[name]['dates']:
            sdates = np.concatenate(splits[name]['dates'])
            np.save(data_dir / f'dates_{name}.npy', sdates)
            del sdates

        n_ins = sy.sum() if len(sy) > 0 else 0
        n_u = len(set(splits[name]['uids'])) if splits[name] is not None and splits[name]['uids'] else 0
        dates_str = " +dates" if has_dates else ""
        print(f"  {name}: {len(sy):,} sequences, {n_ins:,} insider, {n_u} users "
              f"| cont={sc.shape}{dates_str}")

        # Free
        del sc, scat, sy, su
        splits[name] = None
        gc.collect()

    # Verify train/val have no insiders
    y_train = np.load(data_dir / 'y_train.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    assert y_train.sum() == 0, "Train has insider sequences!"
    assert y_val.sum() == 0, "Val has insider sequences!"
    print("  ✓ Train and val contain only normal sequences")

    print("\nSequence creation complete!")
    return data_dir


if __name__ == "__main__":
    run_sequence_creation()
