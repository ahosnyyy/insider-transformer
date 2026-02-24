"""
Scoring Module — Session-Aware Daily Features
===============================================
Compute anomaly scores for InsiderTransformerAE using session-aware features.

Scoring Method:
- Mean + Max reconstruction error across behavioral features
- Combines average pattern deviation with peak anomaly detection
- Excludes personality traits (context-only features) from scoring
- Single forward pass per sequence for efficient batch processing

Output:
- Per-sequence anomaly scores (higher = more anomalous)
- Used for threshold-based insider threat detection
"""

import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


@torch.no_grad()
def score_dataset(model, loader, device, desc="Scoring", use_amp=False):
    """
    Compute anomaly scores for sequences with session-aware features.
    
    Process:
    1. Forward pass through Transformer autoencoder
    2. Compute reconstruction error per sequence
    3. Combine mean + max error for robust scoring
    4. Return scores for threshold-based detection
    
    Args:
        model: InsiderTransformerAE model trained on normal behavior
        loader: DataLoader yielding (x_cont, x_cat) batches with session features
        device: torch device (CPU/GPU)
        desc: Label for progress messages
        use_amp: Enable float16 autocast on CUDA for faster inference
        
    Returns:
        scores: Numpy array of per-sequence anomaly scores (higher = more anomalous)
    """
    model.eval()
    all_scores = []
    n_done = 0
    n_total = len(loader.dataset)
    t0 = time.time()

    for batch_idx, (x_cont, x_cat) in enumerate(loader):
        x_cont = x_cont.to(device)
        x_cat = x_cat.to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            scores = model.get_anomaly_scores(x_cont, x_cat)
        all_scores.append(scores.cpu().numpy())
        n_done += len(scores)

        if n_done % max(n_total // 10, 2000) < len(scores) or batch_idx == len(loader) - 1:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (n_total - n_done) / rate if rate > 0 else 0
            print(f"    {desc}: {n_done:,}/{n_total:,} "
                  f"({n_done/n_total:.0%}) [{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    elapsed = time.time() - t0
    if n_total > 0:
        rate = n_total / elapsed if elapsed > 0 else float('inf')
        print(f"    {desc} complete: {n_total:,} sequences in {elapsed:.1f}s "
              f"({rate:.0f} seq/s)")

    if len(all_scores) > 0:
        return np.concatenate(all_scores)
    return np.array([])


def compute_ranking_metrics(y_true, y_scores):
    """
    Compute ranking metrics for anomaly detection performance.
    
    AUC (Area Under ROC Curve): Measures overall ranking ability
    AUPRC (Area Under Precision-Recall Curve): More informative for imbalanced datasets
    
    Args:
        y_true: Binary labels (0=normal, 1=insider)
        y_scores: Anomaly scores from model (higher = more anomalous)
        
    Returns:
        dict: {'auc': float, 'auprc': float} ranking metrics
    """
    try:
        auc = roc_auc_score(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)
    except ValueError:
        # Handle case where y_true has only one class
        auc = 0.5
        auprc = 0.0
    return {'auc': auc, 'auprc': auprc}
