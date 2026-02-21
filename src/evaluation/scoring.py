"""
Scoring Module — Daily Version
==============================
Compute anomaly scores for InsiderTransformerAE (single forward pass).
"""

import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


@torch.no_grad()
def score_dataset(model, loader, device, desc="Scoring", use_amp=False):
    """Compute anomaly scores for a dataset with progress reporting.

    Args:
        model: InsiderTransformerAE model.
        loader: DataLoader yielding (x_cont, x_cat) batches.
        device: torch device.
        desc: Label for progress messages.
        use_amp: Enable float16 autocast on CUDA.
    Returns:
        Numpy array of per-sequence anomaly scores.
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
    """Compute AUC and AUPRC."""
    try:
        auc = roc_auc_score(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)
    except ValueError:
        # Handle case where y_true has only one class
        auc = 0.5
        auprc = 0.0
    return {'auc': auc, 'auprc': auprc}
