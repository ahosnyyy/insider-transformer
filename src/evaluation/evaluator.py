"""
Evaluation Module
==================
Evaluate trained model and compute metrics.

Key operations:
1. Load trained model and calculate reconstruction errors
2. Find F1-maximizing threshold (or use fixed threshold)
3. Compute metrics with Insider or Normal as positive class
4. Generate visualizations (confusion matrix, ROC, PR curves)
5. Per-scenario breakdown for insider types
6. Save all outputs to organized directories
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve, auc, precision_recall_curve
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.lstm_autoencoder import LSTMAutoencoder, create_model


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(config: dict, device: torch.device) -> LSTMAutoencoder:
    """Load trained model from checkpoint."""
    project_root = Path(__file__).parent.parent.parent
    checkpoint_path = project_root / "outputs" / "models" / "checkpoint_best.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"  Val loss: {checkpoint['val_loss']:.6f}")
    
    return model


def load_test_data(config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Load test data."""
    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / config['data']['processed_dir']
    
    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    
    return X_test, y_test


def calculate_reconstruction_errors(
    model: LSTMAutoencoder,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256
) -> np.ndarray:
    """Calculate reconstruction error for each sample."""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(device)
            error = model.get_reconstruction_error(batch)
            errors.append(error.cpu().numpy())
    
    return np.concatenate(errors)


def find_optimal_threshold(
    errors: np.ndarray,
    y_true: np.ndarray,
    positive_class: str = 'insider'
) -> Tuple[float, Dict[str, float]]:
    """
    Find threshold that maximizes F1 score for the specified positive class.
    
    Args:
        errors: Reconstruction errors
        y_true: Ground truth labels (1=Insider, 0=Normal)
        positive_class: 'insider' or 'normal'
    
    Returns:
        - Optimal threshold
        - Dictionary of metrics at that threshold
    """
    # Try different thresholds
    thresholds = np.percentile(errors, np.linspace(1, 99, 100))
    
    best_f1 = 0
    best_threshold = thresholds[len(thresholds)//2]
    best_metrics = {}
    
    for threshold in thresholds:
        if positive_class == 'insider':
            # High error = Insider (positive)
            y_pred = (errors > threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        else:
            # Low error = Normal (positive)
            # Flip: Normal=1, Insider=0
            y_pred = (errors <= threshold).astype(int)  # predict Normal
            y_true_flipped = 1 - y_true  # Normal=1, Insider=0
            f1 = f1_score(y_true_flipped, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'f1': f1
            }
    
    return best_threshold, best_metrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    errors: np.ndarray
) -> Dict[str, float]:
    """Compute all evaluation metrics with Insider as positive class."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Handle edge cases
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }
    
    # AUC
    fpr_curve, tpr_curve, _ = roc_curve(y_true, errors)
    metrics['auc'] = auc(fpr_curve, tpr_curve)
    
    return metrics


def compute_metrics_normal_positive(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    errors: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics with Normal as positive class.
    
    This flips the interpretation:
    - TP = Normal correctly predicted as Normal
    - FP = Insider incorrectly predicted as Normal
    - FN = Normal incorrectly predicted as Insider
    - TN = Insider correctly predicted as Insider
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Original: tn, fp, fn, tp (with Insider=1 as positive)
    tn_orig, fp_orig, fn_orig, tp_orig = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Flip: Normal (0) is now positive
    tp = tn_orig  # Normal predicted Normal
    fp = fn_orig  # Insider predicted Normal
    fn = fp_orig  # Normal predicted Insider
    tn = tp_orig  # Insider predicted Insider
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }
    
    # AUC (inverted - lower error = more likely normal)
    fpr_curve, tpr_curve, _ = roc_curve(1 - y_true, -errors)
    metrics['auc'] = auc(fpr_curve, tpr_curve)
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    positive_label: str = 'Insider'
):
    """
    Plot and save confusion matrix with positive class in top-left (TP).
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        output_path: Path to save the plot
        positive_label: 'Insider' or 'Normal' - determines matrix orientation
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Standard order: [[TN, FP], [FN, TP]] with Insider=1 as positive
    # We want positive class in top-left (TP position)
    if positive_label == 'Insider':
        # Flip to put Insider first: [[TP, FN], [FP, TN]]
        cm_display = cm[::-1, ::-1]
        labels = ['Insider', 'Normal']
    else:
        # Normal is positive, keep as-is: Normal (0) is already first
        # [[TN, FP], [FN, TP]] -> with Normal=pos: [[TP, FN], [FP, TN]]
        cm_display = cm
        labels = ['Normal', 'Insider']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_display, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({positive_label} = Positive)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_error_distribution(
    errors: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    output_path: Path
):
    """Plot reconstruction error distribution."""
    plt.figure(figsize=(10, 6))
    
    # Separate by class
    normal_errors = errors[y_true == 0]
    insider_errors = errors[y_true == 1]
    
    plt.hist(normal_errors, bins=100, alpha=0.7, label='Normal', density=True)
    plt.hist(insider_errors, bins=100, alpha=0.7, label='Insider', density=True)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    errors: np.ndarray,
    output_path: Path,
    positive_label: str = 'Insider'
):
    """Plot ROC curve."""
    if positive_label == 'Normal':
        fpr, tpr, _ = roc_curve(1 - y_true, -errors)
    else:
        fpr, tpr, _ = roc_curve(y_true, errors)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({positive_label} = Positive)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return roc_auc


def plot_pr_curve(
    y_true: np.ndarray,
    errors: np.ndarray,
    output_path: Path,
    positive_label: str = 'Insider'
):
    """Plot Precision-Recall curve."""
    if positive_label == 'Normal':
        # For Normal as positive, lower error = positive
        precision, recall, _ = precision_recall_curve(1 - y_true, -errors)
    else:
        # For Insider as positive, higher error = positive
        precision, recall, _ = precision_recall_curve(y_true, errors)
    
    # Calculate AUPRC
    auprc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUPRC = {auprc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({positive_label} = Positive)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return auprc


def compute_per_scenario_metrics(
    sessions_df,
    errors: np.ndarray,
    threshold: float
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics broken down by insider scenario.
    Only applies to insider sessions.
    """
    import pandas as pd
    
    # Get scenario labels
    scenarios = sessions_df['scenario'].values
    y_true = sessions_df['is_insider'].values
    y_pred = (errors > threshold).astype(int)
    
    scenario_metrics = {}
    unique_scenarios = [s for s in pd.unique(scenarios) if s is not None]
    
    for scenario in unique_scenarios:
        mask = scenarios == scenario
        if mask.sum() == 0:
            continue
            
        y_true_scenario = y_true[mask]
        y_pred_scenario = y_pred[mask]
        
        tp = ((y_pred_scenario == 1) & (y_true_scenario == 1)).sum()
        fn = ((y_pred_scenario == 0) & (y_true_scenario == 1)).sum()
        
        scenario_metrics[str(scenario)] = {
            'total_sessions': int(mask.sum()),
            'detected': int(tp),
            'missed': int(fn),
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
        }
    
    return scenario_metrics


def compute_user_level_metrics(
    sessions_df,
    errors: np.ndarray,
    threshold: float,
    positive_class: str = 'insider'
) -> Dict[str, float]:
    """
    Aggregate session predictions to user level.
    A user is predicted as insider if ANY of their sessions exceed threshold.
    """
    import pandas as pd
    
    # Create dataframe with predictions
    df = sessions_df[['user_id', 'is_insider']].copy()
    df['error'] = errors
    df['pred_insider'] = (errors > threshold).astype(int)
    
    # Aggregate to user level
    # User is insider if they have at least one insider session (ground truth)
    # User is predicted insider if at least one session exceeds threshold
    user_df = df.groupby('user_id').agg({
        'is_insider': 'max',  # 1 if any session is insider
        'pred_insider': 'max',  # 1 if any session predicted insider
        'error': 'max'  # Max error across sessions
    }).reset_index()
    
    y_true = user_df['is_insider'].values
    y_pred = user_df['pred_insider'].values
    
    if positive_class == 'normal':
        # Flip interpretation
        y_true = 1 - y_true
        y_pred = 1 - y_pred
    
    # Compute metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    metrics = {
        'total_users': len(user_df),
        'insider_users': int((user_df['is_insider'] == 1).sum()),
        'normal_users': int((user_df['is_insider'] == 0).sum()),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }
    
    return metrics


def print_comparison_with_paper(metrics: Dict[str, float]):
    """Print comparison with paper's results."""
    print("\n" + "=" * 60)
    print("COMPARISON WITH PAPER")
    print("=" * 60)
    print("\n{:<20} {:>15} {:>15}".format("Metric", "Ours", "Paper (CM)"))
    print("-" * 50)
    print("{:<20} {:>15.2f}% {:>15.2f}%".format("Accuracy", metrics['accuracy']*100, 90.62))
    print("{:<20} {:>15.2f}% {:>15.2f}%".format("Precision", metrics['precision']*100, 11.27))
    print("{:<20} {:>15.2f}% {:>15.2f}%".format("Recall", metrics['recall']*100, 3.50))
    print("{:<20} {:>15.2f}% {:>15.2f}%".format("F1 Score", metrics['f1']*100, 5.34))
    print("{:<20} {:>15.2f}% {:>15.2f}%".format("FPR", metrics['fpr']*100, 2.25))
    print("\n{:<20}".format("Confusion Matrix (Ours):"))
    print("  TN: {:,}  FP: {:,}".format(metrics['tn'], metrics['fp']))
    print("  FN: {:,}  TP: {:,}".format(metrics['fn'], metrics['tp']))
    print("\n{:<20}".format("Confusion Matrix (Paper):"))
    print("  TN: 180,732  FP: 4,163")
    print("  FN: 14,572   TP: 529")


def load_sessions_df(config: dict):
    """Load sessions dataframe for per-scenario and user-level metrics."""
    import pandas as pd
    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / config['data']['processed_dir']
    sessions_path = processed_dir / "sessions.parquet"
    
    if sessions_path.exists():
        return pd.read_parquet(sessions_path)
    return None


def save_evaluation_outputs(
    output_dir: Path,
    metrics: dict,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    errors: np.ndarray,
    threshold: float,
    positive_label: str,
    sessions_df=None,
    scenario_metrics: dict = None
):
    """Save all evaluation outputs to a directory."""
    import json
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics (both npy and JSON)
    metrics['threshold'] = threshold
    np.save(output_dir / "metrics.npy", metrics)
    
    # Save as JSON (human-readable)
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save plots
    plot_confusion_matrix(y_test, y_pred, output_dir / "confusion_matrix.png", positive_label)
    plot_error_distribution(errors, y_test, threshold, output_dir / "error_distribution.png")
    plot_roc_curve(y_test, errors, output_dir / "roc_curve.png", positive_label)
    plot_pr_curve(y_test, errors, output_dir / "pr_curve.png", positive_label)
    
    # Save per-scenario metrics (only for insider positive)
    if scenario_metrics:
        with open(output_dir / "per_scenario.json", 'w') as f:
            json.dump(scenario_metrics, f, indent=2)
    
    print(f"  Outputs saved to: {output_dir}")


def main(positive_class: str = 'both', exclude_scenarios: list = None, fixed_threshold: float = None):
    """
    Main evaluation entry point.
    
    Args:
        positive_class: 'insider', 'normal', or 'both'
        exclude_scenarios: List of scenario numbers to exclude (e.g., [3] or [2, 3])
        fixed_threshold: If set, use this threshold instead of finding optimal
    """
    from typing import List
    
    config = load_config()
    
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"  Positive class: {positive_class}")
    if exclude_scenarios:
        print(f"  Excluding scenarios: {exclude_scenarios}")
    if fixed_threshold is not None:
        print(f"  Using fixed threshold: {fixed_threshold}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(config, device)
    
    # Load test data
    print("\nLoading test data...")
    X_test, y_test = load_test_data(config)
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Insider samples: {y_test.sum():,}")
    print(f"  Normal samples: {(y_test == 0).sum():,}")
    
    # Load sessions for scenario metrics
    sessions_df = load_sessions_df(config)
    if sessions_df is not None:
        # Filter to test set indices (last portion of data)
        test_size = len(X_test)
        sessions_df = sessions_df.tail(test_size).reset_index(drop=True)
    
    # Calculate reconstruction errors
    print("\nCalculating reconstruction errors...")
    errors = calculate_reconstruction_errors(model, X_test, device)
    print(f"  Mean error (normal): {errors[y_test == 0].mean():.6f}")
    print(f"  Mean error (insider): {errors[y_test == 1].mean():.6f}")
    
    # Apply scenario exclusion if specified
    if exclude_scenarios and sessions_df is not None and 'scenario' in sessions_df.columns:
        # Create mask for samples to KEEP (not in excluded scenarios)
        exclude_mask = sessions_df['scenario'].isin([float(s) for s in exclude_scenarios])
        keep_mask = ~exclude_mask
        
        excluded_count = exclude_mask.sum()
        print(f"\n  Excluding {excluded_count:,} samples from scenarios {exclude_scenarios}")
        
        # Filter all arrays
        X_test = X_test[keep_mask]
        y_test = y_test[keep_mask]
        errors = errors[keep_mask]
        sessions_df = sessions_df[keep_mask].reset_index(drop=True)
        
        print(f"  Remaining samples: {len(X_test):,}")
        print(f"  Remaining insider samples: {y_test.sum():,}")
    
    # Setup output directories
    project_root = Path(__file__).parent.parent.parent
    eval_dir = project_root / "outputs" / "evaluation"
    
    # ===== INSIDER POSITIVE =====
    if positive_class in ['insider', 'both']:
        print("\n" + "=" * 60)
        print("METRICS (Insider = Positive Class)")
        print("=" * 60)
        
        # Find threshold for insider as positive
        if fixed_threshold is not None:
            threshold_insider = fixed_threshold
            print(f"  Using fixed threshold: {threshold_insider:.6f}")
        else:
            print("  Finding optimal threshold (maximizing Insider F1)...")
            threshold_insider, threshold_metrics = find_optimal_threshold(errors, y_test, 'insider')
            print(f"  Optimal threshold: {threshold_insider:.6f}")
            print(f"  F1 at threshold: {threshold_metrics['f1']:.4f}")
        
        # Predictions: high error = Insider
        y_pred_insider = (errors > threshold_insider).astype(int)
        
        metrics_insider = compute_metrics(y_test, y_pred_insider, errors)
        
        print(f"  Accuracy:  {metrics_insider['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics_insider['precision']*100:.2f}%")
        print(f"  Recall:    {metrics_insider['recall']*100:.2f}%")
        print(f"  F1 Score:  {metrics_insider['f1']*100:.2f}%")
        print(f"  AUC:       {metrics_insider['auc']:.4f}")
        print(f"  FPR:       {metrics_insider['fpr']*100:.2f}%")
        print(f"  Confusion Matrix:")
        print(f"    TN: {metrics_insider['tn']:,}  FP: {metrics_insider['fp']:,}")
        print(f"    FN: {metrics_insider['fn']:,}  TP: {metrics_insider['tp']:,}")
        
        # Per-scenario metrics (only for insider)
        scenario_metrics = None
        if sessions_df is not None and 'scenario' in sessions_df.columns:
            print("\n  Per-Scenario Breakdown:")
            scenario_metrics = compute_per_scenario_metrics(sessions_df, errors, threshold_insider)
            for scenario, sm in scenario_metrics.items():
                print(f"    {scenario}: {sm['detected']}/{sm['total_sessions']} detected (Recall: {sm['recall']*100:.1f}%)")
        
        # Save outputs
        print("\n  Saving outputs...")
        save_evaluation_outputs(
            eval_dir / "insider_positive",
            metrics_insider, y_test, y_pred_insider, errors, threshold_insider, 'Insider',
            sessions_df, scenario_metrics
        )
    
    # ===== NORMAL POSITIVE =====
    if positive_class in ['normal', 'both']:
        print("\n" + "=" * 60)
        print("METRICS (Normal = Positive Class)")
        print("=" * 60)
        
        # Find threshold for normal as positive
        if fixed_threshold is not None:
            threshold_normal = fixed_threshold
            print(f"  Using fixed threshold: {threshold_normal:.6f}")
        else:
            print("  Finding optimal threshold (maximizing Normal F1)...")
            threshold_normal, threshold_metrics = find_optimal_threshold(errors, y_test, 'normal')
            print(f"  Optimal threshold: {threshold_normal:.6f}")
            print(f"  F1 at threshold: {threshold_metrics['f1']:.4f}")
        
        # Predictions: low error = Normal (so y_pred for insider=1 is still error > threshold)
        y_pred_normal = (errors > threshold_normal).astype(int)
        
        metrics_normal = compute_metrics_normal_positive(y_test, y_pred_normal, errors)
        
        print(f"  Accuracy:  {metrics_normal['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics_normal['precision']*100:.2f}%")
        print(f"  Recall:    {metrics_normal['recall']*100:.2f}%")
        print(f"  F1 Score:  {metrics_normal['f1']*100:.2f}%")
        print(f"  AUC:       {metrics_normal['auc']:.4f}")
        print(f"  FPR:       {metrics_normal['fpr']*100:.2f}%")
        print(f"  Confusion Matrix (Normal=Pos):")
        print(f"    TN: {metrics_normal['tn']:,}  FP: {metrics_normal['fp']:,}")
        print(f"    FN: {metrics_normal['fn']:,}  TP: {metrics_normal['tp']:,}")
        
        # Save outputs
        print("\n  Saving outputs...")
        save_evaluation_outputs(
            eval_dir / "normal_positive",
            metrics_normal, y_test, y_pred_normal, errors, threshold_normal, 'Normal',
            sessions_df, None
        )
    
    # Comparison with paper (using insider positive)
    if positive_class in ['insider', 'both']:
        print_comparison_with_paper(metrics_insider)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"\nOutputs saved to: {eval_dir}")


if __name__ == "__main__":
    main()

