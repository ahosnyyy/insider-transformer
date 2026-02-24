"""
Visualization Module — InsiderTransformerAE
============================================
Plotter class for generating all evaluation and training plots.

All plots are derived from saved outputs — no model loading required.

Plots:
  1. Training / validation loss curve
  2. Learning rate schedule
  3. Score distribution (normal vs insider)
  4. Precision-Recall curve
  5. ROC curve
  6. Threshold method comparison
  7. Per-scenario breakdown
  8. Anomaly score scatter
  9. Confusion matrix (best_f1 threshold)
  10. Session-level confusion matrix
  11. Detection timeline with individual user plots

Used by scripts/05_plot.py as a thin CLI wrapper.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

DPI = 150

# Seaborn defaults
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


# =========================================================================
# Individual plot functions
# =========================================================================

def plot_loss_curve(history: dict, plots_dir: Path):
    """
    Plot training and validation loss curves with best epoch marker.

    Shows model convergence during training on session-aware features.

    Args:
        history: Training history dict with 'train_loss' and 'val_loss'
        plots_dir: Directory to save the plot

    Output:
        Saves 'loss_curve.png' with training/validation curves and best epoch marker
    """
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = np.arange(1, len(train_loss) + 1)
    best_epoch = int(np.argmin(val_loss)) + 1
    best_val = min(val_loss)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=epochs, y=train_loss, label='Train', linewidth=2, ax=ax, color='#3498db')
    sns.lineplot(x=epochs, y=val_loss, label='Validation', linewidth=2, ax=ax, color='#e74c3c')
    ax.scatter([best_epoch], [best_val], color='#2ecc71', s=100, zorder=5,
               label=f'Best epoch {best_epoch} ({best_val:.6f})', edgecolors='darkgreen', linewidths=1.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss (MSE)', fontsize=11)
    ax.set_title('Training / Validation Loss', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(plots_dir / 'loss_curve.png', dpi=DPI)
    plt.close(fig)
    print(f"  Saved: loss_curve.png")


def plot_lr_schedule(history: dict, plots_dir: Path):
    """
    Plot learning rate schedule across training epochs.

    Shows the cosine learning rate schedule used with AdamW optimizer
    during training of the session-aware Transformer model.

    Args:
        history: Training history dict with 'lr' list
        plots_dir: Directory to save the plot

    Output:
        Saves 'lr_schedule.png' showing learning rate progression
    """
    lr = history['lr']
    epochs = np.arange(1, len(lr) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x=epochs, y=lr, linewidth=2, ax=ax, color='#9b59b6')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Learning Rate', fontsize=11)
    ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))
    fig.tight_layout()
    fig.savefig(plots_dir / 'lr_schedule.png', dpi=DPI)
    plt.close(fig)
    print(f"  Saved: lr_schedule.png")


def plot_score_distribution(scores: np.ndarray, y: np.ndarray,
                            eval_results: dict, plots_dir: Path):
    """
    Plot anomaly score distribution for normal vs insider users.

    Shows how well the session-aware Transformer separates normal from anomalous
    behavior using reconstruction error on session-aware features.

    Args:
        scores: Anomaly scores from reconstruction error
        y: Binary labels (0=normal, 1=insider)
        eval_results: Evaluation results dict containing metrics
        plots_dir: Directory to save the plot

    Output:
        Saves 'score_distribution.png' with score distributions and statistics
    """
    normal_scores = scores[y == 0]
    insider_scores = scores[y == 1]

    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(scores.min(), np.percentile(scores, 99.5), 120)
    sns.histplot(normal_scores, bins=bins, kde=True, label='Normal',
                 color='#3498db', alpha=0.6, ax=ax, stat='density')
    sns.histplot(insider_scores, bins=bins, kde=True, label='Insider',
                 color='#e74c3c', alpha=0.6, ax=ax, stat='density')

    thresh_results = eval_results.get('threshold_results', {})
    best_f1 = thresh_results.get('best_f1', {})
    if best_f1:
        ax.axvline(best_f1['threshold'], color='red', linestyle='--', linewidth=1.5,
                    label=f"best_f1 = {best_f1['threshold']:.4f}")

    sep = eval_results.get('score_separation', {})
    if 'normal_p95' in sep:
        ax.axvline(sep['normal_p95'], color='green', linestyle=':', linewidth=1.5,
                    label=f"Normal P95 = {sep['normal_p95']:.4f}")

    gm = eval_results.get('global_metrics', {})
    text_parts = []
    if 'auprc' in gm:
        text_parts.append(f"AUPRC = {gm['auprc']:.4f}")
    if 'auroc' in gm:
        text_parts.append(f"AUROC = {gm['auroc']:.4f}")
    if text_parts:
        ax.text(0.97, 0.95, '\n'.join(text_parts), transform=ax.transAxes,
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel('Anomaly Score', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Anomaly Score Distribution', fontsize=12, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    fig.tight_layout()
    fig.savefig(plots_dir / 'score_distribution.png', dpi=DPI)
    plt.close(fig)
    print(f"  Saved: score_distribution.png")


def plot_pr_curve(scores: np.ndarray, y: np.ndarray,
                  eval_results: dict, plots_dir: Path):
    precision, recall, _ = precision_recall_curve(y, scores)
    auprc = eval_results.get('global_metrics', {}).get('auprc', 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.fill_between(recall, precision, alpha=0.2, color='#3498db')
    sns.lineplot(x=recall, y=precision, linewidth=2.5, ax=ax, color='#2980b9',
                 label=f'AUPRC = {auprc:.4f}')

    thresh_results = eval_results.get('threshold_results', {})
    markers = {'best_f1': 'o', 'percentile_99': 's', 'percentile_95': 'D',
               'mean_plus_3std': '^', 'fixed_recall_95': 'v', 'fixed_recall_99': 'P'}
    for method, marker in markers.items():
        if method in thresh_results:
            r = thresh_results[method]
            ax.scatter(r['recall'], r['precision'], marker=marker, s=70, zorder=5,
                       label=f"{method} (F1={r['f1']:.3f})", edgecolors='black', linewidths=0.5)

    baseline = y.mean()
    ax.axhline(baseline, color='grey', linestyle='--', linewidth=1.5, alpha=0.6,
               label=f'Baseline = {baseline:.4f}')
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(plots_dir / 'pr_curve.png', dpi=DPI)
    plt.close(fig)
    print(f"  Saved: pr_curve.png")


def plot_roc(scores: np.ndarray, y: np.ndarray,
             eval_results: dict, plots_dir: Path):
    fpr, tpr, _ = roc_curve(y, scores)
    auroc = eval_results.get('global_metrics', {}).get('auroc', 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.fill_between(fpr, tpr, alpha=0.2, color='#3498db')
    sns.lineplot(x=fpr, y=tpr, linewidth=2.5, ax=ax, color='#2980b9',
                 label=f'AUROC = {auroc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')

    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(plots_dir / 'roc_curve.png', dpi=DPI)
    plt.close(fig)
    print(f"  Saved: roc_curve.png")


def plot_threshold_comparison(eval_results: dict, plots_dir: Path):
    thresh_results = eval_results.get('threshold_results', {})
    if not thresh_results:
        print("  [skip] No threshold results available")
        return

    methods = list(thresh_results.keys())
    f1s = [thresh_results[m]['f1'] for m in methods]
    precisions = [thresh_results[m]['precision'] for m in methods]
    recalls = [thresh_results[m]['recall'] for m in methods]
    oracle_mask = [thresh_results[m].get('is_oracle', False) for m in methods]

    data = []
    for i, method in enumerate(methods):
        data.append({'Method': method, 'Metric': 'F1', 'Value': f1s[i], 'Oracle': oracle_mask[i]})
        data.append({'Method': method, 'Metric': 'Precision', 'Value': precisions[i], 'Oracle': oracle_mask[i]})
        data.append({'Method': method, 'Metric': 'Recall', 'Value': recalls[i], 'Oracle': oracle_mask[i]})

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.8), 5))

    sns.barplot(data=df, x='Method', y='Value', hue='Metric', ax=ax,
                palette=['#3498db', '#2ecc71', '#e67e22'])

    bars = ax.patches
    for i, is_oracle in enumerate(oracle_mask):
        if is_oracle:
            for j in range(3):
                idx = i * 3 + j
                if idx < len(bars):
                    bars[idx].set_hatch('//')
                    bars[idx].set_edgecolor('black')

    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Threshold Method Comparison  (// = oracle)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(title='Metric', loc='best', fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / 'threshold_comparison.png', dpi=DPI)
    plt.close(fig)
    print(f"  Saved: threshold_comparison.png")


def plot_scenario_breakdown(eval_results: dict, plots_dir: Path):
    scenario_data = eval_results.get('per_scenario_results')
    if not scenario_data:
        print("  [skip] No scenario data available")
        return

    names = list(scenario_data.keys())
    seq_recall = [scenario_data[n]['sequence_recall'] for n in names]
    user_recall = [scenario_data[n]['user_recall'] for n in names]
    insider_seqs = [scenario_data[n]['insider_sequences'] for n in names]

    data = []
    for i, name in enumerate(names):
        data.append({'Scenario': name.replace('scenario_', 'S'), 'Metric': 'Sequence Recall',
                     'Recall': seq_recall[i], 'Count': insider_seqs[i]})
        data.append({'Scenario': name.replace('scenario_', 'S'), 'Metric': 'User Recall',
                     'Recall': user_recall[i], 'Count': insider_seqs[i]})

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2), 5))

    sns.barplot(data=df, x='Scenario', y='Recall', hue='Metric', ax=ax,
                palette=['#3498db', '#e67e22'])

    bars = ax.patches
    for i, count in enumerate(insider_seqs):
        if i * 2 < len(bars):
            bar = bars[i * 2]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                   f'n={count}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Recall', fontsize=11)
    ax.set_title('Detection by Scenario', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.legend(title='Metric', loc='best', fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / 'scenario_breakdown.png', dpi=DPI)
    plt.close(fig)
    print(f"  Saved: scenario_breakdown.png")


def plot_score_scatter(scores: np.ndarray, y: np.ndarray,
                       eval_results: dict, plots_dir: Path):
    normal_idx = np.where(y == 0)[0]
    insider_idx = np.where(y == 1)[0]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(normal_idx, scores[normal_idx], c='#3498db', alpha=0.4,
               s=8, label='Normal', rasterized=True, edgecolors='none')
    ax.scatter(insider_idx, scores[insider_idx], c='#e74c3c', alpha=0.7,
               s=10, label='Insider', rasterized=True, edgecolors='none')

    thresh_results = eval_results.get('threshold_results', {})
    best_f1 = thresh_results.get('best_f1', {})
    if best_f1:
        ax.axhline(best_f1['threshold'], color='red', linestyle='--', linewidth=1.5,
                    label=f"Threshold = {best_f1['threshold']:.4f}")

    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Anomaly Score', fontsize=11)
    ax.set_title('Reconstruction-based Anomaly Scores', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(plots_dir / 'score_scatter.png', dpi=DPI)
    plt.close(fig)
    print(f"  Saved: score_scatter.png")


def plot_detection_timeline(eval_results: dict, plots_dir: Path):
    """
    Plot detection timeline for insider users with session-aware context.

    Shows when the model detected each insider user relative to their ground-truth
    activity window, leveraging session-aware features for precise temporal analysis.

    Generates both a consolidated timeline and individual per-user plots
    for detailed investigation of detection patterns.

    Args:
        eval_results: Evaluation results dict containing user breakdown data
        plots_dir: Directory to save plots

    Output:
        Saves 'detection_timeline.png' (consolidated) and individual plots
        in 'detection_timeline_individual/' subdirectory
    """
    pur = eval_results.get('per_user_results', [])
    if not pur:
        print("  [skip] No per_user_results")
        return

    users_with_ts = [u for u in pur
                     if u.get('first_insider_timestamp') and u['detected'] > 0]
    users_missed = [u for u in pur if u['detected'] == 0]

    all_plotable = users_with_ts + users_missed
    if not all_plotable:
        print("  [skip] No timestamp data in per_user_results")
        return

    individual_dir = plots_dir / 'detection_timeline_individual'
    individual_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Combined overview plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, max(4, len(all_plotable) * 0.6 + 1)))

    y_labels = []
    for i, ur in enumerate(all_plotable):
        uid = ur['user']
        y_labels.append(uid)

        gt_start = ur.get('ground_truth_start')
        gt_end = ur.get('ground_truth_end')
        if gt_start and gt_end:
            try:
                gs = np.datetime64(gt_start)
                ge = np.datetime64(gt_end)
                ax.barh(i, width=(ge - gs) / np.timedelta64(1, 'D'),
                        left=gs.astype('datetime64[D]').astype(float),
                        height=0.4, color='tab:blue', alpha=0.2, label='_nolegend_')
            except (ValueError, TypeError):
                pass

        fi_ts = ur.get('first_insider_timestamp')
        fd_ts = ur.get('first_detection_timestamp')

        if fi_ts and fi_ts != 'None':
            try:
                ts = np.datetime64(fi_ts)
                ax.plot(ts.astype('datetime64[D]').astype(float), i,
                        'o', color='tab:blue', markersize=8, zorder=5)
            except (ValueError, TypeError):
                pass

        if fd_ts and fd_ts != 'None':
            try:
                ts = np.datetime64(fd_ts)
                ax.plot(ts.astype('datetime64[D]').astype(float), i,
                        'X', color='tab:red', markersize=10, zorder=6)
            except (ValueError, TypeError):
                pass
        elif ur['detected'] == 0:
            ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] != ax.get_xlim()[0] else 0,
                    i, ' MISSED', va='center', fontsize=8, color='red')

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.invert_yaxis()

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue',
               markersize=8, label='First insider sequence'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='tab:red',
               markersize=10, label='First detection'),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=8)

    ax.set_xlabel('Date')
    ax.set_title('Detection Timeline by Insider User')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / 'detection_timeline.png', dpi=DPI)
    plt.close(fig)
    print(f"  Saved: detection_timeline.png")

    # ------------------------------------------------------------------
    # Individual plots per user
    # ------------------------------------------------------------------
    individual_count = 0
    for ur in all_plotable:
        uid = ur['user']
        gt_start = ur.get('ground_truth_start')
        gt_end = ur.get('ground_truth_end')
        fi_ts = ur.get('first_insider_timestamp')
        fd_ts = ur.get('first_detection_timestamp')

        if not (gt_start and gt_end):
            continue

        try:
            gs = np.datetime64(gt_start)
            ge = np.datetime64(gt_end)
        except (ValueError, TypeError):
            continue

        fig, ax = plt.subplots(figsize=(12, 3))

        window_days = (ge - gs) / np.timedelta64(1, 'D')
        ax.barh(0, width=window_days, left=gs.astype('datetime64[D]').astype(float),
                height=0.6, color='tab:blue', alpha=0.2, label='Ground-truth window')

        if fi_ts and fi_ts != 'None':
            try:
                ts = np.datetime64(fi_ts)
                ax.plot(ts.astype('datetime64[D]').astype(float), 0,
                        'o', color='tab:blue', markersize=12, zorder=5,
                        label='First insider sequence', markeredgecolor='darkblue', markeredgewidth=1.5)
            except (ValueError, TypeError):
                pass

        if fd_ts and fd_ts != 'None':
            try:
                ts = np.datetime64(fd_ts)
                ax.plot(ts.astype('datetime64[D]').astype(float), 0,
                        'X', color='tab:red', markersize=14, zorder=6,
                        label='First detection', markeredgecolor='darkred', markeredgewidth=1.5)

                if fi_ts and fi_ts != 'None':
                    try:
                        fi_dt = np.datetime64(fi_ts)
                        latency_h = float((ts - fi_dt) / np.timedelta64(1, 'h'))
                        latency_d = latency_h / 24.0
                        ax.text(0.02, 0.95, f'Detection latency: {latency_d:.1f} days ({latency_h:.1f} hours)',
                               transform=ax.transAxes, fontsize=10, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    except (ValueError, TypeError):
                        pass
            except (ValueError, TypeError):
                pass
        elif ur['detected'] == 0:
            ax.text(0.5, 0.5, 'MISSED', transform=ax.transAxes,
                   fontsize=16, color='red', ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        ax.set_yticks([0])
        ax.set_yticklabels([uid], fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)

        stats_text = []
        if 'detected' in ur and 'insider_sequences' in ur:
            stats_text.append(f"Detected: {ur['detected']}/{ur['insider_sequences']} sequences")
        if 'detection_rate' in ur:
            stats_text.append(f"Detection rate: {ur['detection_rate']:.1%}")
        if 'max_score' in ur:
            stats_text.append(f"Max score: {ur['max_score']:.4f}")
        if 'scenario' in ur:
            stats_text.append(f"Scenario: {ur.get('scenario', 'N/A')}")

        if stats_text:
            ax.text(0.98, 0.05, '\n'.join(stats_text), transform=ax.transAxes,
                   fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))

        ax.set_title(f'Detection Timeline: {uid}', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(axis='x', alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        fig.tight_layout()

        safe_uid = "".join(c for c in uid if c.isalnum() or c in ('-', '_'))
        fig.savefig(individual_dir / f'detection_timeline_{safe_uid}.png', dpi=DPI)
        plt.close(fig)
        individual_count += 1

    if individual_count > 0:
        print(f"  Saved: {individual_count} individual plots to detection_timeline_individual/")


def plot_confusion_matrix(eval_results: dict, plots_dir: Path):
    thresh_results = eval_results.get('threshold_results', {})
    best_f1 = thresh_results.get('best_f1')
    if not best_f1:
        print("  [skip] No best_f1 threshold results")
        return

    tp, fp, fn, tn = best_f1['tp'], best_f1['fp'], best_f1['fn'], best_f1['tn']
    cm = np.array([[tp, fn], [fp, tn]])
    labels = ['Insider', 'Normal']

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'}, linewidths=2, linecolor='gray')

    quadrant_labels = [['TP', 'FN'], ['FP', 'TN']]
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.7, quadrant_labels[i][j], ha='center', va='center',
                    fontsize=10, color='gray', fontweight='bold')

    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title(f"Confusion Matrix (best_f1, threshold={best_f1['threshold']:.4f})",
                 fontsize=13, fontweight='bold', pad=15)
    fig.tight_layout()
    fig.savefig(plots_dir / 'confusion_matrix.png', dpi=DPI)
    plt.close(fig)
    print(f"  Saved: confusion_matrix.png")


def plot_session_confusion_matrix(eval_results: dict, plots_dir: Path):
    """
    Plot session-level confusion matrix for precise threat localization.

    Shows how well the model identifies specific suspicious sessions within
    flagged days, leveraging session-aware features for fine-grained analysis.

    Args:
        eval_results: Evaluation results dict containing session_level_metrics
        plots_dir: Directory to save the plot

    Output:
        Saves 'session_confusion_matrix.png' with session-level confusion matrix
    """
    session_metrics = eval_results.get('session_level_metrics')
    if not session_metrics:
        print("  [skip] No session_level_metrics available")
        return

    total_insider = session_metrics.get('total_insider_sessions', 0)
    flagged_insider = session_metrics.get('flagged_insider_sessions', 0)
    total_flagged = session_metrics.get('total_flagged_sessions', 0)

    tp = flagged_insider
    fp = total_flagged - flagged_insider
    fn = total_insider - flagged_insider
    tn = 0  # Not directly available — sessions are only evaluated during insider period

    cm = np.array([[tp, fn], [fp, tn]])
    labels = ['Insider Session', 'Normal Session']

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(cm, annot=True, fmt=',d', cmap='Greens', ax=ax,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'}, linewidths=2, linecolor='gray')

    quadrant_labels = [['TP', 'FN'], ['FP', 'TN (N/A)']]
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.7, quadrant_labels[i][j], ha='center', va='center',
                    fontsize=10, color='gray', fontweight='bold')

    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title("Session-Level Confusion Matrix\n(Insider period sessions only)",
                 fontsize=13, fontweight='bold', pad=15)

    precision = session_metrics.get('session_precision', 0)
    recall = session_metrics.get('session_recall', 0)
    f1 = session_metrics.get('session_f1', 0)
    text = f"Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}"
    ax.text(1.15, 0.5, text, transform=ax.transAxes, fontsize=10,
            va='center', ha='left', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.tight_layout()
    fig.savefig(plots_dir / 'confusion_matrix_session.png', dpi=DPI)
    plt.close(fig)
    print(f"  Saved: confusion_matrix_session.png")


# =========================================================================
# Plotter class
# =========================================================================

class Plotter:
    """
    Plotter for InsiderTransformerAE.

    Loads saved training history, evaluation results, scores, and labels,
    then generates all 11 standard plots.

    Args:
        config: Configuration dictionary (from config.yaml)
        use_augmented: Use augmented test labels
        dry_run: Plot from dry-run outputs
        output_dir: Override output directory
        data_dir: Override data directory
    """

    def __init__(
        self,
        config: dict,
        use_augmented: bool = False,
        dry_run: bool = False,
        output_dir: Optional[Path] = None,
        data_dir: Optional[Path] = None,
    ):
        self.config = config
        self.use_augmented = use_augmented
        self.dry_run = dry_run

        project_root = Path(__file__).parent.parent.parent
        self.data_dir = data_dir or project_root / config['data']['processed_dir']
        self.output_dir = output_dir or project_root / 'outputs'
        if self.dry_run:
            self.output_dir = self.output_dir / 'dry_run'
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Load all data sources and generate all plots."""
        print("=" * 60)
        print("INSIDER TRANSFORMER AE — PLOTTING")
        print("=" * 60)
        if self.dry_run:
            print(f"  DRY RUN: Using outputs from {self.output_dir}")
        print(f"  Plots directory: {self.plots_dir}")

        # --------------------------------------------------------------
        # Load data sources (graceful on missing files)
        # --------------------------------------------------------------
        history = None
        hist_path = self.output_dir / 'training_history.json'
        if hist_path.exists():
            with open(hist_path, 'r') as f:
                history = json.load(f)
            print(f"  Loaded training_history.json ({len(history.get('train_loss', []))} epochs)")
        else:
            print("  [warn] training_history.json not found — loss/lr plots skipped")

        eval_results = {}
        eval_path = self.output_dir / 'evaluation_results.json'
        if eval_path.exists():
            with open(eval_path, 'r') as f:
                eval_results = json.load(f)
            print(f"  Loaded evaluation_results.json")
        else:
            print("  [warn] evaluation_results.json not found — some plots limited")

        scores = None
        scores_path = self.output_dir / 'test_anomaly_scores.npy'
        if scores_path.exists():
            scores = np.load(scores_path)
            print(f"  Loaded test_anomaly_scores.npy ({len(scores):,} scores)")
        else:
            print("  [warn] test_anomaly_scores.npy not found — score plots skipped")

        y = None
        label_name = 'y_test_aug.npy' if self.use_augmented else 'y_test.npy'
        label_path = self.data_dir / label_name
        if label_path.exists():
            y = np.load(label_path)
            print(f"  Loaded {label_name} ({len(y):,} labels, {int(y.sum()):,} insider)")
        else:
            print(f"  [warn] {label_name} not found — label-dependent plots skipped")

        # Auto-detect label mismatch
        if scores is not None and y is not None and len(scores) != len(y):
            alt_name = 'y_test_aug.npy' if not self.use_augmented else 'y_test.npy'
            alt_path = self.data_dir / alt_name
            if alt_path.exists():
                y_alt = np.load(alt_path)
                if len(y_alt) == len(scores):
                    y = y_alt
                    print(f"  [auto] Switched to {alt_name} ({len(y):,} labels, "
                          f"{int(y.sum()):,} insider) to match score count")
                else:
                    print(f"  [ERROR] Score/label length mismatch: {len(scores)} vs {len(y)}")
                    print(f"          Neither {label_name} nor {alt_name} matches. "
                          f"Skipping score-based plots.")
                    scores = None
            else:
                print(f"  [ERROR] Score/label length mismatch: {len(scores)} vs {len(y)}")
                print(f"          Skipping all score-based plots.")
                scores = None

        # --------------------------------------------------------------
        # Generate plots
        # --------------------------------------------------------------
        plot_count = 0

        print("\n[1/11] Loss curve...")
        if history:
            plot_loss_curve(history, self.plots_dir)
            plot_count += 1

        print("[2/11] Learning rate schedule...")
        if history:
            plot_lr_schedule(history, self.plots_dir)
            plot_count += 1

        print("[3/11] Score distribution...")
        if scores is not None and y is not None:
            plot_score_distribution(scores, y, eval_results, self.plots_dir)
            plot_count += 1

        print("[4/11] Precision-Recall curve...")
        if scores is not None and y is not None:
            plot_pr_curve(scores, y, eval_results, self.plots_dir)
            plot_count += 1

        print("[5/11] ROC curve...")
        if scores is not None and y is not None:
            plot_roc(scores, y, eval_results, self.plots_dir)
            plot_count += 1

        print("[6/11] Threshold comparison...")
        if eval_results:
            plot_threshold_comparison(eval_results, self.plots_dir)
            plot_count += 1

        print("[7/11] Scenario breakdown...")
        if eval_results:
            plot_scenario_breakdown(eval_results, self.plots_dir)
            plot_count += 1

        print("[8/11] Score scatter...")
        if scores is not None and y is not None:
            plot_score_scatter(scores, y, eval_results, self.plots_dir)
            plot_count += 1

        print("[9/11] Confusion matrix...")
        if eval_results:
            plot_confusion_matrix(eval_results, self.plots_dir)
            plot_count += 1

        print("[10/11] Session-level confusion matrix...")
        if eval_results:
            plot_session_confusion_matrix(eval_results, self.plots_dir)
            plot_count += 1

        print("[11/11] Detection timeline...")
        if eval_results:
            plot_detection_timeline(eval_results, self.plots_dir)
            plot_count += 1

        print("\n" + "=" * 60)
        print(f"PLOTTING COMPLETE — {plot_count}/11 plots generated")
        print("=" * 60)
        print(f"  Saved to: {self.plots_dir}")
