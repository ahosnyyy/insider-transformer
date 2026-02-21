"""
Shared Evaluation Helpers
=========================
Functions used by both 04_evaluate.py and 06_inference.py.

Contains: model loading, threshold computation, metrics, per-user/per-scenario
breakdowns, severity classification, risk indicator identification, SOC report
generation, and latency summaries.
"""

import json
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    precision_recall_curve, roc_curve,
    precision_score, recall_score, f1_score,
    confusion_matrix,
)

from models.transformer import create_model, count_parameters
from evaluation.scoring import score_dataset


# =========================================================================
# Model loading
# =========================================================================

def load_model(checkpoint_path, device, config):
    """Load trained InsiderTransformerAE from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file or directory containing best_model.pt
        device: torch device
        config: config dict

    Returns:
        (model, n_continuous)
    """
    if isinstance(checkpoint_path, (str, Path)):
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / 'best_model.pt'

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    n_continuous = checkpoint['n_continuous']
    cat_cardinalities = checkpoint['cat_cardinalities']
    behavioral_indices = checkpoint.get('behavioral_indices', None)

    model = create_model(config, n_continuous, cat_cardinalities, behavioral_indices)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Loaded best model (epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.6f})")
    print(f"  Parameters: {count_parameters(model):,}")
    if behavioral_indices is not None:
        print(f"  Behavioral features: {len(behavioral_indices)}/{n_continuous}")
    return model, n_continuous, checkpoint


def compute_calibration_scores(model, data_dir, device, batch_size=256, use_amp=False):
    """Compute scores on validation set for threshold calibration.

    Uses validation set (preferred) or falls back to training set.
    """
    for name in ['X_val_continuous.npy', 'X_train_continuous.npy']:
        cont_path = Path(data_dir) / name
        if cont_path.exists():
            break
    else:
        raise FileNotFoundError("No train or validation data found for calibration")

    cat_name = name.replace('continuous', 'categorical')
    source = 'validation' if 'val' in name else 'training'
    print(f"  Using {source} set for calibration")

    X_cont = np.load(Path(data_dir) / name)
    X_cat = np.load(Path(data_dir) / cat_name)

    dataset = TensorDataset(
        torch.tensor(X_cont, dtype=torch.float32),
        torch.tensor(X_cat, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return score_dataset(model, loader, device, desc="Calibration", use_amp=use_amp)


# =========================================================================
# Thresholds
# =========================================================================

def get_threshold(method, cal_scores, y_test=None, scores_test=None):
    """Compute threshold using specified method.

    Note: best_f1 / fixed_recall methods use test labels -- these are
    oracle/upper-bound thresholds, not deployable without labeled data.
    """
    if method == 'percentile_99':
        return np.percentile(cal_scores, 99)
    elif method == 'percentile_95':
        return np.percentile(cal_scores, 95)
    elif method == 'mean_plus_3std':
        return cal_scores.mean() + 3 * cal_scores.std()
    elif method == 'best_f1':
        assert y_test is not None and scores_test is not None
        precisions, recalls, thresholds = precision_recall_curve(y_test, scores_test)
        f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-10)
        return thresholds[np.argmax(f1s)]
    elif method == 'fixed_recall_95':
        assert y_test is not None and scores_test is not None
        _, tpr, thresholds = roc_curve(y_test, scores_test)
        idx = np.argmin(np.abs(tpr - 0.95))
        return thresholds[min(idx, len(thresholds) - 1)]
    elif method == 'fixed_recall_99':
        assert y_test is not None and scores_test is not None
        _, tpr, thresholds = roc_curve(y_test, scores_test)
        idx = np.argmin(np.abs(tpr - 0.99))
        return thresholds[min(idx, len(thresholds) - 1)]
    else:
        raise ValueError(f"Unknown threshold method: {method}")


# =========================================================================
# Metrics
# =========================================================================

def compute_threshold_metrics(y_true, y_pred):
    """Compute threshold-dependent metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        'fnr': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


def compute_score_separation(scores, y_true):
    """How well-separated are insider vs normal scores?"""
    normal_scores = scores[y_true == 0]
    insider_scores = scores[y_true == 1]

    stats = {
        'normal_mean': float(normal_scores.mean()),
        'normal_std': float(normal_scores.std()),
        'normal_median': float(np.median(normal_scores)),
        'insider_mean': float(insider_scores.mean()),
        'insider_std': float(insider_scores.std()),
        'insider_median': float(np.median(insider_scores)),
        'mean_ratio': float(insider_scores.mean() / max(normal_scores.mean(), 1e-10)),
        'median_ratio': float(np.median(insider_scores) / max(np.median(normal_scores), 1e-10)),
    }

    normal_p95 = np.percentile(normal_scores, 95)
    overlap_frac = float((insider_scores < normal_p95).mean())
    stats['overlap_below_normal_p95'] = overlap_frac
    stats['normal_p95'] = float(normal_p95)

    return stats


# =========================================================================
# Breakdowns
# =========================================================================

def per_user_breakdown(scores, y_true, y_pred, user_ids,
                       timestamps=None, insider_meta=None):
    """Per-user detection results for insider users.

    Args:
        timestamps: optional 1-D array of datetime64 per sequence.
        insider_meta: optional dict user_id -> {start_date, end_date, ...}
    """
    insider_users = np.unique(user_ids[y_true == 1])
    has_ts = timestamps is not None

    results = []
    for u in insider_users:
        mask = user_ids == u
        u_scores = scores[mask]
        u_labels = y_true[mask]
        u_preds = y_pred[mask]
        n_insider = int(u_labels.sum())
        n_detected = int((u_preds & u_labels).sum())

        insider_indices = np.where(u_labels == 1)[0]
        detected_insiders = np.where((u_preds == 1) & (u_labels == 1))[0]
        latency_seq = int(detected_insiders[0] - insider_indices[0]) if len(detected_insiders) > 0 else -1

        entry = {
            'user': str(u),
            'total_sequences': int(mask.sum()),
            'insider_sequences': n_insider,
            'detected': n_detected,
            'detection_rate': float(n_detected / max(n_insider, 1)),
            'max_score': float(u_scores.max()),
            'mean_score': float(u_scores.mean()),
            'latency_sequences': latency_seq,
        }

        if has_ts:
            u_ts = timestamps[mask]
            valid_ts = not np.all(np.isnat(u_ts))
            if valid_ts and len(insider_indices) > 0:
                first_insider_ts = u_ts[insider_indices[0]]
                entry['first_insider_timestamp'] = str(first_insider_ts)

                if len(detected_insiders) > 0:
                    first_det_ts = u_ts[detected_insiders[0]]
                    entry['first_detection_timestamp'] = str(first_det_ts)
                    delta = np.timedelta64(first_det_ts - first_insider_ts, 'h')
                    latency_h = float(delta / np.timedelta64(1, 'h'))
                    entry['latency_hours'] = latency_h
                    entry['latency_days'] = round(latency_h / 24.0, 2)
                else:
                    entry['first_detection_timestamp'] = None
                    entry['latency_hours'] = -1.0
                    entry['latency_days'] = -1.0
            else:
                entry['first_insider_timestamp'] = None
                entry['first_detection_timestamp'] = None
                entry['latency_hours'] = -1.0
                entry['latency_days'] = -1.0

        u_str = str(u)
        if insider_meta and u_str in insider_meta:
            im = insider_meta[u_str]
            entry['ground_truth_start'] = im.get('start_date')
            entry['ground_truth_end'] = im.get('end_date')
            entry['scenario'] = im.get('scenario', 0)
            entry['scenario_details'] = im.get('details', '')

            if has_ts and entry.get('first_detection_timestamp') and im.get('end_date'):
                try:
                    gt_end = np.datetime64(im['end_date'])
                    first_det = np.datetime64(entry['first_detection_timestamp'])
                    lead = float((gt_end - first_det) / np.timedelta64(1, 'h'))
                    entry['detection_lead_time_hours'] = round(lead, 1)
                except (ValueError, TypeError):
                    entry['detection_lead_time_hours'] = None

        results.append(entry)

    return results


def compute_latency_summary(user_results):
    """Aggregate detection latency statistics from per-user results."""
    latencies_seq = []
    latencies_hours = []
    caught = 0
    missed = 0

    for ur in user_results:
        if ur['detected'] > 0:
            caught += 1
            latencies_seq.append(ur['latency_sequences'])
            if 'latency_hours' in ur and ur['latency_hours'] >= 0:
                latencies_hours.append(ur['latency_hours'])
        else:
            missed += 1

    summary = {
        'caught_users': caught,
        'missed_users': missed,
        'total_insider_users': caught + missed,
    }

    if latencies_seq:
        summary['mean_latency_sequences'] = float(np.mean(latencies_seq))
        summary['median_latency_sequences'] = float(np.median(latencies_seq))

    if latencies_hours:
        summary['mean_latency_hours'] = round(float(np.mean(latencies_hours)), 1)
        summary['median_latency_hours'] = round(float(np.median(latencies_hours)), 1)
        summary['min_latency_hours'] = round(float(np.min(latencies_hours)), 1)
        summary['max_latency_hours'] = round(float(np.max(latencies_hours)), 1)

    return summary


def user_level_metrics(y_true, y_pred, user_ids):
    """Aggregate user-level P/R/F1: a user is insider if ANY sequence exceeds threshold."""
    all_users = np.unique(user_ids)

    user_true = []
    user_pred = []
    for u in all_users:
        mask = user_ids == u
        user_true.append(int(y_true[mask].max()))
        user_pred.append(int(y_pred[mask].max()))

    user_true = np.array(user_true)
    user_pred = np.array(user_pred)
    tn, fp, fn, tp = confusion_matrix(user_true, user_pred, labels=[0, 1]).ravel()

    return {
        'total_users': len(all_users),
        'insider_users': int(user_true.sum()),
        'normal_users': int((user_true == 0).sum()),
        'precision': float(precision_score(user_true, user_pred, zero_division=0)),
        'recall': float(recall_score(user_true, user_pred, zero_division=0)),
        'f1': float(f1_score(user_true, user_pred, zero_division=0)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


def load_user_scenarios(config, project_root=None):
    """Load user -> scenario mapping from DuckDB insiders table."""
    try:
        import duckdb
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        db_path = Path(project_root) / config['data']['duckdb_path']
        if not db_path.exists():
            return None

        conn = duckdb.connect(str(db_path), read_only=True)
        insiders = conn.execute(
            "SELECT user_id, scenario FROM insiders"
        ).fetchall()
        conn.close()

        return {uid: int(scen) if scen is not None else 0 for uid, scen in insiders}

    except ImportError:
        return None


def load_insider_metadata(config, project_root=None):
    """Load full insider metadata from DuckDB.

    Returns:
        dict keyed by user_id -> {scenario, start_date, end_date, details}
        or None if unavailable.
    """
    try:
        import duckdb
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        db_path = Path(project_root) / config['data']['duckdb_path']
        if not db_path.exists():
            return None

        conn = duckdb.connect(str(db_path), read_only=True)
        rows = conn.execute(
            "SELECT user_id, scenario, start_date, end_date, details FROM insiders"
        ).fetchall()
        conn.close()

        meta = {}
        for uid, scen, start, end, details in rows:
            meta[uid] = {
                'scenario': int(scen) if scen is not None else 0,
                'start_date': str(start) if start is not None else None,
                'end_date': str(end) if end is not None else None,
                'details': str(details) if details is not None else '',
            }
        return meta

    except ImportError:
        return None


def per_scenario_breakdown(y_true, y_pred, user_ids, user_scenario):
    """Per-scenario detection breakdown."""
    if user_scenario is None:
        return None

    insider_users = np.unique(user_ids[y_true == 1])

    scenario_stats = defaultdict(lambda: {
        'total': 0, 'detected': 0, 'users': 0, 'users_detected': 0
    })

    for u in insider_users:
        scenario = user_scenario.get(str(u), user_scenario.get(u, 0))
        mask = user_ids == u
        u_labels = y_true[mask]
        u_preds = y_pred[mask]
        insider_mask = u_labels == 1
        n_insider = insider_mask.sum()
        n_detected = (u_preds[insider_mask] == 1).sum()

        scenario_stats[scenario]['total'] += int(n_insider)
        scenario_stats[scenario]['detected'] += int(n_detected)
        scenario_stats[scenario]['users'] += 1
        if n_detected > 0:
            scenario_stats[scenario]['users_detected'] += 1

    results = {}
    for scen in sorted(scenario_stats.keys()):
        s = scenario_stats[scen]
        results[f'scenario_{scen}'] = {
            'insider_sequences': s['total'],
            'detected_sequences': s['detected'],
            'sequence_recall': float(s['detected'] / max(s['total'], 1)),
            'insider_users': s['users'],
            'users_detected': s['users_detected'],
            'user_recall': float(s['users_detected'] / max(s['users'], 1)),
        }
    return results


# =========================================================================
# SOC Report
# =========================================================================

def classify_severity(detection_rate, max_score, threshold):
    """Derive alert severity from detection rate and score magnitude."""
    if detection_rate > 0.5 or max_score > 2 * threshold:
        return "HIGH"
    elif detection_rate > 0.25:
        return "MEDIUM"
    return "LOW"


def identify_risk_indicators(model, x_cont, x_cat, config, device, top_k=3):
    """Identify which feature groups drive the anomaly for given sequences.

    Computes per-feature reconstruction error and maps the highest-error
    features back to human-readable names from config.
    
    Feature order matches feature_engineering.py:
    1. continuous features
    2. interaction features
    3. user_normalized features (with _zscore, _rolling_mean, _rolling_std suffixes)
    4. cyclical features
    5. binary features
    """
    x_c = torch.tensor(x_cont, dtype=torch.float32).to(device)
    x_cat_t = torch.tensor(x_cat, dtype=torch.long).to(device)

    with torch.no_grad():
        predictions, mask = model(x_c, x_cat_t, mask=None)

    diff = (predictions - x_c).pow(2)
    per_feat_error = diff.mean(dim=(0, 1)).cpu().numpy()

    # Build feature list in the same order as feature_engineering.py
    cont_features = config.get('features', {}).get('continuous', [])
    interaction_features = config.get('features', {}).get('interaction', [])
    user_norm = config.get('features', {}).get('user_normalized', [])
    cyclical = config.get('features', {}).get('cyclical', [])
    binary = config.get('features', {}).get('binary', [])
    
    # Build scalable_cols (continuous + interaction + user_normalized variants)
    scalable_cols = list(cont_features) + list(interaction_features)
    for col in user_norm:
        scalable_cols.extend([f'{col}_zscore', f'{col}_rolling_mean', f'{col}_rolling_std'])
    
    # Final order: scalable + cyclical + binary
    all_features = scalable_cols + list(cyclical) + list(binary)

    if len(all_features) != len(per_feat_error):
        # Fallback: use generic names if lengths don't match
        all_features = [f"feature_{i}" for i in range(len(per_feat_error))]

    top_idx = np.argsort(per_feat_error)[::-1][:top_k]

    feature_group_map = {
        # Continuous features
        'http_request_count': 'Web browsing anomaly',
        'unique_domains': 'Web browsing anomaly',
        'emails_sent': 'Email volume deviation',
        'total_attachments': 'Email attachment anomaly',
        'external_email_count': 'External email spike',
        'unique_recipients': 'Unusual email recipients',
        'file_copy_count': 'File copy anomaly',
        'exe_copy_count': 'Executable file copying',
        'doc_copy_count': 'Document copying anomaly',
        'archive_copy_count': 'Archive file copying',
        'usb_connect_count': 'USB device usage anomaly',
        'usb_disconnect_count': 'USB device usage anomaly',
        'total_actions': 'Overall activity spike',
        'time_since_last_event': 'Unusual session timing',
        'duration_minutes': 'Unusual session duration',
        'openness': 'Personality trait deviation',
        'conscientiousness': 'Personality trait deviation',
        'extraversion': 'Personality trait deviation',
        'agreeableness': 'Personality trait deviation',
        'neuroticism': 'Personality trait deviation',
        # Interaction features
        'file_usb_interaction': 'File copy with USB device',
        'external_email_attachments': 'External email with attachments',
        'http_after_hours': 'After-hours web browsing',
        'email_after_hours': 'After-hours email activity',
        # User-normalized features (zscore indicates deviation from user baseline)
        'http_request_count_zscore': 'Web browsing deviation from user baseline',
        'http_request_count_rolling_mean': 'Web browsing rolling average',
        'http_request_count_rolling_std': 'Web browsing rolling variability',
        'emails_sent_zscore': 'Email volume deviation from user baseline',
        'emails_sent_rolling_mean': 'Email volume rolling average',
        'emails_sent_rolling_std': 'Email volume rolling variability',
        'file_copy_count_zscore': 'File copy deviation from user baseline',
        'file_copy_count_rolling_mean': 'File copy rolling average',
        'file_copy_count_rolling_std': 'File copy rolling variability',
        'total_actions_zscore': 'Activity level deviation from user baseline',
        'total_actions_rolling_mean': 'Activity level rolling average',
        'total_actions_rolling_std': 'Activity level rolling variability',
        # Cyclical features
        'hour_sin': 'Time-of-day pattern',
        'hour_cos': 'Time-of-day pattern',
        'day_sin': 'Day-of-week pattern',
        'day_cos': 'Day-of-week pattern',
        # Binary features
        'is_after_hours': 'After-hours activity',
        'is_weekend': 'Weekend activity',
        'usb_state_flag': 'USB device connected',
        'role_changed': 'Role change detected',
        'dept_changed': 'Department change detected',
        'functional_unit_changed': 'Functional unit change detected',
        'terminated_flag': 'Termination flag',
    }

    indicators = []
    seen = set()
    for idx in top_idx:
        fname = all_features[idx]
        # Check if it's a user-normalized feature and map to base feature name
        base_name = fname
        if '_zscore' in fname:
            base_name = fname.replace('_zscore', '')
        elif '_rolling_mean' in fname:
            base_name = fname.replace('_rolling_mean', '')
        elif '_rolling_std' in fname:
            base_name = fname.replace('_rolling_std', '')
        
        # Try exact match first, then base name, then generic
        label = feature_group_map.get(fname) or feature_group_map.get(base_name)
        if not label:
            # For user-normalized features, provide more context
            if '_zscore' in fname:
                label = f"Anomalous {base_name} (deviation from user baseline)"
            elif '_rolling_mean' in fname or '_rolling_std' in fname:
                label = f"Anomalous {base_name} (rolling statistic)"
            else:
                label = f"Anomalous {fname}"
        
        if label not in seen:
            indicators.append(label)
            seen.add(label)

    return indicators


def generate_soc_report(user_results, scores, y_true, user_ids,
                        timestamps, insider_meta, threshold, threshold_method,
                        model, X_test_cont, X_test_cat, config, device,
                        output_dir):
    """Generate a structured SOC alert report (JSON).

    Args:
        y_true: Can be None for deployment mode (no ground truth). In that
            case, anomalous sequences are identified purely by threshold.
    """
    alerts = []
    missed_users = []
    severity_counts = defaultdict(int)

    for i, ur in enumerate(user_results):
        uid = ur['user']
        mask = user_ids == uid
        u_scores = scores[mask]

        if ur.get('detected', 0) == 0 and ur.get('flagged', 0) == 0:
            missed_users.append(uid)
            continue

        detection_rate = ur.get('detection_rate', ur.get('pct_anomalous', 0))
        severity = classify_severity(detection_rate, ur['max_score'], threshold)
        severity_counts[severity] += 1

        top_n = min(5, len(u_scores))
        top_idx = np.argsort(u_scores)[::-1][:top_n]
        top_windows = []
        for rank, idx in enumerate(top_idx):
            window = {
                'anomaly_score': round(float(u_scores[idx]), 6),
                'rank': rank + 1,
            }
            if timestamps is not None:
                u_ts = timestamps[mask]
                window['timestamp'] = str(u_ts[idx])
            top_windows.append(window)

        u_cont = X_test_cont[mask]
        u_cat = X_test_cat[mask]
        anomalous_mask = u_scores >= threshold
        if anomalous_mask.any():
            sample_cont = u_cont[anomalous_mask][:min(5, int(anomalous_mask.sum()))]
            sample_cat = u_cat[anomalous_mask][:min(5, int(anomalous_mask.sum()))]
            risk_indicators = identify_risk_indicators(
                model, sample_cont, sample_cat, config, device
            )
        else:
            risk_indicators = []

        alert = {
            'alert_id': f"INSIDER-{uid}-{i+1:03d}",
            'severity': severity,
            'user_id': uid,
            'detection_summary': {
                'total_sequences': ur.get('total_sequences', ur.get('num_sequences', 0)),
                'flagged_sequences': ur.get('detected', ur.get('flagged', 0)),
                'detection_rate': detection_rate,
                'max_anomaly_score': ur['max_score'],
                'mean_anomaly_score': ur['mean_score'],
            },
            'risk_indicators': risk_indicators,
            'top_anomalous_windows': top_windows,
        }

        if 'insider_sequences' in ur:
            alert['detection_summary']['insider_sequences'] = ur['insider_sequences']
        if 'latency_sequences' in ur:
            alert['detection_summary']['latency_sequences'] = ur['latency_sequences']
        if 'first_detection_timestamp' in ur:
            alert['detection_summary']['first_detection'] = ur.get('first_detection_timestamp')
            alert['detection_summary']['latency_hours'] = ur.get('latency_hours', -1)

        if insider_meta and uid in insider_meta:
            im = insider_meta[uid]
            alert['scenario'] = im.get('scenario', 0)
            alert['scenario_description'] = im.get('details', '')
            alert['ground_truth_window'] = {
                'start': im.get('start_date'),
                'end': im.get('end_date'),
            }
            if 'detection_lead_time_hours' in ur:
                alert['detection_summary']['lead_time_hours'] = ur.get('detection_lead_time_hours')

        alerts.append(alert)

    report = {
        'report_metadata': {
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'model': 'InsiderTransformerAE v1',
            'threshold_method': threshold_method,
            'threshold_value': round(float(threshold), 6),
            'total_users_evaluated': int(len(np.unique(user_ids))),
            'total_sequences_evaluated': int(len(scores)),
        },
        'alerts': alerts,
        'summary': {
            'total_alerts': len(alerts),
            'by_severity': dict(severity_counts),
            'users_missed': missed_users,
        },
    }

    if output_dir is not None:
        report_path = Path(output_dir) / 'soc_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

    return report
