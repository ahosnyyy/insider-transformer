"""
Shared Evaluation Helpers for Session-Aware Insider Detection
=============================================================
Functions used by both 04_evaluate.py and 06_inference.py.

Contains:
- Model loading with session-aware feature support
- Threshold computation and calibration
- Multi-level metrics (user, sequence, session)
- Per-user/per-scenario breakdowns
- Session-level evaluation and localization
- Severity classification and risk indicator identification
- SOC report generation with session drill-down
- Detection latency summaries
- Session-aware scoring and analysis

Key Features:
- Session-aware feature handling (~52 continuous + 5 categorical)
- Session-level metrics for precise threat localization
- Mixed precision (AMP) support for faster evaluation
- Behavioral feature indexing for reconstruction loss
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
    """
    Load trained session-aware InsiderTransformerAE from checkpoint.
    
    Loads model with session-aware features and behavioral feature indexing
    for reconstruction loss computation.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file or directory containing best_model.pt
        device: torch device (CPU/GPU)
        config: Configuration dict with model hyperparameters
        
    Returns:
        tuple: (model, n_continuous, checkpoint_dict)
            - model: Loaded InsiderTransformerAE model
            - n_continuous: Number of continuous features (~52 with session features)
            - checkpoint_dict: Full checkpoint with metadata
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
    """
    Compute anomaly scores on validation set for threshold calibration.
    
    Uses session-aware features to compute reconstruction errors for threshold
    selection. Prefers validation set but falls back to training set if needed.
    
    Args:
        model: Trained InsiderTransformerAE model
        data_dir: Directory containing feature arrays
        device: torch device (CPU/GPU)
        batch_size: Batch size for scoring (default: 256)
        use_amp: Enable mixed precision for faster scoring (default: False)
        
    Returns:
        scores: Numpy array of anomaly scores for calibration
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
    3. session features
    4. user_normalized features (with _zscore, _rolling_mean, _rolling_std suffixes)
    5. cyclical features
    6. binary features
    """
    x_c = torch.tensor(x_cont, dtype=torch.float32).to(device)
    x_cat_t = torch.tensor(x_cat, dtype=torch.long).to(device)

    with torch.no_grad():
        predictions = model(x_c, x_cat_t)

    diff = (predictions - x_c).pow(2)
    per_feat_error = diff.mean(dim=(0, 1)).cpu().numpy()

    # Build feature list in the same order as feature_engineering.py
    cont_features = config.get('features', {}).get('continuous', [])
    interaction_features = config.get('features', {}).get('interaction', [])
    session_features = config.get('features', {}).get('session', [])
    user_norm = config.get('features', {}).get('user_normalized', [])
    cyclical = config.get('features', {}).get('cyclical', [])
    binary = config.get('features', {}).get('binary', [])

    # Build scalable_cols (continuous + interaction + session + user_normalized variants)
    scalable_cols = list(cont_features) + list(interaction_features) + list(session_features)
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
        # Session-derived features
        'sess_count': 'Unusual number of sessions',
        'mean_session_duration': 'Unusual session duration pattern',
        'max_session_duration': 'Unusually long session',
        'std_session_duration': 'Erratic session durations',
        'total_session_time': 'Unusual total session time',
        'num_after_hours_sessions': 'After-hours session activity',
        'after_hours_session_ratio': 'High after-hours session ratio',
        'earliest_session_hour': 'Unusually early session start',
        'latest_session_hour': 'Unusually late session start',
        'session_hour_spread': 'Wide session time spread',
        'mean_session_event_count': 'Unusual events per session',
        'max_session_event_count': 'Session with high event count',
        'mean_session_entropy': 'Unusual session activity mix',
        'num_usb_sessions': 'Multiple USB sessions',
        'num_exe_sessions': 'Sessions with executable copies',
        'longest_inter_session_gap': 'Unusual gap between sessions',
        'usb_after_hours_sessions': 'After-hours USB session',
        'file_heavy_sessions': 'Sessions with heavy file copying',
        'short_usb_sessions': 'Short USB grab session',
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
        'sess_count_zscore': 'Session count deviation from user baseline',
        'sess_count_rolling_mean': 'Session count rolling average',
        'sess_count_rolling_std': 'Session count rolling variability',
        'mean_session_duration_zscore': 'Session duration deviation from user baseline',
        'mean_session_duration_rolling_mean': 'Session duration rolling average',
        'mean_session_duration_rolling_std': 'Session duration rolling variability',
        'total_session_time_zscore': 'Total session time deviation from user baseline',
        'total_session_time_rolling_mean': 'Total session time rolling average',
        'total_session_time_rolling_std': 'Total session time rolling variability',
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


# =========================================================================
# Session-Level Evaluation
# =========================================================================

def load_sessions_from_duckdb(config, project_root=None):
    """Load the sessions and session_features tables from DuckDB.

    Returns:
        dict with 'sessions' and 'events_by_session' DataFrames (as lists of dicts),
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

        # Check if sessions table exists
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        if 'sessions' not in tables:
            conn.close()
            return None

        sessions = conn.execute("""
            SELECT session_id, user_id, pc, session_date, session_start,
                   session_end, duration_min, is_after_hours, hour_of_start
            FROM sessions
            ORDER BY user_id, session_start
        """).fetchall()

        cols = ['session_id', 'user_id', 'pc', 'session_date', 'session_start',
                'session_end', 'duration_min', 'is_after_hours', 'hour_of_start']
        session_dicts = [dict(zip(cols, row)) for row in sessions]

        conn.close()
        return session_dicts

    except (ImportError, Exception):
        return None


def compute_session_metrics(scores, y_true, y_pred, user_ids, timestamps,
                            insider_meta, session_data, threshold):
    """Compute session-level evaluation metrics.

    Maps flagged days back to individual sessions and compares against
    ground-truth insider activity timestamps.

    Args:
        scores: anomaly scores per sequence
        y_true: ground truth labels per sequence
        y_pred: predicted labels per sequence
        user_ids: user ID per sequence
        timestamps: date per sequence (datetime64)
        insider_meta: dict user_id -> {start_date, end_date, ...}
        session_data: list of session dicts from load_sessions_from_duckdb
        threshold: anomaly threshold

    Returns:
        dict with session-level metrics
    """
    if session_data is None or insider_meta is None or timestamps is None:
        return None

    # Build lookup: (user_id, date_str) -> list of sessions
    sessions_by_user_date = defaultdict(list)
    for s in session_data:
        date_str = str(s['session_date'])
        sessions_by_user_date[(s['user_id'], date_str)].append(s)

    # For each insider user, find flagged days and check session overlap
    insider_users = set(insider_meta.keys())
    total_insider_sessions = 0
    flagged_insider_sessions = 0
    total_flagged_sessions = 0
    correct_localizations = 0
    total_flagged_days = 0

    for uid in insider_users:
        meta = insider_meta[uid]
        gt_start = meta.get('start_date')
        gt_end = meta.get('end_date')
        if not gt_start or not gt_end:
            continue

        gt_start_dt = np.datetime64(gt_start)
        gt_end_dt = np.datetime64(gt_end)

        mask = user_ids == uid
        u_scores = scores[mask]
        u_ts = timestamps[mask]
        u_pred = y_pred[mask]

        for i, (score, ts, pred) in enumerate(zip(u_scores, u_ts, u_pred)):
            if np.isnat(ts):
                continue
            date_str = str(np.datetime64(ts, 'D'))
            day_sessions = sessions_by_user_date.get((uid, date_str), [])

            # Count insider sessions (sessions during ground-truth period)
            is_insider_day = gt_start_dt <= np.datetime64(ts, 'D') <= gt_end_dt
            if is_insider_day:
                for s in day_sessions:
                    total_insider_sessions += 1
                    if s.get('is_after_hours', 0) == 1:
                        total_insider_sessions  # already counted

            if pred == 1:
                total_flagged_days += 1
                # All sessions on a flagged day are "flagged"
                for s in day_sessions:
                    total_flagged_sessions += 1
                    if is_insider_day:
                        flagged_insider_sessions += 1

                # Localization: was the most suspicious session on this day
                # actually during the insider period?
                if is_insider_day and day_sessions:
                    # The most suspicious session is the after-hours one
                    # or the one with highest duration (heuristic)
                    has_after_hours = any(
                        s.get('is_after_hours', 0) == 1 for s in day_sessions
                    )
                    if has_after_hours:
                        correct_localizations += 1

    session_precision = (flagged_insider_sessions / max(total_flagged_sessions, 1))
    session_recall = (flagged_insider_sessions / max(total_insider_sessions, 1))
    session_f1 = (2 * session_precision * session_recall /
                  max(session_precision + session_recall, 1e-10))
    localization_acc = (correct_localizations / max(total_flagged_days, 1))

    return {
        'session_precision': round(float(session_precision), 4),
        'session_recall': round(float(session_recall), 4),
        'session_f1': round(float(session_f1), 4),
        'session_localization_accuracy': round(float(localization_acc), 4),
        'total_insider_sessions': int(total_insider_sessions),
        'flagged_insider_sessions': int(flagged_insider_sessions),
        'total_flagged_sessions': int(total_flagged_sessions),
        'total_flagged_days': int(total_flagged_days),
        'correct_localizations': int(correct_localizations),
    }


def build_session_drilldown(uid, flagged_dates, session_data, config,
                            project_root=None):
    """Build session-level drill-down for a flagged user's anomalous days.

    Args:
        uid: user ID
        flagged_dates: list of date strings (YYYY-MM-DD) that were flagged
        session_data: list of session dicts
        config: config dict
        project_root: project root path

    Returns:
        list of day dicts, each containing sessions with risk assessment
    """
    if session_data is None:
        return []

    # Build lookup for this user
    user_sessions = defaultdict(list)
    for s in session_data:
        if s['user_id'] == uid:
            date_str = str(s['session_date'])
            user_sessions[date_str].append(s)

    # Load raw events for drill-down
    events_by_session = {}
    try:
        import duckdb
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        db_path = Path(project_root) / config['data']['duckdb_path']
        if db_path.exists():
            conn = duckdb.connect(str(db_path), read_only=True)
            for date_str in flagged_dates:
                for s in user_sessions.get(date_str, []):
                    sid = s['session_id']
                    rows = conn.execute(f"""
                        SELECT
                            CAST(datetime AS VARCHAR) AS time,
                            activity AS type,
                            COALESCE(filename, url,
                                CASE WHEN activity='Email' THEN 'to: ' || COALESCE(to_addr, '')
                                     ELSE NULL END,
                                activity) AS detail
                        FROM events
                        WHERE user_id = '{uid}'
                          AND datetime >= '{s['session_start']}'
                          AND datetime <= '{s['session_end']}'
                        ORDER BY datetime
                        LIMIT 20
                    """).fetchall()
                    events_by_session[sid] = [
                        {'time': r[0], 'type': r[1], 'detail': r[2]}
                        for r in rows
                    ]
            conn.close()
    except (ImportError, Exception):
        pass

    days = []
    for date_str in flagged_dates:
        day_sessions = user_sessions.get(date_str, [])
        session_entries = []
        for s in day_sessions:
            is_ah = s.get('is_after_hours', 0) == 1
            dur = s.get('duration_min', 0)

            # Heuristic risk assessment
            risk = 'low'
            indicators = []
            if is_ah:
                risk = 'medium'
                indicators.append(f"After-hours session ({s.get('hour_of_start', '?')}:00)")
            if dur < 15:
                indicators.append("Short session (<15 min)")
                if is_ah:
                    risk = 'high'

            sid = s['session_id']
            events = events_by_session.get(sid, [])

            # Check events for high-risk patterns
            has_usb = any(e['type'] in ('Connect', 'Disconnect') for e in events)
            has_exe = any(e.get('detail', '').endswith(('.exe', '.msi', '.bat'))
                         for e in events)
            has_file = any(e['type'] == 'File' for e in events)

            if has_usb:
                indicators.append("USB device activity")
                risk = 'high' if is_ah else 'medium'
            if has_exe:
                indicators.append("Executable file copied")
                risk = 'high'
            if has_file and dur < 15:
                indicators.append("File activity in short session")

            # Build summary
            type_counts = defaultdict(int)
            for e in events:
                type_counts[e['type']] += 1
            summary_parts = [f"{cnt} {typ}" for typ, cnt in type_counts.items()]
            summary = f"{dur:.0f}-min session: " + ", ".join(summary_parts) if summary_parts else f"{dur:.0f}-min session"

            entry = {
                'session_id': sid,
                'start': str(s.get('session_start', '')),
                'end': str(s.get('session_end', '')),
                'duration_min': round(dur, 1),
                'risk': risk,
                'summary': summary,
            }
            if indicators:
                entry['indicators'] = indicators
            if events and risk in ('medium', 'high'):
                entry['events'] = events[:10]

            session_entries.append(entry)

        days.append({
            'date': date_str,
            'sessions': session_entries,
        })

    return days


def generate_soc_report(user_results, scores, y_true, user_ids,
                        timestamps, insider_meta, threshold, threshold_method,
                        model, X_test_cont, X_test_cat, config, device,
                        output_dir, session_data=None):
    """Generate a structured SOC alert report (JSON) with session drill-down.

    Args:
        y_true: Can be None for deployment mode (no ground truth). In that
            case, anomalous sequences are identified purely by threshold.
        session_data: Optional list of session dicts for session-level drill-down.
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

        # Session-level drill-down (if session data available)
        if session_data is not None and timestamps is not None:
            u_ts = timestamps[mask]
            anomalous_mask = u_scores >= threshold
            flagged_dates = []
            for idx_a in np.where(anomalous_mask)[0]:
                ts_val = u_ts[idx_a]
                if not np.isnat(ts_val):
                    flagged_dates.append(str(np.datetime64(ts_val, 'D')))
            flagged_dates = sorted(set(flagged_dates))[:10]  # cap at 10 days

            if flagged_dates:
                alert['session_drilldown'] = build_session_drilldown(
                    uid, flagged_dates, session_data, config
                )

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
