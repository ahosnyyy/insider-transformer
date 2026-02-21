"""
Feature Engineering — Daily Aggregation
====================================
All heavy computation in DuckDB SQL. No pandas. Only numpy for final export.

Stages:
  1. Daily aggregation from events (GROUP BY user_id, date)
  2. Enrich with static features (users, psychometric, LDAP changes, labels)
  3. Derived features + log1p + z-scores + rolling stats + cyclical encoding
  4. StandardScaler in numpy + export arrays
"""

import gc
import json
import math
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import duckdb
import yaml

try:
    from ..utils.common import load_config
except (ImportError, ValueError):
    def load_config() -> dict:
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)


# =========================================================================
# Stage 1: Daily Aggregation
# =========================================================================

def _create_daily_aggregation(conn):
    """Aggregate events to daily features per user."""
    print("\n[Stage 1] Creating daily aggregation...")

    sql = """
    CREATE TABLE daily AS
    WITH daily_raw AS (
        SELECT
            user_id,
            CAST(datetime AS DATE) AS date,
            -- Activity counts
            COUNT(CASE WHEN activity='Http' THEN 1 END) AS http_request_count,
            COUNT(DISTINCT CASE WHEN activity='Http' AND url IS NOT NULL
                  THEN SPLIT_PART(SPLIT_PART(url,'/',3),':',1) END) AS unique_domains,
            COUNT(CASE WHEN activity='Email' THEN 1 END) AS emails_sent,
            COALESCE(SUM(CASE WHEN activity='Email' THEN CAST(attachments AS INT) ELSE 0 END), 0) AS total_attachments,
            COUNT(CASE WHEN activity='Email' AND (
                  COALESCE(to_addr,'') NOT LIKE '%@dtaa.com%' OR
                  COALESCE(cc,'') NOT LIKE '%@dtaa.com%' OR
                  COALESCE(bcc,'') NOT LIKE '%@dtaa.com%'
            ) THEN 1 END) AS external_email_count,
            COUNT(DISTINCT CASE WHEN activity='Email' AND to_addr IS NOT NULL THEN to_addr END) AS unique_recipients,
            COUNT(CASE WHEN activity='File' THEN 1 END) AS file_copy_count,
            COUNT(CASE WHEN activity='File' AND (
                  LOWER(filename) LIKE '%.exe' OR LOWER(filename) LIKE '%.msi' OR
                  LOWER(filename) LIKE '%.bat' OR LOWER(filename) LIKE '%.cmd'
            ) THEN 1 END) AS exe_copy_count,
            COUNT(CASE WHEN activity='File' AND (
                  LOWER(filename) LIKE '%.doc' OR LOWER(filename) LIKE '%.docx' OR
                  LOWER(filename) LIKE '%.pdf' OR LOWER(filename) LIKE '%.xls' OR
                  LOWER(filename) LIKE '%.xlsx' OR LOWER(filename) LIKE '%.ppt' OR
                  LOWER(filename) LIKE '%.pptx' OR LOWER(filename) LIKE '%.txt' OR
                  LOWER(filename) LIKE '%.csv'
            ) THEN 1 END) AS doc_copy_count,
            COUNT(CASE WHEN activity='File' AND (
                  LOWER(filename) LIKE '%.zip' OR LOWER(filename) LIKE '%.rar' OR
                  LOWER(filename) LIKE '%.7z' OR LOWER(filename) LIKE '%.tar' OR
                  LOWER(filename) LIKE '%.gz'
            ) THEN 1 END) AS archive_copy_count,
            COUNT(CASE WHEN activity='Connect' THEN 1 END) AS usb_connect_count,
            COUNT(CASE WHEN activity='Disconnect' THEN 1 END) AS usb_disconnect_count,
            COUNT(*) AS total_actions,
            -- Intra-day structure
            COUNT(CASE WHEN activity='Logon' THEN 1 END) AS num_sessions,
            COUNT(DISTINCT EXTRACT(HOUR FROM datetime)) AS active_hours,
            -- After-hours flag (8:00-18:00 = work hours)
            COUNT(*) FILTER (WHERE EXTRACT(HOUR FROM datetime) < 8 OR EXTRACT(HOUR FROM datetime) >= 18) AS after_hours_actions,
            -- PC usage
            COUNT(DISTINCT pc) AS num_distinct_pcs,
            -- Most common PC
            MODE(pc) AS pc
        FROM src.events
        GROUP BY user_id, CAST(datetime AS DATE)
        HAVING COUNT(*) >= 1  -- only days with activity
    ),
    computed AS (
        SELECT
            *,
            -- After-hours ratio
            CAST(after_hours_actions AS FLOAT) / NULLIF(CAST(total_actions AS FLOAT), 0) AS after_hours_ratio,
            -- Peak hour concentration (fraction of events in busiest hour)
            (SELECT MAX(cnt) FROM (
                SELECT COUNT(*) AS cnt
                FROM src.events e2
                WHERE e2.user_id = daily_raw.user_id
                AND CAST(e2.datetime AS DATE) = daily_raw.date
                GROUP BY EXTRACT(HOUR FROM e2.datetime)
            ) t) / NULLIF(CAST(total_actions AS FLOAT), 0) AS peak_hour_concentration,
            -- Day of week (0=Monday, 6=Sunday)
            EXTRACT(ISODOW FROM date) - 1 AS dow
        FROM daily_raw
    )
    SELECT
        user_id,
        date,
        dow,
        http_request_count,
        unique_domains,
        emails_sent,
        total_attachments,
        external_email_count,
        unique_recipients,
        file_copy_count,
        exe_copy_count,
        doc_copy_count,
        archive_copy_count,
        usb_connect_count,
        usb_disconnect_count,
        total_actions,
        num_sessions,
        active_hours,
        after_hours_ratio,
        after_hours_actions,
        peak_hour_concentration,
        num_distinct_pcs,
        pc
    FROM computed
    """

    conn.execute(sql)
    count = conn.execute("SELECT COUNT(*) FROM daily").fetchone()[0]
    print(f"  Daily records: {count:,}")


# =========================================================================
# Stage 2: Enrich with Static Features
# =========================================================================

def _enrich_daily(conn):
    """Join users, psychometric, LDAP changes, terminations, insider labels."""
    print("\n[Stage 2] Enriching with static features + labels...")

    # Median psychometric scores for NULL filling
    med = conn.execute("""
        SELECT
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY openness),
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY conscientiousness),
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY extraversion),
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY agreeableness),
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY neuroticism)
        FROM src.psychometric
    """).fetchone()

    conn.execute(f"""
    CREATE TABLE enriched AS
    SELECT
        d.user_id, d.date, d.dow, d.pc,
        d.http_request_count, d.unique_domains, d.emails_sent, d.total_attachments,
        d.external_email_count, d.unique_recipients, d.file_copy_count,
        d.exe_copy_count, d.doc_copy_count, d.archive_copy_count,
        d.usb_connect_count, d.usb_disconnect_count, d.total_actions,
        d.num_sessions, d.active_hours, d.after_hours_ratio, d.after_hours_actions,
        d.peak_hour_concentration, d.num_distinct_pcs,
        u.role, u.department, u.functional_unit,
        COALESCE(p.openness, {med[0]}) AS openness,
        COALESCE(p.conscientiousness, {med[1]}) AS conscientiousness,
        COALESCE(p.extraversion, {med[2]}) AS extraversion,
        COALESCE(p.agreeableness, {med[3]}) AS agreeableness,
        COALESCE(p.neuroticism, {med[4]}) AS neuroticism,
        COALESCE(uc.role_changed, 0) AS role_changed,
        COALESCE(uc.dept_changed, 0) AS dept_changed,
        COALESCE(uc.fu_changed, 0) AS functional_unit_changed,
        CASE WHEN tu.termination_month IS NOT NULL
              AND STRFTIME(d.date, '%Y-%m') >= tu.termination_month
             THEN 1 ELSE 0 END AS terminated_flag,
        CASE WHEN EXISTS (
            SELECT 1 FROM src.insiders i
            WHERE d.user_id = i.user_id
            AND d.date >= i.start_date AND d.date <= i.end_date
        ) THEN 1 ELSE 0 END AS is_insider
    FROM daily d
    LEFT JOIN src.users u ON d.user_id = u.user_id
    LEFT JOIN src.psychometric p ON d.user_id = p.user_id
    LEFT JOIN src.user_changes uc
        ON d.user_id = uc.user_id
        AND STRFTIME(d.date, '%Y-%m') = uc.ldap_month
    LEFT JOIN src.terminated_users tu ON d.user_id = tu.user_id
    """)

    conn.execute("DROP TABLE daily")

    stats = conn.execute("""
        SELECT COUNT(*), SUM(is_insider), COUNT(DISTINCT user_id)
        FROM enriched
    """).fetchone()
    print(f"  Daily records: {stats[0]:,}, Insider days: {stats[1]:,}, Users: {stats[2]}")


# =========================================================================
# Stage 3: Compute Features
# =========================================================================

def _compute_features(conn, config):
    """Compute all features in SQL: log1p, z-scores, rolling, cyclical."""
    print("\n[Stage 3] Computing features in SQL...")

    user_norm = config['features'].get('user_normalized', [])
    rolling_w = config['processing'].get('rolling_window', 20)

    # Build z-score + rolling SQL fragments
    zscore_frags = []
    for col in user_norm:
        zscore_frags.append(f"""
            (la.{col} - AVG(la.{col}) OVER (PARTITION BY la.user_id)) /
                GREATEST(STDDEV_POP(la.{col}) OVER (PARTITION BY la.user_id), 1e-8)
                AS {col}_zscore,
            AVG(la.{col}) OVER (PARTITION BY la.user_id ORDER BY la.date
                ROWS BETWEEN {rolling_w-1} PRECEDING AND CURRENT ROW)
                AS {col}_rolling_mean,
            COALESCE(STDDEV_POP(la.{col}) OVER (PARTITION BY la.user_id ORDER BY la.date
                ROWS BETWEEN {rolling_w-1} PRECEDING AND CURRENT ROW), 0)
                AS {col}_rolling_std
        """)
    zscore_sql = ",\n".join(zscore_frags) + "," if zscore_frags else ""

    conn.execute(f"""
    CREATE TABLE featured AS
    WITH raw_interact AS (
        -- Interaction features from RAW counts (before log1p), then log1p the product
        SELECT
            user_id, date,
            LN(1 + file_copy_count * CAST(usb_connect_count > usb_disconnect_count AS INTEGER)) AS file_usb_interaction,
            LN(1 + external_email_count * total_attachments) AS external_email_attachments,
            LN(1 + http_request_count * CAST(after_hours_actions > 0 AS INTEGER)) AS http_after_hours,
            LN(1 + emails_sent * CAST(after_hours_actions > 0 AS INTEGER)) AS email_after_hours
        FROM enriched
    ),
    la AS (
        SELECT
            e.user_id, e.date, e.dow, e.pc, e.role, e.department, e.functional_unit,
            e.is_insider,
            -- log1p on count features
            LN(1+e.http_request_count) AS http_request_count,
            LN(1+e.unique_domains) AS unique_domains,
            LN(1+e.emails_sent) AS emails_sent,
            LN(1+e.total_attachments) AS total_attachments,
            LN(1+e.external_email_count) AS external_email_count,
            LN(1+e.unique_recipients) AS unique_recipients,
            LN(1+e.file_copy_count) AS file_copy_count,
            LN(1+e.exe_copy_count) AS exe_copy_count,
            LN(1+e.doc_copy_count) AS doc_copy_count,
            LN(1+e.archive_copy_count) AS archive_copy_count,
            LN(1+e.usb_connect_count) AS usb_connect_count,
            LN(1+e.usb_disconnect_count) AS usb_disconnect_count,
            LN(1+e.total_actions) AS total_actions,
            LN(1+e.after_hours_actions) AS after_hours_actions,
            e.openness, e.conscientiousness, e.extraversion, e.agreeableness, e.neuroticism,
            e.role_changed, e.dept_changed, e.functional_unit_changed, e.terminated_flag,
            e.num_sessions, e.active_hours, e.after_hours_ratio, e.peak_hour_concentration,
            e.num_distinct_pcs,
            -- Interaction features (already log1p'd from raw counts)
            ri.file_usb_interaction,
            ri.external_email_attachments,
            ri.http_after_hours,
            ri.email_after_hours
        FROM enriched e
        JOIN raw_interact ri ON e.user_id = ri.user_id AND e.date = ri.date
    )
    SELECT
        la.user_id, la.date, la.dow, la.pc, la.role, la.department, la.functional_unit,
        la.is_insider,

        -- Continuous (scalable)
        la.http_request_count, la.unique_domains,
        la.emails_sent, la.total_attachments, la.external_email_count,
        la.unique_recipients, la.file_copy_count, la.exe_copy_count,
        la.doc_copy_count, la.archive_copy_count, la.usb_connect_count,
        la.usb_disconnect_count, la.total_actions, la.after_hours_actions,
        la.openness, la.conscientiousness, la.extraversion,
        la.agreeableness, la.neuroticism,
        la.num_sessions, la.active_hours, la.after_hours_ratio,
        la.peak_hour_concentration, la.num_distinct_pcs,

        -- Interaction features (computed from raw counts, already log1p'd)
        la.file_usb_interaction,
        la.external_email_attachments,
        la.http_after_hours,
        la.email_after_hours,

        -- Z-scores + rolling stats
        {zscore_sql}

        -- Cyclical
        SIN(2*PI()*la.dow/7.0) AS day_of_week_sin,
        COS(2*PI()*la.dow/7.0) AS day_of_week_cos,

        -- Binary
        CASE WHEN la.dow >= 5 THEN 1 ELSE 0 END AS is_weekend,
        la.role_changed, la.dept_changed, la.functional_unit_changed, la.terminated_flag,

        -- Categorical (label encoded via DENSE_RANK)
        DENSE_RANK() OVER (ORDER BY la.user_id) - 1 AS user_id_encoded,
        DENSE_RANK() OVER (ORDER BY la.pc) - 1 AS pc_encoded,
        DENSE_RANK() OVER (ORDER BY la.role) - 1 AS role_encoded,
        DENSE_RANK() OVER (ORDER BY la.department) - 1 AS department_encoded,
        DENSE_RANK() OVER (ORDER BY la.functional_unit) - 1 AS functional_unit_encoded

    FROM la
    """)

    conn.execute("DROP TABLE enriched")

    cnt = conn.execute("SELECT COUNT(*) FROM featured").fetchone()[0]
    print(f"  Featured table: {cnt:,} rows")


# =========================================================================
# Stage 4: StandardScaler + Export
# =========================================================================

def _scale_and_export(conn, config, output_dir):
    """Fetch from DuckDB, apply StandardScaler (train-only fit) in numpy, save arrays."""
    from .sequence_creation import split_users

    print("\n[Stage 4] Scaling and exporting...")

    # Define column groups
    cont_cfg = config['features']['continuous']
    interaction_cfg = config['features'].get('interaction', [])
    user_norm = config['features'].get('user_normalized', [])

    scalable_cols = list(cont_cfg) + list(interaction_cfg)
    for col in user_norm:
        scalable_cols.extend([f'{col}_zscore', f'{col}_rolling_mean', f'{col}_rolling_std'])

    cyclical_cols = config['features']['cyclical']
    binary_cols = config['features']['binary']
    cat_cols = [f'{c}_encoded' for c in config['features']['categorical']]
    all_cont_cols = scalable_cols + cyclical_cols + binary_cols

    # Verify columns exist
    existing = [r[0] for r in conn.execute("DESCRIBE featured").fetchall()]
    scalable_cols = [c for c in scalable_cols if c in existing]
    cyclical_cols = [c for c in cyclical_cols if c in existing]
    binary_cols = [c for c in binary_cols if c in existing]
    cat_cols = [c for c in cat_cols if c in existing]
    all_cont_cols = scalable_cols + cyclical_cols + binary_cols

    print(f"  Scalable: {len(scalable_cols)}, Cyclical: {len(cyclical_cols)}, "
          f"Binary: {len(binary_cols)}, Categorical: {len(cat_cols)}")

    # Fetch scalable columns as numpy
    scalable_sql = ", ".join(f'CAST("{c}" AS DOUBLE) AS "{c}"' for c in scalable_cols)
    scalable_data = conn.execute(f"""
        SELECT {scalable_sql} FROM featured ORDER BY user_id, date
    """).fetchnumpy()

    X_scalable = np.column_stack([scalable_data[c].astype(np.float32) for c in scalable_cols])
    del scalable_data
    gc.collect()

    # Determine user split so we fit the scaler on train users only
    meta_split = conn.execute("""
        SELECT user_id, is_insider FROM featured ORDER BY user_id, date
    """).fetchnumpy()
    user_ids_all = meta_split['user_id']
    labels_all = meta_split['is_insider'].astype(np.int64)
    del meta_split
    gc.collect()

    train_users, _, _, _ = split_users(user_ids_all, labels_all, config)
    train_mask = np.isin(user_ids_all, np.array(list(train_users)))
    n_train_days = train_mask.sum()

    means = X_scalable[train_mask].mean(axis=0)
    stds = X_scalable[train_mask].std(axis=0)
    stds[stds < 1e-8] = 1.0
    X_scalable = ((X_scalable - means) / stds).astype(np.float32)
    print(f"  StandardScaler fitted on {n_train_days:,} train days, "
          f"applied to {X_scalable.shape[0]:,} total")

    # Fetch cyclical + binary
    if cyclical_cols + binary_cols:
        other_sql = ", ".join(f'CAST("{c}" AS DOUBLE) AS "{c}"' for c in cyclical_cols + binary_cols)
        other_data = conn.execute(f"""
            SELECT {other_sql} FROM featured ORDER BY user_id, date
        """).fetchnumpy()
        X_other = np.column_stack([
            other_data[c].astype(np.float32) for c in cyclical_cols + binary_cols
        ])
        del other_data
        gc.collect()
    else:
        X_other = np.empty((X_scalable.shape[0], 0), dtype=np.float32)

    X_continuous = np.hstack([X_scalable, X_other])
    del X_scalable, X_other
    gc.collect()

    # Fetch categorical
    cat_sql = ", ".join(f'CAST("{c}" AS INTEGER) AS "{c}"' for c in cat_cols)
    cat_data = conn.execute(f"""
        SELECT {cat_sql} FROM featured ORDER BY user_id, date
    """).fetchnumpy()
    X_categorical = np.column_stack([cat_data[c].astype(np.int64) for c in cat_cols])
    del cat_data
    gc.collect()

    # Reuse user_ids/labels from split computation
    labels = labels_all
    user_ids = user_ids_all
    dates_data = conn.execute("""
        SELECT date FROM featured ORDER BY user_id, date
    """).fetchnumpy()
    dates = dates_data['date']
    del dates_data, labels_all, user_ids_all, train_mask
    gc.collect()

    # Save label encoder mappings
    label_mappings = {}
    for raw_col in config['features']['categorical']:
        enc_col = f'{raw_col}_encoded'
        if enc_col in existing:
            mapping = conn.execute(f"""
                SELECT DISTINCT {raw_col}, {enc_col}
                FROM featured ORDER BY {enc_col}
            """).fetchall()
            label_mappings[raw_col] = {str(row[0]): row[1] for row in mapping}

    # Save everything
    print(f"\n  Saving to {output_dir}/...")
    np.save(output_dir / 'features_continuous.npy', X_continuous)
    print(f"    features_continuous.npy: {X_continuous.shape}")
    np.save(output_dir / 'features_categorical.npy', X_categorical)
    print(f"    features_categorical.npy: {X_categorical.shape}")
    np.save(output_dir / 'labels.npy', labels)
    print(f"    labels.npy: {labels.shape} (insider={labels.sum():,})")
    np.save(output_dir / 'user_ids.npy', user_ids)
    np.save(output_dir / 'dates.npy', dates)

    # Save preprocessing artifacts
    artifacts = {
        'scaler_means': means,
        'scaler_stds': stds,
        'label_mappings': label_mappings,
        'continuous_columns': all_cont_cols,
        'scalable_columns': scalable_cols,
        'cyclical_columns': cyclical_cols,
        'binary_columns': binary_cols,
        'categorical_columns': cat_cols,
    }
    with open(output_dir / 'preprocessing_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"    preprocessing_artifacts.pkl saved")

    with open(output_dir / 'pipeline_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"    pipeline_config.json saved")


# =========================================================================
# Main Pipeline
# =========================================================================

def run_feature_engineering() -> Path:
    config = load_config()
    project_root = Path(__file__).parent.parent.parent

    db_path = project_root / config['data']['duckdb_path']
    output_dir = project_root / config['data']['processed_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    # In-memory DuckDB + attach source as read-only
    conn = duckdb.connect()
    conn.execute(f"ATTACH '{db_path}' AS src (READ_ONLY)")
    conn.execute("SET threads = 4")

    try:
        _create_daily_aggregation(conn)
        _enrich_daily(conn)
        _compute_features(conn, config)
        _scale_and_export(conn, config, output_dir)
        return output_dir
    finally:
        conn.close()


if __name__ == "__main__":
    run_feature_engineering()
