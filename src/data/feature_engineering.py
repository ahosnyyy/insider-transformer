"""
Feature Engineering — Session-Aware Daily Aggregation
====================================================
All heavy computation in DuckDB SQL. No pandas. Only numpy for final export.

Stages:
  1a. Daily aggregation from events (GROUP BY user_id, date)
      - Activity counts: HTTP, email, file, device, logon events
      - Intra-day structure: sessions, active hours, after-hours ratio
      - PC usage: distinct PCs, most common PC

  1b. Session detection and per-session features
      - Pair Logon/Logoff events per user per PC per day
      - Handle edge cases: missing logoff, multiple logons, overnight sessions
      - Compute per-session: duration, event counts, entropy, timing

  1c. Daily aggregation of session features
      - Session count, duration stats, timing stats
      - Behavior stats: event counts, entropy
      - Interaction features: USB+after-hours, email+attachments

  1d. Merge session statistics into daily table
      - LEFT JOIN session features onto daily aggregation
      - Preserves all daily records, adds session context

  2.  Enrich with static features
      - User profiles: role, department, functional_unit
      - Psychometric scores: Big Five personality traits
      - LDAP changes: role/department changes per month
      - Terminations: employee termination dates
      - Insider labels: ground truth threat scenarios

  3.  Derived features and transformations
      - log1p transform on all count features (handles skewness)
      - Interaction features: cross-feature products
      - Cyclical encoding: day of week sin/cos
      - Binary flags: weekend, role changes, termination
      - Per-user z-scores and 20-day rolling stats

  4.  Scaling and export
      - StandardScaler fitted on training users only
      - Export continuous and categorical arrays separately
      - Save preprocessing artifacts (scaler, label mappings)
      - Output: ~52 continuous + 5 categorical features per day
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
    """
    Create daily aggregation table with activity counts and intra-day structure.
    
    Computes 18 daily features:
    - Activity counts: HTTP requests, emails, file copies, USB connections
    - Intra-day metrics: sessions, active hours, after-hours ratio
    - PC usage: distinct PCs, most common PC
    
    Args:
        conn: DuckDB connection with events table
    """
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
# Stage 1b: Session Detection
# =========================================================================

def _create_sessions_table(conn):
    """
    Detect sessions by pairing Logon/Logoff events per user per PC.
    
    Session Definition:
    - Starts with Logon event
    - Ends with Logoff event OR next Logon OR end of day
    - Maximum duration: 12 hours (capped for outliers)
    
    Edge Cases Handled:
    - Missing Logoff: End session at next Logon or day end
    - Multiple Logons: Each Logon starts new session (implicit logoff)
    - Orphan Logoff: Ignored (no matching Logon)
    - Overnight sessions: Assigned to start date
    - Very long sessions: Capped at 12 hours for feature computation
    
    Args:
        conn: DuckDB connection with events table
    """
    print("\n[Stage 1b] Detecting sessions from Logon/Logoff events...")

    conn.execute("""
    CREATE TABLE sessions AS
    WITH logon_events AS (
        SELECT
            user_id, pc, datetime AS session_start,
            CAST(datetime AS DATE) AS session_date,
            EXTRACT(HOUR FROM datetime) AS hour_of_start,
            ROW_NUMBER() OVER (PARTITION BY user_id, pc ORDER BY datetime) AS rn
        FROM src.events
        WHERE activity = 'Logon'
    ),
    logoff_events AS (
        SELECT
            user_id, pc, datetime AS logoff_time,
            ROW_NUMBER() OVER (PARTITION BY user_id, pc ORDER BY datetime) AS rn
        FROM src.events
        WHERE activity = 'Logoff'
    ),
    paired AS (
        SELECT
            lo.user_id,
            lo.pc,
            lo.session_date,
            lo.session_start,
            lo.hour_of_start,
            -- Match to the first Logoff on the same user+pc AFTER this Logon
            (SELECT MIN(lf.logoff_time)
             FROM logoff_events lf
             WHERE lf.user_id = lo.user_id
               AND lf.pc = lo.pc
               AND lf.logoff_time > lo.session_start
               AND lf.logoff_time <= lo.session_start + INTERVAL '12 hours'
            ) AS session_end_raw
        FROM logon_events lo
    ),
    with_end AS (
        SELECT
            user_id, pc, session_date, session_start, hour_of_start,
            COALESCE(
                session_end_raw,
                -- Fallback: end of day if no logoff found
                (session_date + INTERVAL '1 day' - INTERVAL '1 second')::TIMESTAMP
            ) AS session_end
        FROM paired
    )
    SELECT
        ROW_NUMBER() OVER (ORDER BY user_id, session_start) AS session_id,
        user_id,
        pc,
        session_date,
        session_start,
        session_end,
        LEAST(
            EXTRACT(EPOCH FROM (session_end - session_start)) / 60.0,
            720.0
        ) AS duration_min,
        CASE WHEN hour_of_start < 8 OR hour_of_start >= 18 THEN 1 ELSE 0 END AS is_after_hours,
        hour_of_start
    FROM with_end
    """)

    count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    n_users = conn.execute("SELECT COUNT(DISTINCT user_id) FROM sessions").fetchone()[0]
    ah = conn.execute("SELECT SUM(is_after_hours) FROM sessions").fetchone()[0]
    print(f"  Sessions detected: {count:,} across {n_users} users "
          f"({ah:,} after-hours)")


# =========================================================================
# Stage 1c: Per-Session Features + Daily Session Aggregation
# =========================================================================

def _compute_session_features(conn):
    """Compute per-session activity features, then aggregate to daily level."""
    print("\n[Stage 1c] Computing per-session features...")

    # Step 1: Per-session activity counts
    conn.execute("""
    CREATE TABLE session_counts AS
    SELECT
        s.session_id,
        s.user_id,
        s.session_date,
        s.duration_min,
        s.is_after_hours,
        s.hour_of_start,
        COUNT(e.id) AS session_event_count,
        COUNT(CASE WHEN e.activity = 'File' THEN 1 END) AS session_file_count,
        COUNT(CASE WHEN e.activity = 'Email' THEN 1 END) AS session_email_count,
        COUNT(CASE WHEN e.activity = 'Http' THEN 1 END) AS session_http_count,
        CASE WHEN COUNT(CASE WHEN e.activity = 'Connect' THEN 1 END) > 0
             THEN 1 ELSE 0 END AS session_usb_used,
        CASE WHEN COUNT(CASE WHEN e.activity = 'Email' AND (
            COALESCE(e.to_addr, '') NOT LIKE '%@dtaa.com%'
        ) THEN 1 END) > 0 THEN 1 ELSE 0 END AS session_external_email,
        CASE WHEN COUNT(CASE WHEN e.activity = 'File' AND (
            LOWER(e.filename) LIKE '%.exe' OR LOWER(e.filename) LIKE '%.msi' OR
            LOWER(e.filename) LIKE '%.bat' OR LOWER(e.filename) LIKE '%.cmd'
        ) THEN 1 END) > 0 THEN 1 ELSE 0 END AS session_exe_copied,
        COUNT(DISTINCT CASE WHEN e.activity = 'Http' AND e.url IS NOT NULL
              THEN SPLIT_PART(SPLIT_PART(e.url, '/', 3), ':', 1) END) AS session_unique_domains,
        COALESCE(SUM(CASE WHEN e.activity = 'Email'
                     THEN CAST(e.attachments AS INT) ELSE 0 END), 0) AS session_attachment_count
    FROM sessions s
    LEFT JOIN src.events e
        ON e.user_id = s.user_id
        AND e.pc = s.pc
        AND e.datetime >= s.session_start
        AND e.datetime <= s.session_end
        AND e.activity NOT IN ('Logon', 'Logoff')
    GROUP BY s.session_id, s.user_id, s.session_date,
             s.duration_min, s.is_after_hours, s.hour_of_start
    """)

    # Step 2: Compute Shannon entropy per session from activity type distribution
    conn.execute("""
    CREATE TABLE session_entropy AS
    WITH act_counts AS (
        SELECT
            s.session_id,
            e.activity,
            COUNT(*) AS cnt,
            SUM(COUNT(*)) OVER (PARTITION BY s.session_id) AS total
        FROM sessions s
        JOIN src.events e
            ON e.user_id = s.user_id
            AND e.pc = s.pc
            AND e.datetime >= s.session_start
            AND e.datetime <= s.session_end
            AND e.activity NOT IN ('Logon', 'Logoff')
        GROUP BY s.session_id, e.activity
    ),
    probs AS (
        SELECT
            session_id,
            CAST(cnt AS DOUBLE) / CAST(total AS DOUBLE) AS p
        FROM act_counts
        WHERE total > 0
    )
    SELECT
        session_id,
        -1.0 * SUM(p * LN(p) / LN(2)) AS session_event_entropy
    FROM probs
    GROUP BY session_id
    """)

    # Step 3: Join counts + entropy
    conn.execute("""
    CREATE TABLE session_features AS
    SELECT
        sc.*,
        COALESCE(se.session_event_entropy, 0) AS session_event_entropy
    FROM session_counts sc
    LEFT JOIN session_entropy se ON sc.session_id = se.session_id
    """)

    conn.execute("DROP TABLE session_counts")
    conn.execute("DROP TABLE session_entropy")

    cnt = conn.execute("SELECT COUNT(*) FROM session_features").fetchone()[0]
    print(f"  Session features computed: {cnt:,} sessions")

    # Aggregate session features to daily level
    print("  Aggregating session features to daily level...")
    conn.execute("""
    CREATE TABLE daily_session_stats AS
    SELECT
        user_id,
        session_date AS date,
        -- Session count & duration stats
        COUNT(*) AS sess_count,
        AVG(duration_min) AS mean_session_duration,
        MAX(duration_min) AS max_session_duration,
        COALESCE(STDDEV_POP(duration_min), 0) AS std_session_duration,
        SUM(duration_min) AS total_session_time,
        -- Session timing stats
        SUM(is_after_hours) AS num_after_hours_sessions,
        CAST(SUM(is_after_hours) AS DOUBLE) / NULLIF(COUNT(*), 0) AS after_hours_session_ratio,
        MIN(hour_of_start) AS earliest_session_hour,
        MAX(hour_of_start) AS latest_session_hour,
        MAX(hour_of_start) - MIN(hour_of_start) AS session_hour_spread,
        -- Session behavior stats
        AVG(session_event_count) AS mean_session_event_count,
        MAX(session_event_count) AS max_session_event_count,
        AVG(COALESCE(session_event_entropy, 0)) AS mean_session_entropy,
        SUM(session_usb_used) AS num_usb_sessions,
        SUM(session_exe_copied) AS num_exe_sessions,
        -- Session interaction features
        SUM(CASE WHEN session_usb_used = 1 AND is_after_hours = 1
                 THEN 1 ELSE 0 END) AS usb_after_hours_sessions,
        SUM(CASE WHEN session_file_count > 10
                 THEN 1 ELSE 0 END) AS file_heavy_sessions,
        SUM(CASE WHEN duration_min < 15 AND session_usb_used = 1
                 THEN 1 ELSE 0 END) AS short_usb_sessions
    FROM session_features
    GROUP BY user_id, session_date
    """)

    cnt = conn.execute("SELECT COUNT(*) FROM daily_session_stats").fetchone()[0]
    print(f"  Daily session stats: {cnt:,} user-days")

    # Compute inter-session gaps per user per day
    conn.execute("""
    CREATE TABLE session_gaps AS
    SELECT
        user_id,
        session_date AS date,
        MAX(gap_min) AS longest_inter_session_gap
    FROM (
        SELECT
            user_id,
            session_date,
            EXTRACT(EPOCH FROM (
                session_start - LAG(session_end) OVER (
                    PARTITION BY user_id, session_date ORDER BY session_start
                )
            )) / 60.0 AS gap_min
        FROM sessions
    ) gaps
    WHERE gap_min IS NOT NULL AND gap_min >= 0
    GROUP BY user_id, session_date
    """)

    # Merge gaps into daily session stats
    conn.execute("""
    CREATE TABLE daily_session_merged AS
    SELECT
        ds.*,
        COALESCE(sg.longest_inter_session_gap, 0) AS longest_inter_session_gap
    FROM daily_session_stats ds
    LEFT JOIN session_gaps sg ON ds.user_id = sg.user_id AND ds.date = sg.date
    """)

    conn.execute("DROP TABLE daily_session_stats")
    conn.execute("DROP TABLE session_gaps")
    conn.execute("ALTER TABLE daily_session_merged RENAME TO daily_session_stats")

    print("  Session feature aggregation complete.")


# =========================================================================
# Stage 1d: Merge Session Stats into Daily Table
# =========================================================================

def _merge_session_stats(conn):
    """LEFT JOIN daily_session_stats onto the daily table."""
    print("\n[Stage 1d] Merging session stats into daily table...")

    conn.execute("""
    CREATE TABLE daily_merged AS
    SELECT
        d.*,
        COALESCE(ss.sess_count, 0) AS sess_count,
        COALESCE(ss.mean_session_duration, 0) AS mean_session_duration,
        COALESCE(ss.max_session_duration, 0) AS max_session_duration,
        COALESCE(ss.std_session_duration, 0) AS std_session_duration,
        COALESCE(ss.total_session_time, 0) AS total_session_time,
        COALESCE(ss.num_after_hours_sessions, 0) AS num_after_hours_sessions,
        COALESCE(ss.after_hours_session_ratio, 0) AS after_hours_session_ratio,
        COALESCE(ss.earliest_session_hour, 0) AS earliest_session_hour,
        COALESCE(ss.latest_session_hour, 0) AS latest_session_hour,
        COALESCE(ss.session_hour_spread, 0) AS session_hour_spread,
        COALESCE(ss.mean_session_event_count, 0) AS mean_session_event_count,
        COALESCE(ss.max_session_event_count, 0) AS max_session_event_count,
        COALESCE(ss.mean_session_entropy, 0) AS mean_session_entropy,
        COALESCE(ss.num_usb_sessions, 0) AS num_usb_sessions,
        COALESCE(ss.num_exe_sessions, 0) AS num_exe_sessions,
        COALESCE(ss.longest_inter_session_gap, 0) AS longest_inter_session_gap,
        COALESCE(ss.usb_after_hours_sessions, 0) AS usb_after_hours_sessions,
        COALESCE(ss.file_heavy_sessions, 0) AS file_heavy_sessions,
        COALESCE(ss.short_usb_sessions, 0) AS short_usb_sessions
    FROM daily d
    LEFT JOIN daily_session_stats ss ON d.user_id = ss.user_id AND d.date = ss.date
    """)

    conn.execute("DROP TABLE daily")
    conn.execute("DROP TABLE daily_session_stats")
    conn.execute("DROP TABLE session_features")
    conn.execute("ALTER TABLE daily_merged RENAME TO daily")

    cnt = conn.execute("SELECT COUNT(*) FROM daily").fetchone()[0]
    n_cols = len(conn.execute("DESCRIBE daily").fetchall())
    print(f"  Merged daily table: {cnt:,} rows, {n_cols} columns")


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
        -- Session-derived features
        d.sess_count, d.mean_session_duration, d.max_session_duration,
        d.std_session_duration, d.total_session_time,
        d.num_after_hours_sessions, d.after_hours_session_ratio,
        d.earliest_session_hour, d.latest_session_hour, d.session_hour_spread,
        d.mean_session_event_count, d.max_session_event_count,
        d.mean_session_entropy, d.num_usb_sessions, d.num_exe_sessions,
        d.longest_inter_session_gap, d.usb_after_hours_sessions,
        d.file_heavy_sessions, d.short_usb_sessions,
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
            ri.email_after_hours,
            -- Session-derived features (log1p on counts, pass-through on ratios/stats)
            LN(1+e.sess_count) AS sess_count,
            LN(1+e.mean_session_duration) AS mean_session_duration,
            LN(1+e.max_session_duration) AS max_session_duration,
            e.std_session_duration,
            LN(1+e.total_session_time) AS total_session_time,
            LN(1+e.num_after_hours_sessions) AS num_after_hours_sessions,
            e.after_hours_session_ratio,
            e.earliest_session_hour,
            e.latest_session_hour,
            e.session_hour_spread,
            LN(1+e.mean_session_event_count) AS mean_session_event_count,
            LN(1+e.max_session_event_count) AS max_session_event_count,
            e.mean_session_entropy,
            LN(1+e.num_usb_sessions) AS num_usb_sessions,
            LN(1+e.num_exe_sessions) AS num_exe_sessions,
            LN(1+e.longest_inter_session_gap) AS longest_inter_session_gap,
            LN(1+e.usb_after_hours_sessions) AS usb_after_hours_sessions,
            LN(1+e.file_heavy_sessions) AS file_heavy_sessions,
            LN(1+e.short_usb_sessions) AS short_usb_sessions
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

        -- Session-derived features
        la.sess_count,
        la.mean_session_duration, la.max_session_duration,
        la.std_session_duration, la.total_session_time,
        la.num_after_hours_sessions, la.after_hours_session_ratio,
        la.earliest_session_hour, la.latest_session_hour, la.session_hour_spread,
        la.mean_session_event_count, la.max_session_event_count,
        la.mean_session_entropy,
        la.num_usb_sessions, la.num_exe_sessions,
        la.longest_inter_session_gap,
        la.usb_after_hours_sessions, la.file_heavy_sessions, la.short_usb_sessions,

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

    session_cfg = config['features'].get('session', [])
    scalable_cols = list(cont_cfg) + list(interaction_cfg) + list(session_cfg)
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
# Inference: Transform with Saved Artifacts (no re-fit)
# =========================================================================

def transform_with_saved_artifacts(conn, artifacts, user_filter=None):
    """
    Apply saved preprocessing artifacts to session-aware features for inference.
    
    Reuses the exact scaler parameters and categorical mappings from training
    to ensure consistent feature representation. Handles unseen categorical values
    by assigning them to max_id + 1. Does NOT fit new scalers - preserves
    training preprocessing exactly.
    
    Args:
        conn: DuckDB connection with session-aware 'featured' table already created
        artifacts: Preprocessing artifacts dict from training (scaler, mappings, metadata)
        user_filter: Optional list/set of user IDs to restrict analysis to
        
    Returns:
        dict: Processed feature arrays ready for sequence creation
            - X_continuous: (n_days, n_continuous) scaled session-aware features
            - X_categorical: (n_days, n_categorical) encoded categorical features
            - user_ids: (n_days,) user ID per day
            - dates: (n_days,) datetime64 per day
            
    Note:
        This function ensures inference uses identical preprocessing as training,
        critical for consistent model predictions on session-aware features.
    """
    scalable_cols = artifacts['scalable_columns']
    cyclical_cols = artifacts['cyclical_columns']
    binary_cols = artifacts['binary_columns']
    cat_cols = artifacts['categorical_columns']
    means = artifacts['scaler_means']
    stds = artifacts['scaler_stds']
    label_mappings = artifacts['label_mappings']

    # Verify columns exist in featured table
    existing = [r[0] for r in conn.execute("DESCRIBE featured").fetchall()]
    scalable_cols = [c for c in scalable_cols if c in existing]
    cyclical_cols = [c for c in cyclical_cols if c in existing]
    binary_cols = [c for c in binary_cols if c in existing]

    # Optional user filter
    where_clause = ""
    if user_filter is not None:
        user_list = ", ".join(f"'{u}'" for u in user_filter)
        where_clause = f"WHERE user_id IN ({user_list})"

    order_clause = f"{where_clause} ORDER BY user_id, date" if where_clause else "ORDER BY user_id, date"

    # Fetch scalable columns and apply saved scaler
    scalable_sql = ", ".join(f'CAST("{c}" AS DOUBLE) AS "{c}"' for c in scalable_cols)
    scalable_data = conn.execute(f"""
        SELECT {scalable_sql} FROM featured {order_clause}
    """).fetchnumpy()

    X_scalable = np.column_stack([scalable_data[c].astype(np.float32) for c in scalable_cols])
    del scalable_data

    # Apply saved scaler (transform only, no fit)
    X_scalable = ((X_scalable - means[:len(scalable_cols)]) / stds[:len(scalable_cols)]).astype(np.float32)

    # Fetch cyclical + binary
    if cyclical_cols + binary_cols:
        other_sql = ", ".join(f'CAST("{c}" AS DOUBLE) AS "{c}"' for c in cyclical_cols + binary_cols)
        other_data = conn.execute(f"""
            SELECT {other_sql} FROM featured {order_clause}
        """).fetchnumpy()
        X_other = np.column_stack([
            other_data[c].astype(np.float32) for c in cyclical_cols + binary_cols
        ])
        del other_data
    else:
        X_other = np.empty((X_scalable.shape[0], 0), dtype=np.float32)

    X_continuous = np.hstack([X_scalable, X_other])
    del X_scalable, X_other

    # Fetch categorical — use saved label mappings, handle unseen values
    cat_raw_names = [c.replace('_encoded', '') for c in cat_cols]
    cat_sql = ", ".join(f'CAST("{c}" AS VARCHAR) AS "{c}"' for c in cat_raw_names)
    cat_data = conn.execute(f"""
        SELECT {cat_sql} FROM featured {order_clause}
    """).fetchnumpy()

    cat_arrays = []
    for raw_name in cat_raw_names:
        mapping = label_mappings.get(raw_name, {})
        max_id = max(mapping.values()) + 1 if mapping else 0
        raw_vals = cat_data[raw_name]
        encoded = np.array([mapping.get(str(v), max_id) for v in raw_vals], dtype=np.int64)
        cat_arrays.append(encoded)
    X_categorical = np.column_stack(cat_arrays) if cat_arrays else np.empty((X_continuous.shape[0], 0), dtype=np.int64)
    del cat_data

    # Fetch metadata
    meta = conn.execute(f"""
        SELECT user_id, date FROM featured {order_clause}
    """).fetchnumpy()

    n_days = X_continuous.shape[0]
    n_cont = X_continuous.shape[1]
    expected_cont = len(artifacts['continuous_columns'])
    if n_cont != expected_cont:
        print(f"  [warn] Feature count mismatch: got {n_cont}, expected {expected_cont}")

    print(f"  Transformed {n_days:,} days, {n_cont} continuous + {X_categorical.shape[1]} categorical features")

    return {
        'X_continuous': X_continuous,
        'X_categorical': X_categorical,
        'user_ids': meta['user_id'],
        'dates': meta['date'],
    }


def _apply_event_date_filter(conn, db_path, date_range, config):
    """Optionally replace src.events with a date-filtered copy.

    When ``date_range`` is provided, copies only the relevant events into
    a local table and re-attaches the DuckDB so that ``src.events`` points
    to the filtered data.  All downstream SQL stages reference
    ``src.events`` and work unchanged.

    The start date is extended by ``lookback + rolling_window`` days so
    that rolling statistics and sequence creation have enough history.

    Args:
        conn: DuckDB connection (already attached as ``src``).
        db_path: path to the DuckDB file (needed for re-attach).
        date_range: (start_date, end_date) 'YYYY-MM-DD' strings, or None.
        config: config dict.
    """
    if date_range is None:
        return

    start_date, end_date = date_range
    lookback = config['model'].get('lookback', 60)
    rolling_w = config['processing'].get('rolling_window', 20)
    buffer_days = lookback + rolling_w

    print(f"  Date filter: {start_date} to {end_date} "
          f"(+{buffer_days}d buffer for history)")

    # Materialize filtered events into a local in-memory table
    n_before = conn.execute("SELECT COUNT(*) FROM src.events").fetchone()[0]
    conn.execute(f"""
        CREATE TABLE _filtered_events AS
        SELECT * FROM src.events
        WHERE CAST(datetime AS DATE) >= DATE '{start_date}' - INTERVAL '{buffer_days}' DAY
          AND CAST(datetime AS DATE) <= DATE '{end_date}'
    """)
    n_after = conn.execute("SELECT COUNT(*) FROM _filtered_events").fetchone()[0]
    print(f"  Events: {n_before:,} total → {n_after:,} in window")

    # Detach original DB, re-attach as writable in-memory schema named 'src'
    # so that src.events resolves to the filtered table.
    conn.execute("DETACH src")
    conn.execute("ATTACH ':memory:' AS src")
    conn.execute("CREATE TABLE src.events AS SELECT * FROM _filtered_events")
    conn.execute("DROP TABLE _filtered_events")

    # Re-attach the real DB as 'srcdb' for static tables (users, psychometric, etc.)
    conn.execute(f"ATTACH '{db_path}' AS srcdb (READ_ONLY)")
    # Create views so src.users, src.psychometric, etc. resolve correctly
    for table in ['users', 'psychometric', 'user_changes', 'terminated_users', 'insiders']:
        try:
            conn.execute(f"CREATE VIEW src.{table} AS SELECT * FROM srcdb.{table}")
        except Exception:
            pass  # table may not exist


def run_inference_feature_engineering(db_path, config, artifacts,
                                      user_filter=None, date_range=None):
    """
    Run session-aware feature engineering pipeline for real-time inference.
    
    Reuses the same SQL stages as training (daily aggregation + session detection)
    but applies saved preprocessing artifacts instead of fitting new ones.
    Supports date range filtering for incremental scoring and user filtering
    for targeted analysis.
    
    Pipeline Stages:
    1. Filter events by date range (if specified)
    2. Daily aggregation with activity counts
    3. Session detection and per-session features
    4. Merge session statistics into daily table
    5. Enrich with static features (users, psychometric, changes)
    6. Compute derived features and transformations
    7. Apply saved scaling and encoding artifacts
    
    Args:
        db_path: Path to DuckDB database with raw events and session data
        config: Configuration dict with feature definitions
        artifacts: Preprocessing artifacts dict (scaler, mappings, metadata)
        user_filter: Optional list of user IDs to restrict analysis to
        date_range: Optional (start_date, end_date) tuple in 'YYYY-MM-DD' format.
                   When provided, only events within this window are processed.
                   Lookback period automatically extended for rolling stats.
    
    Returns:
        dict: Feature arrays for inference
            - X_continuous: (n_days, n_continuous) session-aware features
            - X_categorical: (n_days, n_categorical) categorical features  
            - user_ids: (n_days,) user ID per day
            - dates: (n_days,) datetime64 per day
    """
    conn = duckdb.connect()
    conn.execute(f"ATTACH '{db_path}' AS src (READ_ONLY)")
    conn.execute("SET threads = 4")

    try:
        _apply_event_date_filter(conn, db_path, date_range, config)
        _create_daily_aggregation(conn)
        _create_sessions_table(conn)
        _compute_session_features(conn)
        _merge_session_stats(conn)
        _enrich_daily(conn)
        _compute_features(conn, config)
        result = transform_with_saved_artifacts(conn, artifacts, user_filter)
        return result
    finally:
        conn.close()


# =========================================================================
# Persist Sessions to DuckDB
# =========================================================================

def _persist_sessions(conn, db_path):
    """Write the sessions table to the source DuckDB for evaluation use."""
    print("\n[Persist] Saving sessions table to DuckDB...")

    # Check if sessions table still exists in memory
    tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    if 'sessions' not in tables:
        print("  WARNING: sessions table not found in memory, skipping persist.")
        return

    # Re-attach as writable to persist
    conn.execute("DETACH src")
    conn.execute(f"ATTACH '{db_path}' AS src")

    # Drop existing sessions table if present
    conn.execute("DROP TABLE IF EXISTS src.sessions")

    conn.execute("""
        CREATE TABLE src.sessions AS
        SELECT * FROM sessions
    """)

    cnt = conn.execute("SELECT COUNT(*) FROM src.sessions").fetchone()[0]
    print(f"  Persisted {cnt:,} sessions to {db_path}")

    # Re-attach as read-only for safety
    conn.execute("DETACH src")
    conn.execute(f"ATTACH '{db_path}' AS src (READ_ONLY)")


# =========================================================================
# Main Pipeline
# =========================================================================

def run_feature_engineering() -> Path:
    """
    Run complete session-aware feature engineering pipeline.
    
    Pipeline Stages:
    1. Daily aggregation from events (18 features)
    2. Session detection and per-session features
    3. Daily aggregation of session features (18 features)
    4. Merge session stats into daily table
    5. Enrich with static features (users, psychometric, changes)
    6. Compute derived features and transformations
    7. Scale features and export to numpy arrays
    
    Args:
        None (uses config.yaml for paths and settings)
        
    Returns:
        Path: Output directory containing feature arrays and artifacts
        
    Output Files:
        - X_continuous.npy: (n_days, n_continuous_features)
        - X_categorical.npy: (n_days, n_categorical_features)
        - y.npy: (n_days,) binary labels
        - dates.npy: (n_days,) datetime64
        - user_ids.npy: (n_days,) user IDs
        - preprocessing_artifacts.pkl: scaler, mappings, metadata
    """
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
        # Stage 1: Daily aggregation from raw events
        _create_daily_aggregation(conn)
        
        # Stage 1b: Session detection from logon/logoff events
        _create_sessions_table(conn)
        
        # Stage 1c: Per-session feature computation
        _compute_session_features(conn)
        
        # Stage 1d: Merge session stats into daily table
        _merge_session_stats(conn)
        
        # Stage 2: Enrich with static features (users, psychometric, changes)
        _enrich_daily(conn)
        
        # Stage 3: Compute derived features and transformations
        _compute_features(conn, config)
        
        # Stage 4: Scale features and export to numpy arrays
        _scale_and_export(conn, config, output_dir)

        # Persist sessions table to source DuckDB for evaluation
        _persist_sessions(conn, db_path)

        return output_dir
    finally:
        conn.close()


if __name__ == "__main__":
    run_feature_engineering()
