"""
Parquet to DuckDB Ingestion
===========================
Load Parquet files into DuckDB for SQL-based feature engineering.
Creates unified events table, users table, and labels table.
"""

import os
import sys
from pathlib import Path
import duckdb
import yaml

try:
    from ..utils.common import load_config
except (ImportError, ValueError):
    def load_config():
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)


def create_database(config: dict) -> duckdb.DuckDBPyConnection:
    """Create DuckDB database and return connection."""
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / config['data']['duckdb_path']
    
    # Create directory if needed
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing database for fresh start
    if db_path.exists():
        os.remove(db_path)
    
    print(f"Creating database: {db_path}")
    return duckdb.connect(str(db_path))


def create_events_table(conn: duckdb.DuckDBPyConnection, parquet_dir: Path):
    """Create unified events table from all activity Parquet files."""
    print("\n--- Creating events table ---")
    
    # Build SQL dynamically based on which parquet files exist
    queries = []
    
    # Logon events
    logon_path = parquet_dir / "logon.parquet"
    if logon_path.exists():
        queries.append(f"""
        SELECT 
            id,
            strptime(date, '%m/%d/%Y %H:%M:%S') as datetime,
            "user" as user_id,
            pc,
            activity,
            NULL as to_addr,
            NULL as cc,
            NULL as bcc,
            NULL as size,
            NULL as attachments,
            NULL as url,
            NULL as filename,
            NULL as content
        FROM read_parquet('{logon_path}')
        """)
        print(f"  Including: logon.parquet")
    
    # Device events
    device_path = parquet_dir / "device.parquet"
    if device_path.exists():
        queries.append(f"""
        SELECT 
            id,
            strptime(date, '%m/%d/%Y %H:%M:%S') as datetime,
            "user" as user_id,
            pc,
            activity,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
        FROM read_parquet('{device_path}')
        """)
        print(f"  Including: device.parquet")
    
    # Email events
    email_path = parquet_dir / "email.parquet"
    if email_path.exists():
        queries.append(f"""
        SELECT 
            id,
            strptime(date, '%m/%d/%Y %H:%M:%S') as datetime,
            "user" as user_id,
            pc,
            'Email' as activity,
            "to" as to_addr,
            cc,
            bcc,
            size,
            attachments,
            NULL as url,
            NULL as filename,
            content
        FROM read_parquet('{email_path}')
        """)
        print(f"  Including: email.parquet")
    
    # File events
    file_path = parquet_dir / "file.parquet"
    if file_path.exists():
        queries.append(f"""
        SELECT 
            id,
            strptime(date, '%m/%d/%Y %H:%M:%S') as datetime,
            "user" as user_id,
            pc,
            'File' as activity,
            NULL, NULL, NULL, NULL, NULL, NULL,
            filename,
            content
        FROM read_parquet('{file_path}')
        """)
        print(f"  Including: file.parquet")
    
    # HTTP events (optional - may fail to convert due to size)
    http_path = parquet_dir / "http.parquet"
    if http_path.exists():
        queries.append(f"""
        SELECT 
            id,
            strptime(date, '%m/%d/%Y %H:%M:%S') as datetime,
            "user" as user_id,
            pc,
            'Http' as activity,
            NULL, NULL, NULL, NULL, NULL,
            url,
            NULL as filename,
            content
        FROM read_parquet('{http_path}')
        """)
        print(f"  Including: http.parquet")
    else:
        print(f"  WARNING: http.parquet not found - HTTP events will be missing!")
    
    if not queries:
        raise ValueError("No parquet files found!")
    
    # Combine queries with UNION ALL
    sql = "CREATE TABLE events AS\n" + "\nUNION ALL\n".join(queries)
    
    conn.execute(sql)
    
    # Create index on user_id and datetime for fast session queries
    conn.execute("CREATE INDEX idx_events_user_time ON events(user_id, datetime)")
    
    # Get row count
    count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    print(f"  Total events: {count:,}")
    
    # Show activity breakdown
    activities = conn.execute("""
        SELECT activity, COUNT(*) as cnt 
        FROM events 
        GROUP BY activity
        ORDER BY cnt DESC
    """).fetchall()
    for act, cnt in activities:
        print(f"    {act}: {cnt:,}")
    
    return count


def create_users_table(conn: duckdb.DuckDBPyConnection, parquet_dir: Path):
    """Create users table from LDAP data (latest record per user)."""
    print("\n--- Creating users table ---")
    
    sql = f"""
    CREATE TABLE users AS
    WITH ranked AS (
        SELECT 
            user_id,
            employee_name,
            email,
            role,
            business_unit,
            functional_unit,
            department,
            team,
            supervisor,
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY user_id DESC) as rn
        FROM read_parquet('{parquet_dir}/ldap.parquet')
    )
    SELECT 
        user_id,
        employee_name,
        email,
        role,
        business_unit,
        functional_unit,
        department,
        team,
        supervisor
    FROM ranked
    WHERE rn = 1
    """
    
    conn.execute(sql)
    
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    print(f"  Total users: {count:,}")
    
    return count


def create_labels_table(conn: duckdb.DuckDBPyConnection, raw_dir: Path):
    """Create insider labels table from ground truth."""
    print("\n--- Creating labels table ---")
    
    insiders_path = raw_dir / "answers" / "insiders.csv"
    
    sql = f"""
    CREATE TABLE insiders AS
    SELECT 
        dataset,
        scenario,
        details,
        "user" as user_id,
        strptime(start, '%m/%d/%Y %H:%M:%S') as start_date,
        strptime("end", '%m/%d/%Y %H:%M:%S') as end_date
    FROM read_csv('{insiders_path}', header=true)
    WHERE dataset = '4.2'
    """
    
    conn.execute(sql)
    
    count = conn.execute("SELECT COUNT(*) FROM insiders").fetchone()[0]
    print(f"  Insider users (dataset 4.2): {count}")
    
    # Show scenario breakdown
    scenarios = conn.execute("""
        SELECT scenario, COUNT(*) as count 
        FROM insiders 
        GROUP BY scenario
        ORDER BY scenario
    """).fetchall()
    for scenario, cnt in scenarios:
        print(f"    Scenario {scenario}: {cnt} users")
    
    return count


def create_feature_encoding_tables(conn: duckdb.DuckDBPyConnection, config: dict):
    """Create lookup tables for feature encodings."""
    print("\n--- Creating encoding tables ---")
    
    # Functional unit encoding
    fu_mapping = config['encodings']['functional_unit']
    values = [(k, v) for k, v in fu_mapping.items()]
    
    conn.execute("""
        CREATE TABLE functional_unit_encoding (
            name VARCHAR,
            code INTEGER
        )
    """)
    conn.executemany(
        "INSERT INTO functional_unit_encoding VALUES (?, ?)",
        values
    )
    print(f"  Functional unit encodings: {len(values)}")
    


def create_psychometric_table(conn: duckdb.DuckDBPyConnection, parquet_dir: Path):
    """Create psychometric table from Big Five personality scores."""
    print("\n--- Creating psychometric table ---")
    
    psycho_path = parquet_dir / "psychometric.parquet"
    if not psycho_path.exists():
        print("  WARNING: psychometric.parquet not found, skipping.")
        return 0
    
    sql = f"""
    CREATE TABLE psychometric AS
    SELECT 
        user_id,
        O as openness,
        C as conscientiousness,
        E as extraversion,
        A as agreeableness,
        N as neuroticism
    FROM read_parquet('{psycho_path}')
    """
    
    conn.execute(sql)
    
    count = conn.execute("SELECT COUNT(*) FROM psychometric").fetchone()[0]
    print(f"  Users with psychometric data: {count}")
    
    # Show score ranges
    stats = conn.execute("""
        SELECT 
            MIN(openness) as min_o, MAX(openness) as max_o,
            MIN(conscientiousness) as min_c, MAX(conscientiousness) as max_c,
            MIN(extraversion) as min_e, MAX(extraversion) as max_e,
            MIN(agreeableness) as min_a, MAX(agreeableness) as max_a,
            MIN(neuroticism) as min_n, MAX(neuroticism) as max_n
        FROM psychometric
    """).fetchone()
    print(f"    O: [{stats[0]}-{stats[1]}], C: [{stats[2]}-{stats[3]}], "
          f"E: [{stats[4]}-{stats[5]}], A: [{stats[6]}-{stats[7]}], "
          f"N: [{stats[8]}-{stats[9]}]")
    
    return count


def create_user_changes_table(conn: duckdb.DuckDBPyConnection, parquet_dir: Path):
    """
    Detect role/department/functional_unit changes across LDAP monthly snapshots.
    
    Creates a table with one row per user per month where any change occurred.
    """
    print("\n--- Creating user_changes table ---")
    
    ldap_path = parquet_dir / "ldap.parquet"
    if not ldap_path.exists():
        print("  WARNING: ldap.parquet not found, skipping.")
        return 0
    
    # The LDAP parquet has all monthly snapshots merged.
    # We need to reconstruct month info from the source files.
    # Since we don't have a month column, we'll load from raw LDAP CSVs
    # and process them with month identifiers.
    import pyarrow.csv as pa_csv
    import pyarrow as pa
    
    project_root = Path(__file__).parent.parent.parent
    config = load_config()
    raw_dir = project_root / config['data']['raw_dir']
    ldap_dir = raw_dir / "LDAP"
    
    if not ldap_dir.exists():
        print("  WARNING: LDAP directory not found, skipping.")
        return 0
    
    # Load all LDAP files with month identifier
    ldap_files = sorted(ldap_dir.glob("*.csv"))
    if not ldap_files:
        print("  WARNING: No LDAP CSV files found, skipping.")
        return 0
    
    # Register each monthly file as a temporary table, then union with month column
    monthly_parts = []
    for csv_file in ldap_files:
        month_str = csv_file.stem  # e.g., "2010-01"
        monthly_parts.append(f"""
            SELECT *, '{month_str}' as ldap_month
            FROM read_csv('{csv_file}', header=true)
        """)
    
    # Create a unified LDAP view with month column
    union_sql = " UNION ALL ".join(monthly_parts)
    
    # Detect changes using LAG() window function
    sql = f"""
    CREATE TABLE user_changes AS
    WITH ldap_monthly AS (
        {union_sql}
    ),
    with_prev AS (
        SELECT 
            user_id,
            ldap_month,
            role,
            department,
            functional_unit,
            LAG(role) OVER (PARTITION BY user_id ORDER BY ldap_month) as prev_role,
            LAG(department) OVER (PARTITION BY user_id ORDER BY ldap_month) as prev_department,
            LAG(functional_unit) OVER (PARTITION BY user_id ORDER BY ldap_month) as prev_functional_unit
        FROM ldap_monthly
    )
    SELECT 
        user_id,
        ldap_month,
        CASE WHEN prev_role IS NOT NULL AND role != prev_role THEN 1 ELSE 0 END as role_changed,
        CASE WHEN prev_department IS NOT NULL AND department != prev_department THEN 1 ELSE 0 END as dept_changed,
        CASE WHEN prev_functional_unit IS NOT NULL AND functional_unit != prev_functional_unit THEN 1 ELSE 0 END as fu_changed
    FROM with_prev
    WHERE prev_role IS NOT NULL  -- skip first month (no previous to compare)
    """
    
    conn.execute(sql)
    
    total = conn.execute("SELECT COUNT(*) FROM user_changes").fetchone()[0]
    role_changes = conn.execute("SELECT SUM(role_changed) FROM user_changes").fetchone()[0]
    dept_changes = conn.execute("SELECT SUM(dept_changed) FROM user_changes").fetchone()[0]
    fu_changes = conn.execute("SELECT SUM(fu_changed) FROM user_changes").fetchone()[0]
    
    print(f"  Total user-month records: {total:,}")
    print(f"  Role changes detected: {role_changes}")
    print(f"  Department changes detected: {dept_changes}")
    print(f"  Functional unit changes detected: {fu_changes}")
    
    return total


def create_terminated_users_table(conn: duckdb.DuckDBPyConnection, parquet_dir: Path):
    """
    Detect terminated users — those who disappear from LDAP between consecutive months.
    """
    print("\n--- Creating terminated_users table ---")
    
    project_root = Path(__file__).parent.parent.parent
    config = load_config()
    raw_dir = project_root / config['data']['raw_dir']
    ldap_dir = raw_dir / "LDAP"
    
    if not ldap_dir.exists():
        print("  WARNING: LDAP directory not found, skipping.")
        return 0
    
    ldap_files = sorted(ldap_dir.glob("*.csv"))
    if len(ldap_files) < 2:
        print("  WARNING: Need at least 2 LDAP files for termination detection.")
        return 0
    
    # For each consecutive pair of months, find users in month N but not in month N+1
    termination_parts = []
    for i in range(len(ldap_files) - 1):
        current_file = ldap_files[i]
        next_file = ldap_files[i + 1]
        next_month = next_file.stem
        
        termination_parts.append(f"""
            SELECT 
                c.user_id,
                '{next_month}' as termination_month
            FROM read_csv('{current_file}', header=true) c
            WHERE c.user_id NOT IN (
                SELECT user_id FROM read_csv('{next_file}', header=true)
            )
        """)
    
    union_sql = " UNION ALL ".join(termination_parts)
    
    # Take the earliest termination month per user
    sql = f"""
    CREATE TABLE terminated_users AS
    WITH all_terminations AS (
        {union_sql}
    )
    SELECT 
        user_id,
        MIN(termination_month) as termination_month
    FROM all_terminations
    GROUP BY user_id
    """
    
    conn.execute(sql)
    
    count = conn.execute("SELECT COUNT(*) FROM terminated_users").fetchone()[0]
    print(f"  Terminated users detected: {count}")
    
    if count > 0:
        # Show monthly breakdown
        monthly = conn.execute("""
            SELECT termination_month, COUNT(*) as cnt
            FROM terminated_users
            GROUP BY termination_month
            ORDER BY termination_month
        """).fetchall()
        for month, cnt in monthly:
            print(f"    {month}: {cnt} users")
    
    return count


def print_summary(conn: duckdb.DuckDBPyConnection):
    """Print database summary."""
    print("\n" + "=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)
    
    tables = conn.execute("SHOW TABLES").fetchall()
    for (table,) in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count:,} rows")


def main():
    """Main entry point for Parquet to DuckDB ingestion."""
    config = load_config()
    
    project_root = Path(__file__).parent.parent.parent
    parquet_dir = project_root / config['data']['parquet_dir']
    raw_dir = project_root / config['data']['raw_dir']
    
    print("=" * 60)
    print("Parquet to DuckDB Ingestion")
    print("=" * 60)
    print(f"Parquet dir: {parquet_dir}")
    
    # Check if parquet files exist
    if not (parquet_dir / "logon.parquet").exists():
        print("\nERROR: Parquet files not found. Run csv_to_parquet.py first.")
        sys.exit(1)
    
    # Create database
    conn = create_database(config)
    
    try:
        # Create tables
        create_events_table(conn, parquet_dir)
        create_users_table(conn, parquet_dir)
        create_labels_table(conn, raw_dir)
        create_feature_encoding_tables(conn, config)
        create_psychometric_table(conn, parquet_dir)
        create_user_changes_table(conn, parquet_dir)
        create_terminated_users_table(conn, parquet_dir)
        
        # Print summary
        print_summary(conn)
        
        print("\n" + "=" * 60)
        print("DuckDB database created successfully!")
        print("=" * 60)
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
