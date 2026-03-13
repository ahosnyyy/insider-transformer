#!/usr/bin/env python3
"""
Inspect the insider.duckdb database created by parquet_to_duckdb.py.
ATTACHes the DB as 'src' schema (matching feature_engineering.py).

Default DB path: data/processed/insider.duckdb
Tables: events, users, insiders, psychometric,
        user_changes, terminated_users, functional_unit_encoding
"""

import sys
import argparse
from pathlib import Path

try:
    import duckdb
except ImportError:
    print("Error: duckdb not installed. Run: pip install duckdb")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "processed" / "insider.duckdb"

KNOWN_TABLES = [
    "events", "users", "insiders", "psychometric",
    "user_changes", "terminated_users", "functional_unit_encoding",
]


def connect(db_path):
    """Connect to DuckDB and ATTACH the database as 'src' schema."""
    db = Path(db_path)
    if not db.exists():
        print(f"ERROR: Database not found: {db}")
        print("Run 'python scripts/01_prepare_data.py' first.")
        sys.exit(1)

    conn = duckdb.connect()
    conn.execute(f"ATTACH '{db}' AS src (READ_ONLY)")
    print(f"Connected to: {db}")
    return conn


def list_tables(conn):
    """List all tables in the src schema."""
    print("\n=== Tables in src schema ===")
    for table in KNOWN_TABLES:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM src.{table}").fetchone()[0]
            print(f"  src.{table}: {count:,} rows")
        except Exception:
            print(f"  src.{table}: NOT FOUND")


def inspect_table(conn, table_name, limit=10):
    """Inspect a table: schema, row count, sample data, stats."""
    qualified = f"src.{table_name}"
    print(f"\n{'=' * 60}")
    print(f"Table: {qualified}")
    print("=" * 60)

    try:
        schema = conn.execute(f"DESCRIBE {qualified}").fetchall()
    except Exception as e:
        print(f"Error: {e}")
        return

    # Schema
    print(f"\nSchema ({len(schema)} columns):")
    for col in schema:
        print(f"  {col[0]}: {col[1]}")

    count = conn.execute(f"SELECT COUNT(*) FROM {qualified}").fetchone()[0]
    print(f"\nTotal rows: {count:,}")
    if count == 0:
        return

    # Sample rows
    col_names = [col[0] for col in schema]
    sample = conn.execute(f"SELECT * FROM {qualified} LIMIT {limit}").fetchall()
    print(f"\nSample data ({len(sample)} rows):")
    print(f"  {' | '.join(col_names)}")
    print(f"  {'-' * min(120, len(' | '.join(col_names)))}")
    for row in sample:
        vals = []
        for v in row:
            if v is None:
                vals.append("NULL")
            elif isinstance(v, str) and len(v) > 50:
                vals.append(v[:47] + "...")
            else:
                vals.append(str(v))
        print(f"  {' | '.join(vals)}")

    # Numeric stats
    numeric_cols = [
        col[0] for col in schema
        if any(t in col[1].upper() for t in ["INT", "DOUBLE", "FLOAT", "BIGINT"])
    ]
    if numeric_cols:
        print(f"\nNumeric column stats:")
        for col_name in numeric_cols:
            try:
                stats = conn.execute(f"""
                    SELECT
                        COUNT({col_name}) AS non_null,
                        COUNT(DISTINCT {col_name}) AS distinct_vals,
                        MIN({col_name}), MAX({col_name}), AVG({col_name})
                    FROM {qualified}
                """).fetchone()
                print(f"  {col_name}: {stats[0]:,} non-null, "
                      f"{stats[1]:,} distinct, "
                      f"range [{stats[2]} .. {stats[3]}], "
                      f"avg {stats[4]:.2f}")
            except Exception:
                pass

    # Categorical top values
    varchar_cols = [col[0] for col in schema if "VARCHAR" in col[1].upper()]
    if varchar_cols:
        print(f"\nTop values for text columns:")
        for col_name in varchar_cols:
            try:
                top = conn.execute(f"""
                    SELECT {col_name}, COUNT(*) AS cnt
                    FROM {qualified}
                    WHERE {col_name} IS NOT NULL
                    GROUP BY {col_name}
                    ORDER BY cnt DESC
                    LIMIT 5
                """).fetchall()
                print(f"  {col_name}:")
                for val, cnt in top:
                    display = str(val)[:40]
                    print(f"    {display}: {cnt:,}")
            except Exception:
                pass


def inspect_events(conn):
    """Deep inspection of src.events table."""
    print("\n" + "=" * 60)
    print("Events Table — Deep Analysis")
    print("=" * 60)

    try:
        conn.execute("SELECT 1 FROM src.events LIMIT 1")
    except Exception:
        print("src.events not found.")
        return

    # Time range
    time_range = conn.execute("""
        SELECT
            MIN(datetime), MAX(datetime),
            COUNT(DISTINCT CAST(datetime AS DATE))
        FROM src.events
    """).fetchone()
    print(f"\nTime range: {time_range[0]} to {time_range[1]}")
    print(f"Unique days: {time_range[2]:,}")

    # Users
    user_stats = conn.execute("""
        SELECT COUNT(DISTINCT user_id), MIN(user_id), MAX(user_id)
        FROM src.events
    """).fetchone()
    print(f"Users: {user_stats[0]:,} (range: {user_stats[1]} to {user_stats[2]})")

    # Activity breakdown
    activities = conn.execute("""
        SELECT activity, COUNT(*) AS cnt
        FROM src.events
        GROUP BY activity
        ORDER BY cnt DESC
    """).fetchall()
    print(f"\nActivity types:")
    for act, cnt in activities:
        print(f"  {act}: {cnt:,}")

    # PC stats
    pc = conn.execute("""
        SELECT COUNT(DISTINCT pc), COUNT(*) FILTER (WHERE pc IS NULL)
        FROM src.events
    """).fetchone()
    print(f"\nPCs: {pc[0]:,} unique, {pc[1]:,} NULL")

    # Top 3 users by activity
    top_users = conn.execute("""
        SELECT user_id, COUNT(*) AS cnt
        FROM src.events
        GROUP BY user_id
        ORDER BY cnt DESC
        LIMIT 3
    """).fetchall()
    print(f"\nTop users by event count:")
    for uid, cnt in top_users:
        print(f"  {uid}: {cnt:,} events")
        sample = conn.execute("""
            SELECT datetime, activity, pc, url, to_addr
            FROM src.events
            WHERE user_id = ?
            ORDER BY datetime
            LIMIT 5
        """, [uid]).fetchall()
        for row in sample:
            line = f"    {row[0]}: {row[1]} on {row[2]}"
            if row[3]:
                line += f"  url={row[3][:40]}"
            if row[4]:
                line += f"  to={row[4][:30]}"
            print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect insider.duckdb (ATTACH as src schema)"
    )
    parser.add_argument(
        "database", nargs="?", default=str(DEFAULT_DB),
        help=f"Path to DuckDB file (default: {DEFAULT_DB})"
    )
    parser.add_argument("--table", help="Inspect a specific table (e.g. events, users)")
    parser.add_argument("--limit", type=int, default=10, help="Sample rows to show")
    parser.add_argument("--events", action="store_true", help="Deep events analysis")
    parser.add_argument("--all", action="store_true", help="Inspect all tables")
    args = parser.parse_args()

    conn = connect(args.database)

    try:
        if args.table:
            inspect_table(conn, args.table, args.limit)
        elif args.events:
            inspect_events(conn)
        elif args.all:
            list_tables(conn)
            for table in KNOWN_TABLES:
                try:
                    inspect_table(conn, table, args.limit)
                except Exception as e:
                    print(f"\nSkipping {table}: {e}")
        else:
            list_tables(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
