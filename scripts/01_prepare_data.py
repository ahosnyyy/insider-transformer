"""
01_prepare_data.py — Data Preparation Pipeline

Step 1: Convert CSV files to Parquet format for efficient storage
        - Activity logs: logon, device, email, file, http
        - LDAP monthly snapshots (merged into single file)
        - Psychometric assessment data
        - Handles large files (>14GB HTTP logs) with streaming conversion

Step 2: Ingest Parquet files into DuckDB database
        - Creates unified events table from all activity types
        - Builds users table from latest LDAP record per user
        - Loads insider threat labels from ground truth
        - Detects role/department changes across months
        - Identifies terminated users
        - Stores psychometric scores and functional unit encodings

Output: data/processed/insider.duckdb with 7 tables ready for feature engineering
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.csv_to_parquet import main as convert_to_parquet
from src.data.parquet_to_duckdb import main as ingest_to_duckdb


def main():
    print("=" * 70)
    print("INSIDER THREAT DETECTION — DATA PREPARATION")
    print("=" * 70)

    try:
        print("\n[STEP 1/2] Converting CSV files to Parquet format...")
        print("  Converting: logon, device, email, file, http (streaming), LDAP, psychometric")
        convert_to_parquet()
    except Exception as e:
        print(f"\nERROR in CSV-to-Parquet conversion: {e}")
        sys.exit(1)

    try:
        print("\n[STEP 2/2] Ingesting Parquet files into DuckDB...")
        print("  Creating: events, users, insiders, psychometric,")
        print("            user_changes, terminated_users, functional_unit_encoding")
        ingest_to_duckdb()
    except Exception as e:
        print(f"\nERROR in DuckDB ingestion: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 70)
    print("\nNext step: Run python scripts/02_feature_engineering.py to extract features")


if __name__ == "__main__":
    main()
