"""
01_prepare_data.py — Data Preparation Pipeline

Step 1: Convert CSV files to Parquet format (including psychometric.csv)
Step 2: Ingest Parquet files into DuckDB (events, users, insiders,
        psychometric, user_changes, terminated_users, functional_unit_encoding)
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
        print("  Sources: logon, device, email, file, http, LDAP, psychometric")
        convert_to_parquet()
    except Exception as e:
        print(f"\nERROR in CSV-to-Parquet conversion: {e}")
        sys.exit(1)

    try:
        print("\n[STEP 2/2] Ingesting Parquet files into DuckDB...")
        print("  Tables: events, users, insiders, psychometric,")
        print("          user_changes, terminated_users, encodings")
        ingest_to_duckdb()
    except Exception as e:
        print(f"\nERROR in DuckDB ingestion: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 70)
    print("\nNext step: Run 02_feature_engineering.py to extract features")


if __name__ == "__main__":
    main()
