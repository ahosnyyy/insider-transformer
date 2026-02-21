"""
CSV to Parquet Converter
========================
Converts large CSV files to Parquet format for efficient storage and processing.
Handles the 14GB+ HTTP logs using true chunked/streaming processing.
"""

import os
import sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as csv
import yaml

try:
    from ..utils.common import load_config
except (ImportError, ValueError):
    def load_config():
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)


def convert_csv_to_parquet_chunked(
    csv_path: Path,
    parquet_path: Path,
    chunk_size_mb: int = 128
) -> int:
    """
    Convert a large CSV file to Parquet using streaming/chunked processing.
    
    This uses PyArrow's streaming reader to avoid loading entire file in memory.
    
    Args:
        csv_path: Path to input CSV file
        parquet_path: Path to output Parquet file
        chunk_size_mb: Block size in MB for streaming read
        
    Returns:
        Total number of rows converted
    """
    print(f"\nConverting: {csv_path.name}")
    file_size_gb = csv_path.stat().st_size / 1e9
    print(f"  Size: {file_size_gb:.2f} GB")
    
    # For large files (>1GB), use streaming reader
    if file_size_gb > 1.0:
        print(f"  Using chunked streaming (block_size={chunk_size_mb}MB)...")
        return convert_large_csv_streaming(csv_path, parquet_path, chunk_size_mb)
    else:
        return convert_small_csv(csv_path, parquet_path)


def convert_small_csv(csv_path: Path, parquet_path: Path) -> int:
    """Convert small CSV files (< 1GB) using standard approach."""
    try:
        table = csv.read_csv(csv_path)
        
        pq.write_table(
            table,
            parquet_path,
            compression='snappy',
            row_group_size=100_000
        )
        
        total_rows = table.num_rows
        print(f"  Rows: {total_rows:,}")
        print(f"  Output: {parquet_path}")
        print(f"  Parquet size: {parquet_path.stat().st_size / 1e6:.1f} MB")
        
        return total_rows
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0


def convert_large_csv_streaming(
    csv_path: Path,
    parquet_path: Path,
    chunk_size_mb: int = 128
) -> int:
    """
    Convert large CSV files using streaming reader.
    
    Reads file in chunks and writes to Parquet incrementally.
    """
    try:
        # Configure streaming reader
        read_options = csv.ReadOptions(block_size=chunk_size_mb * 1024 * 1024)
        
        # Open streaming reader
        reader = csv.open_csv(
            csv_path,
            read_options=read_options
        )
        
        # Get schema from first batch
        writer = None
        total_rows = 0
        batch_count = 0
        
        for batch in reader:
            if writer is None:
                # Create writer with schema from first batch
                schema = batch.schema
                writer = pq.ParquetWriter(
                    parquet_path,
                    schema,
                    compression='snappy'
                )
            
            # Write batch
            table = pa.Table.from_batches([batch])
            writer.write_table(table)
            
            total_rows += batch.num_rows
            batch_count += 1
            
            if batch_count % 10 == 0:
                print(f"    Processed {total_rows:,} rows ({batch_count} batches)...")
        
        if writer is not None:
            writer.close()
        
        print(f"  Rows: {total_rows:,}")
        print(f"  Output: {parquet_path}")
        print(f"  Parquet size: {parquet_path.stat().st_size / 1e6:.1f} MB")
        
        return total_rows
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 0


def convert_ldap_files(raw_dir: Path, parquet_dir: Path) -> int:
    """
    Merge all LDAP monthly files into a single Parquet file.
    Takes the latest record per user.
    """
    ldap_dir = raw_dir / "LDAP"
    if not ldap_dir.exists():
        print(f"LDAP directory not found: {ldap_dir}")
        return 0
    
    print(f"\nMerging LDAP files from: {ldap_dir}")
    
    # Read all LDAP files
    tables = []
    for csv_file in sorted(ldap_dir.glob("*.csv")):
        table = csv.read_csv(csv_file)
        tables.append(table)
    
    if not tables:
        print("  No LDAP files found")
        return 0
    
    # Concatenate all tables
    merged = pa.concat_tables(tables)
    
    # Write to parquet (we'll deduplicate later in DuckDB)
    parquet_path = parquet_dir / "ldap.parquet"
    pq.write_table(merged, parquet_path, compression='snappy')
    
    print(f"  Total LDAP records: {merged.num_rows:,}")
    print(f"  Output: {parquet_path}")
    
    return merged.num_rows


def main():
    """Main entry point for CSV to Parquet conversion."""
    config = load_config()
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    raw_dir = project_root / config['data']['raw_dir']
    parquet_dir = project_root / config['data']['parquet_dir']
    
    # Create output directory
    parquet_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CSV to Parquet Conversion")
    print("=" * 60)
    print(f"Input:  {raw_dir}")
    print(f"Output: {parquet_dir}")
    
    # Files to convert (order matters - small files first)
    csv_files = [
        "logon.csv",
        "device.csv", 
        "email.csv",
        "file.csv",
        "http.csv",  # Large file ~14GB - processed last with streaming
    ]
    
    total_rows = 0
    failed_files = []
    
    for csv_name in csv_files:
        csv_path = raw_dir / csv_name
        if csv_path.exists():
            parquet_name = csv_name.replace('.csv', '.parquet')
            parquet_path = parquet_dir / parquet_name
            
            # Skip if already converted
            if parquet_path.exists():
                print(f"\nSkipping {csv_name} (already converted)")
                continue
                
            rows = convert_csv_to_parquet_chunked(csv_path, parquet_path)
            if rows == 0:
                failed_files.append(csv_name)
            total_rows += rows
        else:
            print(f"\nWarning: {csv_name} not found")
    
    # Convert LDAP files
    ldap_rows = convert_ldap_files(raw_dir, parquet_dir)
    total_rows += ldap_rows
    
    # Convert psychometric file
    psycho_path = raw_dir / "psychometric.csv"
    psycho_parquet = parquet_dir / "psychometric.parquet"
    if psycho_path.exists():
        if psycho_parquet.exists():
            print(f"\nSkipping psychometric.csv (already converted)")
        else:
            print(f"\nConverting: psychometric.csv")
            rows = convert_small_csv(psycho_path, psycho_parquet)
            total_rows += rows
    else:
        print(f"\nWarning: psychometric.csv not found")
    
    print("\n" + "=" * 60)
    print(f"Conversion complete! Total rows: {total_rows:,}")
    if failed_files:
        print(f"WARNING: Failed to convert: {failed_files}")
    print("=" * 60)


if __name__ == "__main__":
    main()
