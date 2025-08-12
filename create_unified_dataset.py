#!/usr/bin/env python3
"""
Create Unified Dataset for AEP 3.0 Training

This script combines all processed station datasets into one unified dataset
for machine learning model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_unified_dataset():
    """Create unified dataset from all processed station data."""
    
    # Define paths
    processed_dir = Path("data/processed_data")
    output_path = processed_dir / "unified_dataset.csv"
    
    # Find all processed CSV files
    dataset_files = []
    states = ["Delhi", "Haryana", "Karnataka", "Maharashtra"]
    
    for state in states:
        state_dir = processed_dir / state
        if state_dir.exists():
            csv_files = list(state_dir.glob("*.csv"))
            dataset_files.extend(csv_files)
            logger.info(f"Found {len(csv_files)} datasets in {state}")
    
    if not dataset_files:
        logger.error("No processed datasets found!")
        return
    
    logger.info(f"Total datasets to merge: {len(dataset_files)}")
    
    # Create unified dataset using memory-efficient chunked approach
    logger.info("Creating unified dataset using memory-efficient approach...")
    
    # First pass: get column structure and count records
    sample_df = pd.read_csv(dataset_files[0], nrows=5)
    base_columns = list(sample_df.columns)
    total_records = 0
    
    # Count total records first
    for file_path in dataset_files:
        try:
            df_info = pd.read_csv(file_path, nrows=0)  # Just get column info
            record_count = sum(1 for _ in open(file_path)) - 1  # Count lines minus header
            total_records += record_count
            logger.info(f"Found {file_path.stem} ({file_path.parent.name}): {record_count} records")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
    
    logger.info(f"Total records to process: {total_records:,}")
    
    # Create output file and write header
    output_columns = base_columns + ['state', 'station_id', 'station_state']
    
    # Write datasets one by one to avoid memory issues
    first_write = True
    processed_records = 0
    
    for i, file_path in enumerate(dataset_files, 1):
        try:
            # Extract station info from file path
            state = file_path.parent.name
            station = file_path.stem
            
            logger.info(f"Processing {i}/{len(dataset_files)}: {station} ({state})")
            
            # Read dataset in chunks to manage memory
            chunk_size = 10000
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # Add metadata columns
                chunk['state'] = state
                chunk['station_id'] = station
                chunk['station_state'] = f"{station}_{state}"
                
                # Write to output file
                chunk.to_csv(output_path, mode='a', header=first_write, index=False)
                first_write = False
                
                processed_records += len(chunk)
                
                # Clear chunk from memory
                del chunk
            
            logger.info(f"✓ Completed {station}: {processed_records:,}/{total_records:,} records processed")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    # Read final result for summary (just metadata, not full dataset)
    logger.info("Reading final dataset for summary...")
    try:
        # Read just a sample for column info
        sample_unified = pd.read_csv(output_path, nrows=1000)
        final_record_count = sum(1 for _ in open(output_path)) - 1  # Count total lines
    except Exception as e:
        logger.error(f"Error reading final dataset: {e}")
        return
    
    # Print summary statistics
    print("\n" + "="*60)
    print("UNIFIED DATASET CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"📁 Output file: {output_path}")
    print(f"📊 Total records: {final_record_count:,}")
    print(f"📋 Total columns: {len(sample_unified.columns)}")
    print(f"🏢 States included: {len(states)}")
    print(f"🏭 Stations included: {len(dataset_files)}")
    
    # Show dataset breakdown by state (from sample)
    print("\n📈 Records per state (estimated from sample):")
    if 'state' in sample_unified.columns:
        state_counts = sample_unified['state'].value_counts()
        for state, count in state_counts.items():
            estimated_total = int(count * final_record_count / len(sample_unified))
            print(f"  {state}: ~{estimated_total:,} records")
    
    # Show column information
    print(f"\n📋 Dataset columns ({len(sample_unified.columns)}):")
    for i, col in enumerate(sample_unified.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Show data types
    print(f"\n🔢 Data types:")
    dtype_counts = sample_unified.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Show file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n💾 File size: {file_size_mb:.1f} MB")
    
    print("\n🎯 Dataset ready for ML model training!")
    print("="*60)
    
    return output_path

if __name__ == "__main__":
    create_unified_dataset()
