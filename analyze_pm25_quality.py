#!/usr/bin/env python3
"""
PM2.5 Data Quality Analysis Script
Analyzes CSV files in data/processed_data/ for PM2.5 data quality issues.
"""

import pandas as pd
import os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_pm25_file(file_path):
    """
    Analyze a single CSV file for PM2.5 data quality issues.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        dict: Analysis results with file name, issue type, and notes
    """
    file_name = os.path.basename(file_path)
    result = {
        'Station_File_Name': file_name,
        'Issue_Type': 'None',
        'Notes': 'OK'
    }
    
    try:
        # Try to load only PM2.5 column first to check if it exists
        try:
            # Read just the header to check column names
            header = pd.read_csv(file_path, nrows=0)
            pm25_col = None
            
            # Check for different possible PM2.5 column names
            possible_names = ['PM2.5', 'PM2.5 (ug/m3)', 'PM2.5(ug/m3)', 'PM25', 'pm2.5']
            for col_name in possible_names:
                if col_name in header.columns:
                    pm25_col = col_name
                    break
            
            if pm25_col is None:
                result['Issue_Type'] = 'Missing Column'
                result['Notes'] = f'PM2.5 column not found. Available columns: {list(header.columns)[:10]}...'
                return result
        except Exception as e:
            result['Issue_Type'] = 'Read Error'
            result['Notes'] = f'Cannot read file header: {str(e)}'
            return result
        
        # Load only the PM2.5 column efficiently
        df = pd.read_csv(file_path, usecols=[pm25_col])
        
        # Extract PM2.5 column and drop NA values
        pm25_data = df[pm25_col].dropna()
        
        if len(pm25_data) == 0:
            result['Issue_Type'] = 'No Data'
            result['Notes'] = 'All PM2.5 values are NA'
            return result
        
        # Perform descriptive statistics
        stats = pm25_data.describe()
        unique_count = pm25_data.nunique()
        
        # Check for anomalies
        issues = []
        
        # Check if all values are constant
        if unique_count == 1:
            issues.append('All values constant')
        
        # Check if standard deviation is near zero
        if stats['std'] < 1e-6:
            issues.append(f'Near-zero std ({stats["std"]:.2e})')
        
        # Check if maximum value is less than 10
        if stats['max'] < 10:
            issues.append(f'Max too low ({stats["max"]:.2f})')
        
        # Check if maximum value is greater than 1000
        if stats['max'] > 1000:
            issues.append(f'Max too high ({stats["max"]:.2f})')
        
        if issues:
            result['Issue_Type'] = '; '.join(issues)
            result['Notes'] = f'Mean: {stats["mean"]:.2f}, Std: {stats["std"]:.2f}, Min: {stats["min"]:.2f}, Max: {stats["max"]:.2f}, Unique: {unique_count}'
        else:
            result['Notes'] = f'Mean: {stats["mean"]:.2f}, Std: {stats["std"]:.2f}, Min: {stats["min"]:.2f}, Max: {stats["max"]:.2f}, Unique: {unique_count}'
            
    except Exception as e:
        result['Issue_Type'] = 'Read Error'
        result['Notes'] = f'Error processing file: {str(e)}'
    
    return result

def main():
    """Main function to analyze all CSV files in processed_data folder."""
    
    # Base directory
    base_dir = Path("data/processed_data")
    
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist!")
        return
    
    # Collect all CSV files
    csv_files = []
    
    # Check state folders
    for state_folder in ['Delhi', 'Haryana', 'Karnataka', 'Maharashtra']:
        state_path = base_dir / state_folder
        if state_path.exists():
            csv_files.extend(list(state_path.glob("*.csv")))
    
    # Check for unified dataset (but skip it to avoid computational overhead)
    unified_file = base_dir / "unified_dataset.csv"
    if unified_file.exists():
        print(f"Note: Skipping large unified_dataset.csv ({unified_file.stat().st_size / 1024 / 1024:.1f} MB) to save computational power")
    
    if not csv_files:
        print("No CSV files found in state folders!")
        return
    
    print(f"Found {len(csv_files)} CSV files to analyze...")
    print("=" * 80)
    
    # Analyze each file
    results = []
    for i, file_path in enumerate(csv_files, 1):
        print(f"Analyzing {i}/{len(csv_files)}: {file_path.name}...")
        result = analyze_pm25_file(file_path)
        results.append(result)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    # Display results
    print("\n" + "=" * 80)
    print("PM2.5 DATA QUALITY ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Show all results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 60)
    
    print(summary_df.to_string(index=False))
    
    # Identify problematic files
    problematic_files = summary_df[summary_df['Issue_Type'] != 'None']
    
    print(f"\n" + "=" * 80)
    print("FILES WITH ISSUES (RECOMMENDED FOR EXCLUSION/REPROCESSING)")
    print("=" * 80)
    
    if len(problematic_files) > 0:
        print(f"Found {len(problematic_files)} files with issues:")
        for _, row in problematic_files.iterrows():
            print(f"- {row['Station_File_Name']}: {row['Issue_Type']}")
        
        print(f"\nFiles to exclude/reprocess:")
        exclude_list = problematic_files['Station_File_Name'].tolist()
        print(exclude_list)
    else:
        print("No problematic files found! All files have acceptable PM2.5 data.")
    
    # Save summary to file
    summary_df.to_csv('pm25_quality_analysis_summary.csv', index=False)
    print(f"\nSummary saved to: pm25_quality_analysis_summary.csv")

if __name__ == "__main__":
    main()
