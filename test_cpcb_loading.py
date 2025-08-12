#!/usr/bin/env python3
"""
Test script to verify CPCB data loading functionality
"""

import pandas as pd
import pytz
from pathlib import Path

def test_cpcb_loading():
    """Test CPCB data loading for station DL009"""
    
    # Setup paths and timezone
    data_root = Path("data")
    cpcb_dir = data_root / "cpcb"
    ist_tz = pytz.timezone('Asia/Kolkata')
    utc_tz = pytz.UTC
    
    # Test station
    state = "Delhi"
    station_id = "DL009"
    
    print(f"Testing CPCB data loading for {station_id} in {state}")
    print("=" * 60)
    
    # Check if file exists
    cpcb_file = cpcb_dir / state / f"{station_id}.csv"
    print(f"CPCB file path: {cpcb_file}")
    print(f"File exists: {cpcb_file.exists()}")
    
    if not cpcb_file.exists():
        print("ERROR: CPCB file not found!")
        return False
    
    try:
        # Load the CSV file
        print("\nLoading CSV file...")
        df = pd.read_csv(cpcb_file)
        print(f"Initial data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Find PM2.5 column
        print("\nSearching for PM2.5 column...")
        pm25_col = None
        for col in ['PM2.5', 'PM2.5 (ug/m3)', 'pm2.5', 'PM25']:
            if col in df.columns:
                pm25_col = col
                print(f"Found PM2.5 column: '{pm25_col}'")
                break
        
        if pm25_col is None:
            print("ERROR: No PM2.5 column found!")
            return False
        
        # Check PM2.5 data
        print(f"PM2.5 data sample:")
        print(df[pm25_col].head())
        print(f"PM2.5 data type: {df[pm25_col].dtype}")
        
        # Clean PM2.5 data
        print("\nCleaning PM2.5 data...")
        df[pm25_col] = pd.to_numeric(df[pm25_col], errors='coerce')
        initial_count = len(df)
        df = df[df[pm25_col] >= 0]  # Remove negative values
        df = df.dropna(subset=[pm25_col])  # Remove NaN values
        print(f"After cleaning PM2.5: {len(df)} records (removed {initial_count - len(df)})")
        
        # Find timestamp column
        print("\nSearching for timestamp column...")
        timestamp_col = None
        if 'Date' in df.columns and 'Time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
            timestamp_col = 'timestamp'
            print("Created timestamp from Date + Time columns")
        else:
            for col in ['From Date', 'timestamp', 'datetime', 'Date']:
                if col in df.columns:
                    print(f"Found timestamp column: '{col}'")
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    timestamp_col = col
                    break
        
        if timestamp_col is None:
            print("ERROR: No timestamp column found!")
            return False
        
        # Check timestamp data
        print(f"Timestamp data sample:")
        print(df[timestamp_col].head())
        print(f"Timestamp data type: {df[timestamp_col].dtype}")
        
        # Remove rows with invalid timestamps
        initial_count = len(df)
        df = df.dropna(subset=[timestamp_col])
        print(f"After removing invalid timestamps: {len(df)} records (removed {initial_count - len(df)})")
        
        # Convert to IST and filter time window (08:00-16:00 local time)
        print("\nProcessing timestamps...")
        df[timestamp_col] = df[timestamp_col].dt.tz_localize(ist_tz, ambiguous='infer')
        df['hour'] = df[timestamp_col].dt.hour
        initial_count = len(df)
        df = df[(df['hour'] >= 8) & (df['hour'] <= 16)]
        print(f"After filtering time window (8-16h): {len(df)} records (removed {initial_count - len(df)})")
        df = df.drop('hour', axis=1)
        
        # Convert to UTC
        df['timestamp_utc'] = df[timestamp_col].dt.tz_convert(utc_tz)
        
        # Rename PM2.5 column to standard name
        df = df.rename(columns={pm25_col: 'PM2.5'})
        
        print(f"\nFinal cleaned data:")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")
        print(f"PM2.5 range: {df['PM2.5'].min():.2f} to {df['PM2.5'].max():.2f}")
        print(f"Sample data:")
        print(df[['timestamp_utc', 'PM2.5']].head())
        
        print("\n✅ CPCB data loading test SUCCESSFUL!")
        return True
        
    except Exception as e:
        print(f"ERROR: Exception during data loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cpcb_loading()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Tests failed!")
