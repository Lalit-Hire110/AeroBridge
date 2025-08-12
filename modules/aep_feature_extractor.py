import os
import glob
import pandas as pd
import rasterio
import numpy as np  # <-- FIX: Import numpy
from datetime import datetime, timedelta

# ------------------- CONFIG -------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROPPED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'cropped_data')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed_data')
FINAL_DATA_DIR = os.path.join(BASE_DIR, 'data', 'final_data')

os.makedirs(FINAL_DATA_DIR, exist_ok=True)

# -------- Helper: extract timestamp from filename --------
def extract_timestamp(filename):
    try:
        # Format: 3RIMG_01NOV2021_0815_L1B_STD_V01R00_IMG_TIR1_cropped.tif
        parts = filename.split('_')
        date_str = parts[1]
        time_str = parts[2]
        timestamp = datetime.strptime(date_str + time_str, "%d%b%Y%H%M")
        return timestamp
    except Exception as e:
        print(f"[ERROR] Failed to parse timestamp from {filename}: {e}")
        return None

# -------- Helper: extract image features --------
def extract_features_from_image(img_path):
    try:
        with rasterio.open(img_path) as src:
            band = src.read(1).astype(float)
            band[band == src.nodata] = float('nan')
            mean_val = float(np.nanmean(band))
            std_val = float(np.nanstd(band))
        return mean_val, std_val
    except Exception as e:
        print(f"[ERROR] Feature extraction failed for {img_path}: {e}")
        return None, None

# -------- Main pipeline --------
for state in os.listdir(CROPPED_DATA_DIR):
    state_path = os.path.join(CROPPED_DATA_DIR, state)
    if not os.path.isdir(state_path): continue

    for station in os.listdir(state_path):
        print(f"\n🛰️ Processing {state}/{station}...")
        station_path = os.path.join(state_path, station)

        all_features = []

        # Loop through years (2021, 2022)
        for year in os.listdir(station_path):
            year_folder = os.path.join(station_path, year)
            image_files = sorted(glob.glob(os.path.join(year_folder, "*.tif")))

            for img_path in image_files:
                fname = os.path.basename(img_path)
                timestamp = extract_timestamp(fname)

                if not timestamp: continue

                mean_val, std_val = extract_features_from_image(img_path)
                if mean_val is None: continue

                all_features.append({
                    "timestamp": timestamp,
                    "image_name": fname,
                    "feature_mean": mean_val,
                    "feature_std": std_val
                })

        feature_df = pd.DataFrame(all_features)
        if feature_df.empty:
            print(f"[SKIP] No features extracted for {state}/{station}")
            continue

        # Load processed PM2.5 data
        station_csv = os.path.join(PROCESSED_DATA_DIR, state, f"{station}.csv")
        if not os.path.exists(station_csv):
            print(f"[MISSING] Processed data not found for {state}/{station}")
            continue

        pm25_df = pd.read_csv(station_csv)
        pm25_df['timestamp'] = pd.to_datetime(pm25_df['timestamp'])

        # Merge based on exact timestamps
        merged = pd.merge(feature_df, pm25_df, on="timestamp", how="inner")
        if merged.empty:
            print(f"[WARN] No matching timestamps after merge for {state}/{station}")
            continue

        # Save merged data
        final_out_dir = os.path.join(FINAL_DATA_DIR, state)
        os.makedirs(final_out_dir, exist_ok=True)
        final_csv_path = os.path.join(final_out_dir, f"{station}.csv")
        merged.to_csv(final_csv_path, index=False)
        print(f"[✅ SAVED] {final_csv_path}")