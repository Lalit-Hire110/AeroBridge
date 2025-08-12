import os
import pandas as pd
from datetime import timedelta

# Paths
csv_path = r"C:\Users\Lalit Hire\OneDrive\Desktop\APE_07\data\processed_data\Delhi_DL009_merged.csv"
image_folder = r"C:\Users\Lalit Hire\OneDrive\Desktop\APE_07\data\cropped_data\Delhi\DL009\2021"

# Check file existence
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

if not os.path.exists(image_folder):
    raise FileNotFoundError(f"Image folder not found: {image_folder}")

# Load CPCB data
df = pd.read_csv(csv_path, parse_dates=["timestamp_utc"])

# Strip timezone from CPCB timestamps
df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize(None)

# List image files
image_files = [f for f in os.listdir(image_folder) if f.endswith(".tif")]

def extract_timestamp_from_filename(f):
    try:
        parts = f.split("_")
        date_str = parts[1]
        time_str = parts[2]
        return pd.to_datetime(date_str + time_str, format="%d%b%Y%H%M")
    except:
        return None

# Extract image timestamps
image_times = pd.Series([extract_timestamp_from_filename(f) for f in image_files]).dropna().sort_values()

print("📸 First few image times:", image_times.head(10).values)
print("📸 Last few image times:", image_times.tail(10).values)


# Match example CPCB timestamp with nearest image
sample_ts = df['timestamp_utc'].iloc[0]
nearest_image = image_times[(image_times >= sample_ts - timedelta(minutes=15)) & (image_times <= sample_ts + timedelta(minutes=15))]

print("📍 CPCB Time:", sample_ts)
print("🛰️ Matched Image Time(s):", nearest_image.values)

for f in image_files[:20]:
    ts = extract_timestamp_from_filename(f)
    print(f"{f} -> {ts}")


print(df['timestamp_utc'].min(), df['timestamp_utc'].max())
