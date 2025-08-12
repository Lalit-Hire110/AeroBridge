import os
from datetime import datetime, timedelta
import shutil

# CONFIGURABLE:
BASE_DIR = "data/cropped_data"
STATE = "Haryana"  # <-- Change if needed
STATION = "HR_001C6"  # <-- Change if needed
YEARS = [2021, 2022]
BANDS = ["TIR1", "WV"]

# Image naming format: 3RIMG_01NOV2021_0815_L1B_STD_V01R00_IMG_TIR1_cropped.tif
FILENAME_TEMPLATE = "3RIMG_{date}_L1B_STD_V01R00_IMG_{band}_cropped.tif"
DATETIME_FORMAT = "%d%b%Y_%H%M"

# Time range: 8:15 to 15:45 at 30-min intervals
def generate_expected_timestamps(year):
    start = datetime(year, 9, 1, 8, 15)
    end = datetime(year, 11, 30, 15, 45)
    timestamps = []
    current = start
    while current <= end:
        timestamps.append(current)
        current += timedelta(minutes=30)
    return timestamps

def build_filename(ts: datetime, band: str):
    date_str = ts.strftime(DATETIME_FORMAT).upper()
    return FILENAME_TEMPLATE.format(date=date_str, band=band)

def backfill_for_band(year, band):
    print(f"\n📂 Processing {year} - {band}...")
    dir_path = os.path.join(BASE_DIR, STATE, STATION, str(year), band)
    extra_path = os.path.join(dir_path, "extra")
    os.makedirs(extra_path, exist_ok=True)

    expected_ts = generate_expected_timestamps(year)
    files_present = {f: os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".tif")}

    total_expected = len(expected_ts)
    filled_count = 0
    missing_count = 0
    unfillable = []

    for idx, ts in enumerate(expected_ts):
        fname = build_filename(ts, band)
        fpath = os.path.join(dir_path, fname)

        if os.path.exists(fpath):
            continue  # Already exists

        # Need to backfill
        prev_idx = idx - 1
        while prev_idx >= 0:
            prev_ts = expected_ts[prev_idx]
            prev_fname = build_filename(prev_ts, band)
            prev_fpath = os.path.join(dir_path, prev_fname)
            if os.path.exists(prev_fpath):
                shutil.copy(prev_fpath, fpath)
                filled_count += 1
                break
            prev_idx -= 1
        else:
            unfillable.append(fname)
            missing_count += 1

    # Handle extra images
    expected_filenames = set(build_filename(ts, band) for ts in expected_ts)
    for fname in list(files_present.keys()):
        if fname not in expected_filenames:
            shutil.move(files_present[fname], os.path.join(extra_path, fname))

    print(f"✅ BACKFILL COMPLETE for {year} - {band}")
    print(f"📅 Total expected timestamps   : {total_expected}")
    print(f"🟢 Successfully filled (prev)  : {filled_count}")
    print(f"🔴 Still unfillable            : {missing_count}")
    print(f"📦 Extra images moved to /extra: {len(files_present) - (total_expected - missing_count)}")

    if unfillable:
        print(f"🧨 Unfillable timestamps:")
        for fname in unfillable:
            print("   -", fname)

# MAIN
for year in YEARS:
    for band in BANDS:
        backfill_for_band(year, band)