import pandas as pd
from datetime import datetime, timedelta

# -----------------------------
# Load missing and extra CSVs
# -----------------------------
missing_df = pd.read_csv("missing_images.csv")   # has 'missing_timestamp' column
extra_df = pd.read_csv("extra_images.csv")       # has 'extra_timestamp' column

# Convert to datetime objects
def parse_timestamp(ts):
    return datetime.strptime(ts, "%d%b%Y_%H%M")

missing_times = [parse_timestamp(ts) for ts in missing_df["missing_timestamp"]]
extra_times = [parse_timestamp(ts) for ts in extra_df["extra_timestamp"]]

# -----------------------------
# Simulate fill
# -----------------------------
filled = []
unfilled = []

# Create a set to track which extra timestamps have been used (1:1 matching)
used_extra = set()

for miss_ts in missing_times:
    matched = False
    for extra_ts in extra_times:
        if extra_ts in used_extra:
            continue
        time_diff = abs((extra_ts - miss_ts).total_seconds()) / 60  # in minutes
        if time_diff <= 10:
            filled.append((miss_ts, extra_ts))
            used_extra.add(extra_ts)
            matched = True
            break
    if not matched:
        unfilled.append(miss_ts)

# -----------------------------
# Report
# -----------------------------
print("🧪 Simulated Fill Report:")
print(f"🔎 Missing timestamps total      : {len(missing_times)}")
print(f"✅ Can be filled using extras    : {len(filled)}")
print(f"❌ Still unfillable              : {len(unfilled)}")

# Optional: Save results to CSV
pd.DataFrame(filled, columns=["missing_timestamp", "filled_with_extra"]).to_csv("filled_simulated.csv", index=False)
pd.DataFrame(unfilled, columns=["unfilled_timestamp"]).to_csv("still_missing.csv", index=False)
