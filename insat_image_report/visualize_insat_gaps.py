import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# Load input CSVs
# -----------------------------
valid_df = pd.read_csv("valid_images.csv")
missing_df = pd.read_csv("missing_images.csv")
extra_df = pd.read_csv("extra_images.csv")

# Helper to extract just date
def get_date(ts):
    return datetime.strptime(ts, "%d%b%Y_%H%M").date()

# Build daily counts
counts = defaultdict(lambda: {"valid": 0, "missing": 0, "extra": 0})

for ts in valid_df["timestamp"]:
    counts[get_date(ts)]["valid"] += 1

for ts in missing_df["missing_timestamp"]:
    counts[get_date(ts)]["missing"] += 1

for ts in extra_df["extra_timestamp"]:
    counts[get_date(ts)]["extra"] += 1

# Convert to DataFrame
daily_stats = pd.DataFrame([
    {"date": date, "valid_count": c["valid"], "missing_count": c["missing"], "extra_count": c["extra"]}
    for date, c in sorted(counts.items())
])

# Save to CSV
daily_stats.to_csv("insat_daily_gap_report.csv", index=False)

# Plotting
plt.figure(figsize=(15, 6))
plt.bar(daily_stats["date"], daily_stats["valid_count"], label="Valid", color="green")
plt.bar(daily_stats["date"], daily_stats["missing_count"], bottom=daily_stats["valid_count"], label="Missing", color="red")
plt.bar(daily_stats["date"], daily_stats["extra_count"], label="Extra", color="orange", alpha=0.5)

plt.xlabel("Date")
plt.ylabel("Image Count")
plt.title("INSAT Image Status Per Day (Sep–Nov 2021)")
plt.legend()
plt.tight_layout()
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.3)
plt.savefig("insat_gap_plot.png")
plt.show()

print("✅ Visualization Complete! Check:")
print("📁 insat_daily_gap_report.csv")
print("🖼️ insat_gap_plot.png")
