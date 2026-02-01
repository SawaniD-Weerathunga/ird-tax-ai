import pandas as pd
import ast
import re

CSV_PATH = "data/chunks/ird_chunks_preview.csv"

df = pd.read_csv(CSV_PATH)

print("Rows:", len(df))
print("Columns:", list(df.columns))

# 1) Check required columns
required = {"chunk_id","document","page_range","section","word_len","preview"}
missing = required - set(df.columns)
if missing:
    print("❌ Missing columns:", missing)
else:
    print("✅ All required columns exist")

# 2) Check page_range formatting
bad_ranges = 0
for x in df["page_range"].astype(str):
    try:
        r = ast.literal_eval(x)  # expects [start,end]
        if not (isinstance(r, list) and len(r) == 2 and r[0] <= r[1]):
            bad_ranges += 1
    except:
        bad_ranges += 1
print("Bad page_range rows:", bad_ranges)

# 3) Check repeated header noise inside preview
patterns = [
    r"GUIDE TO CORPORATE",
    r"INLAND REVENUE DEPARTMENT",
    r"YEAR OF ASSESSMENT",
]
noise_hits = {}
for p in patterns:
    noise_hits[p] = df["preview"].str.contains(p, case=False, na=False).sum()

print("\n--- Header/Footer noise hits in preview ---")
for k,v in noise_hits.items():
    print(f"{k}: {v}")

# 4) Check word_len distribution
print("\n--- word_len stats ---")
print(df["word_len"].describe())

# 5) Check duplicate previews (rough)
dup_count = df["preview"].duplicated().sum()
print("\nDuplicate preview rows:", dup_count)

print("\n✅ Done.")
