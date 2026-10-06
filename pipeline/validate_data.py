import json
import sys
import pandas as pd

cfg = json.load(open("pipeline/config.json"))
D, T = cfg["date_column"], cfg["target_column"]
KEYS = [cfg["product_column"], cfg["region_column"]]

df = pd.read_csv(sys.argv[1])

missing = [c for c in [D, T] + KEYS if c not in df.columns]
if missing:
    sys.exit(f"STOP: these columns are missing: {missing}")

df[D] = pd.to_datetime(df[D], errors="coerce")
print("Rows:", len(df))
print("Unreadable dates:", df[D].isna().sum())
print("Missing values in target:", df[T].isna().sum())
print("Negative values:", (df[T] < 0).sum())
print("Duplicate rows:", df.duplicated(subset=[D] + KEYS).sum())

rows = []
for (product, region), g in df.dropna(subset=[D]).groupby(KEYS):
    span = (g[D].max() - g[D].min()).days // 7 + 1
    rows.append({
        "product": product,
        "region": region,
        "weeks_of_data": g[D].nunique(),
        "missing_weeks": span - g[D].nunique(),
        "zero_share_%": round((g[T] == 0).mean() * 100),
        "status": "OK" if g[D].nunique() >= 104 else "Too short",
    })

print(pd.DataFrame(rows).to_string(index=False))