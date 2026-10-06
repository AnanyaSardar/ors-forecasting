import json
import sys
import pandas as pd

cfg = json.load(open("pipeline/config.json"))
D, T = cfg["date_column"], cfg["target_column"]
P, R = cfg["product_column"], cfg["region_column"]
MAX_GAP = 2

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else "data/clean_data.csv"

df = pd.read_csv(src)
df[D] = pd.to_datetime(df[D], errors="coerce")
df[T] = pd.to_numeric(df[T], errors="coerce").astype(float)

bad_dates = int(df[D].isna().sum())
df = df.dropna(subset=[D])
before = len(df)
df = df.drop_duplicates(subset=[D, P, R], keep="last")
dups = before - len(df)
negatives = int((df[T] < 0).sum())
df[T] = df[T].mask(df[T] < 0)

parts, rows = [], []
for (product, region), g in df.groupby([P, R]):
    s = g.set_index(D)[T].resample("W").sum(min_count=1)

    typical = s[s > 0].median()
    stockout = (s == 0) & (typical > 10)
    s = s.mask(stockout)

    gap = s.isna()
    run_id = gap.ne(gap.shift(fill_value=False)).cumsum()
    run_len = gap.groupby(run_id).transform("sum")
    fillable = gap & (run_len <= MAX_GAP)
    filled = s.where(~fillable, s.interpolate(limit_area="inside"))
    was_filled = fillable & filled.notna()

    rolling = filled.rolling(13, center=True, min_periods=5).median()
    outlier = filled > 4 * rolling
    filled = filled.mask(outlier, rolling)

    parts.append(pd.DataFrame({
        D: s.index, P: product, R: region, T: filled.values,
        "filled": was_filled.values,
        "stockout_suspect": stockout.values,
        "outlier_check": outlier.values,
    }))
    rows.append({
        "product": product, "region": region, "weeks": len(s),
        "filled": int(was_filled.sum()),
        "still_missing": int(filled.isna().sum()),
        "stockout_suspects": int(stockout.sum()),
        "outlier_checks": int(outlier.sum()),
    })

print("Unreadable dates removed:", bad_dates)
print("Duplicate rows removed:", dups)
print("Negative values set to missing:", negatives)
print(pd.DataFrame(rows).to_string(index=False))

pd.concat(parts).to_csv(dst, index=False)
print("Saved:", dst)