import json
import sys
import pandas as pd

cfg = json.load(open("pipeline/config.json"))
D, T = cfg["date_column"], cfg["target_column"]
P, R = cfg["product_column"], cfg["region_column"]
H = cfg["horizon_weeks"]

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else "data/features.csv"

df = pd.read_csv(src, parse_dates=[D])

weather = None
if cfg.get("weather_file"):
    w = pd.read_csv(cfg["weather_file"], parse_dates=["date"], index_col="date")
    weather = w.resample("W").agg({"rain_mm": "sum", "temp_c": "mean"})

LAGS = [H, H + 1, H + 2, 52]
parts = []
for (product, region), g in df.groupby([P, R]):
    g = g.sort_values(D).set_index(D)
    f = pd.DataFrame(index=g.index)
    f[P] = product
    f[R] = region
    f["y"] = g[T]
    for lag in LAGS:
        f[f"units_lag{lag}"] = g[T].shift(lag)
    f[f"units_avg4_lag{H}"] = g[T].shift(H).rolling(4).mean()
    f["week_of_year"] = g.index.isocalendar().week.astype(int).values
    if weather is not None:
        wk = weather.reindex(g.index)
        for lag in [H, H + 1, H + 2]:
            f[f"rain_lag{lag}"] = wk["rain_mm"].shift(lag)
        f[f"temp_lag{H}"] = wk["temp_c"].shift(H)
    for flag in ["filled", "stockout_suspect", "outlier_check"]:
        f[flag] = g[flag]
    parts.append(f)

out = pd.concat(parts).dropna(subset=["y"])
out.index.name = D
out.to_csv(dst)

feature_cols = [c for c in out.columns if c not in [P, R, "y", "filled", "stockout_suspect", "outlier_check"]]
print("Rows:", len(out))
print("Horizon (weeks):", H)
print("Features:", feature_cols)
print("Saved:", dst)