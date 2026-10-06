import json
import logging
import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from prophet import Prophet

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

cfg = json.load(open("pipeline/config.json"))
D = cfg["date_column"]
P, R = cfg["product_column"], cfg["region_column"]
H, TEST = cfg["horizon_weeks"], cfg["test_weeks"]
MIN_WEEKS = cfg.get("min_weeks", 104)

src = sys.argv[1] if len(sys.argv) > 1 else "data/features.csv"
df = pd.read_csv(src, parse_dates=[D])

SKIP = [D, P, R, "y", "filled", "stockout_suspect", "outlier_check"]
MODELS = ["baseline", "naive", "lightgbm", "prophet"]

def wape(actual, forecast):
    ok = ~np.isnan(actual) & ~np.isnan(forecast)
    return np.abs(actual[ok] - forecast[ok]).sum() / np.abs(actual[ok]).sum() * 100

rows, preds = [], []
for (product, region), g in df.groupby([P, R]):
    g = g.sort_values(D).reset_index(drop=True)
    if len(g) < MIN_WEEKS:
        rows.append({"product": product, "region": region, "use": "skip: too short"})
        continue

    train, test = g.iloc[:-TEST], g.iloc[-TEST:]
    test = test.dropna(subset=["units_lag52"])
    if test.empty:
        rows.append({"product": product, "region": region, "use": "skip: no test weeks"})
        continue

    X = [c for c in g.columns if c not in SKIP]
    out = pd.DataFrame({D: test[D].values, P: product, R: region, "actual": test["y"].values})
    out["baseline"] = test["units_lag52"].values
    out["naive"] = test[f"units_lag{H}"].values

    lgb = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                        min_child_samples=10, random_state=42, verbose=-1)
    lgb.fit(train[X], train["y"])
    out["lightgbm"] = lgb.predict(test[X])

    regs = [c for c in X if c.startswith(("rain_", "temp_"))]
    tp = train.dropna(subset=regs)
    prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    for c in regs:
        prophet.add_regressor(c)
    prophet.fit(tp.rename(columns={D: "ds"})[["ds", "y"] + regs])
    future = test.rename(columns={D: "ds"})[["ds"] + regs].fillna(tp[regs].mean())
    out["prophet"] = prophet.predict(future)["yhat"].values

    scores = {name: wape(out["actual"].values, out[name].values) for name in MODELS}
    best = min(scores, key=scores.get)
    use = best if scores[best] <= scores["baseline"] - 1 else "baseline"
    rows.append({"product": product, "region": region,
                 **{k: round(v, 1) for k, v in scores.items()}, "use": use})
    preds.append(out)

res = pd.DataFrame(rows)
print("WAPE % on the last", TEST, "weeks (lower is better)")
print(res.to_string(index=False))

res.to_csv("data/backtest_scores.csv", index=False)
if preds:
    pd.concat(preds).to_csv("data/backtest_predictions.csv", index=False)
print("Saved: data/backtest_scores.csv, data/backtest_predictions.csv")