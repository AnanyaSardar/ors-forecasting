import json
import logging
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from prophet import Prophet

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

cfg = json.load(open("pipeline/config.json"))
D, T = cfg["date_column"], cfg["target_column"]
P, R = cfg["product_column"], cfg["region_column"]
H = cfg["horizon_weeks"]
BUF = cfg.get("safety_buffer_pct", 10)

clean = pd.read_csv("data/clean_data.csv", parse_dates=[D])
scores = pd.read_csv("data/backtest_scores.csv")

weather = None
if cfg.get("weather_file"):
    w = pd.read_csv(cfg["weather_file"], parse_dates=["date"], index_col="date")
    weather = w.resample("W").agg({"rain_mm": "sum", "temp_c": "mean"})

outs = []
for _, row in scores.iterrows():
    use = row["use"]
    if str(use).startswith("skip"):
        print("Skipped:", row["product"], row["region"], "-", use)
        continue

    g = clean[(clean[P] == row["product"]) & (clean[R] == row["region"])]
    y = g.sort_values(D).set_index(D)[T]
    future = pd.date_range(y.index[-1] + pd.Timedelta(weeks=1), periods=H, freq="W")
    idx = y.index.append(future)
    yx = y.reindex(idx)

    f = pd.DataFrame(index=idx)
    for lag in [H, H + 1, H + 2, 52]:
        f[f"units_lag{lag}"] = yx.shift(lag)
    f[f"units_avg4_lag{H}"] = yx.shift(H).rolling(4).mean()
    f["week_of_year"] = idx.isocalendar().week.astype(int).values
    if weather is not None:
        wk = weather.reindex(idx)
        for lag in [H, H + 1, H + 2]:
            f[f"rain_lag{lag}"] = wk["rain_mm"].shift(lag)
        f[f"temp_lag{H}"] = wk["temp_c"].shift(H)

    known = yx.notna()
    train, fut = f[known], f.loc[future]

    if use == "baseline":
        pred = fut["units_lag52"].values
    elif use == "naive":
        pred = fut[f"units_lag{H}"].values
    elif use == "lightgbm":
        m = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                          min_child_samples=10, random_state=42, verbose=-1)
        m.fit(train, yx[known])
        pred = m.predict(fut)
    else:
        regs = [c for c in f.columns if c.startswith(("rain_", "temp_"))]
        tp = train.dropna(subset=regs)
        frame = tp[regs].copy()
        frame.insert(0, "y", yx[tp.index].values)
        frame.insert(0, "ds", tp.index)
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        for c in regs:
            m.add_regressor(c)
        m.fit(frame)
        nxt = fut[regs].fillna(tp[regs].mean()).copy()
        nxt.insert(0, "ds", fut.index)
        pred = m.predict(nxt)["yhat"].values

    pred = np.clip(pred, 0, None)
    outs.append(pd.DataFrame({
        P: row["product"], R: row["region"], D: future,
        "forecast": np.round(pred).astype(int),
        "stock_to_keep": np.round(pred * (1 + BUF / 100)).astype(int),
        "model": use,
    }))

result = pd.concat(outs)
result.to_csv("data/forecast.csv", index=False)
print(result.to_string(index=False))
print("Saved: data/forecast.csv")