import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

weather = pd.read_csv("data/weather_daily.csv", parse_dates=["date"], index_col="date")
weekly = weather.resample("W").agg(
    {"rain_mm": "sum", "temp_c": "mean", "humidity_pct": "mean"}
)
weekly = weekly.iloc[1:-1]  # drop the partial first and last weeks

rain_lag = weekly["rain_mm"].shift(2).fillna(0)
trend = 1 + 0.0008 * np.arange(len(weekly))
heat = np.clip(weekly["temp_c"] - 25, 0, None)

demand = 400 * trend + 2.0 * rain_lag + 15 * heat
noise = rng.normal(0, 0.08, len(weekly))
weekly["ors_units"] = (demand * (1 + noise)).round().astype(int)

weekly[["ors_units"]].to_csv("data/ors_demand_weekly.csv")

print(weekly["ors_units"].describe())
print("Weeks:", len(weekly))