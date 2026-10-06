import numpy as np
import pandas as pd

def wape(actual, forecast):
    return np.abs(actual - forecast).sum() / np.abs(actual).sum() * 100

lgbm = pd.read_csv("data/lightgbm_v2_forecast.csv", parse_dates=["date"], index_col="date")
prophet = pd.read_csv("data/prophet_forecast.csv", parse_dates=["date"], index_col="date")

df = lgbm.join(prophet[["prophet"]])
df["blend"] = (df["lightgbm_v2"] + df["prophet"]) / 2

for name in ["baseline", "lightgbm_v2", "prophet", "blend"]:
    print(f"{name:12s} WAPE: {wape(df['actual'], df[name]):.1f}%")

df.to_csv("data/final_forecast.csv")