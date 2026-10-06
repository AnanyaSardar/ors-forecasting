import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/lightgbm_v2_forecast.csv", parse_dates=["date"], index_col="date")

plt.figure(figsize=(12, 5))
plt.plot(df.index, df["actual"], label="Actual demand", color="tab:blue")
plt.plot(df.index, df["baseline"], label="Baseline (last year)", color="tab:gray", linestyle="--")
plt.plot(df.index, df["lightgbm_v2"], label="LightGBM v2", color="tab:orange")
plt.ylabel("ORS units per week")
plt.title("Forecast vs actual, 2025")
plt.legend()
plt.tight_layout()
plt.savefig("charts/forecast_vs_actual.png", dpi=150)