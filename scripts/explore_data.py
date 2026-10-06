import pandas as pd
import matplotlib.pyplot as plt

weather = pd.read_csv("data/weather_daily.csv", parse_dates=["date"], index_col="date")
weekly_weather = weather.resample("W").agg(
    {"rain_mm": "sum", "temp_c": "mean", "humidity_pct": "mean"}
)

demand = pd.read_csv("data/ors_demand_weekly.csv", parse_dates=["date"], index_col="date")

table = demand.join(weekly_weather, how="left")
table.to_csv("data/model_table.csv")

print(table.head())
print(table.isna().sum())

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(table.index, table["ors_units"], color="tab:blue")
ax1.set_ylabel("ORS units per week")
ax2 = ax1.twinx()
ax2.bar(table.index, table["rain_mm"], width=5, color="tab:gray", alpha=0.4)
ax2.set_ylabel("Rain per week (mm)")
plt.title("Weekly ORS demand vs rain (Pune)")
plt.tight_layout()
plt.savefig("charts/demand_vs_rain.png", dpi=150)