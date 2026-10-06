import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

def wape(actual, forecast):
    return np.abs(actual - forecast).sum() / np.abs(actual).sum() * 100

table = pd.read_csv("data/model_table.csv", parse_dates=["date"], index_col="date")

df = pd.DataFrame(index=table.index)
df["y"] = table["ors_units"]
for lag in [4, 5, 6, 52]:
    df[f"units_lag{lag}"] = table["ors_units"].shift(lag)
for lag in [2, 3, 4, 5, 6]:
    df[f"rain_lag{lag}"] = table["rain_mm"].shift(lag)
df["temp_lag4"] = table["temp_c"].shift(4)
df["week_of_year"] = table.index.isocalendar().week.astype(int).values
df = df.dropna()

train, test = df.iloc[:-52], df.iloc[-52:]

model = LGBMRegressor(
    n_estimators=300, learning_rate=0.03, num_leaves=15,
    min_child_samples=10, random_state=42, verbose=-1,
)
model.fit(train.drop(columns="y"), train["y"])
pred = model.predict(test.drop(columns="y"))

baseline = table["ors_units"].shift(52).loc[test.index]

print(f"Baseline WAPE:    {wape(test['y'], baseline):.1f}%")
print(f"LightGBM v2 WAPE: {wape(test['y'], pred):.1f}%")

out = pd.DataFrame({"actual": test["y"], "baseline": baseline, "lightgbm_v2": pred})
out.to_csv("data/lightgbm_v2_forecast.csv")
