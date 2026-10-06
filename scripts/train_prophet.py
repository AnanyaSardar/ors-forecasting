import logging
import numpy as np
import pandas as pd
from prophet import Prophet

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

def wape(actual, forecast):
    return np.abs(actual - forecast).sum() / np.abs(actual).sum() * 100

table = pd.read_csv("data/model_table.csv", parse_dates=["date"], index_col="date")

df = pd.DataFrame({"ds": table.index, "y": table["ors_units"].values})
regressors = ["rain_lag2", "rain_lag3", "rain_lag4", "temp_lag4"]
for lag in [2, 3, 4]:
    df[f"rain_lag{lag}"] = table["rain_mm"].shift(lag).values
df["temp_lag4"] = table["temp_c"].shift(4).values
df = df.dropna().reset_index(drop=True)

train, test = df.iloc[:-52], df.iloc[-52:]

model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
for col in regressors:
    model.add_regressor(col)
model.fit(train)
pred = model.predict(test.drop(columns="y"))["yhat"].values

baseline = table["ors_units"].shift(52).loc[test["ds"]].values
lgbm = pd.read_csv("data/lightgbm_v2_forecast.csv")["lightgbm_v2"].values

print(f"Baseline WAPE:    {wape(test['y'].values, baseline):.1f}%")
print(f"Prophet WAPE:     {wape(test['y'].values, pred):.1f}%")
print(f"LightGBM v2 WAPE: {wape(test['y'].values, lgbm):.1f}%")

out = pd.DataFrame({"date": test["ds"].values, "actual": test["y"].values, "prophet": pred})
out.to_csv("data/prophet_forecast.csv", index=False)