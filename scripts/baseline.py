import numpy as np
import pandas as pd

table = pd.read_csv("data/model_table.csv", parse_dates=["date"], index_col="date")

def wape(actual, forecast):
    return np.abs(actual - forecast).sum() / np.abs(actual).sum() * 100

TEST_WEEKS = 52
test = table["ors_units"].iloc[-TEST_WEEKS:]
baseline = table["ors_units"].shift(52).iloc[-TEST_WEEKS:]

print("Test period:", test.index[0].date(), "to", test.index[-1].date())
print(f"Baseline WAPE: {wape(test, baseline):.1f}%")