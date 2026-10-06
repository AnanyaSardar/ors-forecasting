import pandas as pd

df = pd.read_csv("data/sample_company_data.csv", parse_dates=["date"])

df = df.drop(index=range(100, 102))
df = df.drop(index=range(200, 212))
df.loc[df.index[150:152], "units"] = 0
df.loc[df.index[250], "units"] = -50
df.loc[df.index[300], "units"] = df.loc[df.index[300], "units"] * 12

dups = df.sample(40, random_state=7)
df = pd.concat([df, dups]).sort_values("date")
df.to_csv("data/messy_company_data.csv", index=False)
print("Rows:", len(df))