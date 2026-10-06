import requests
import pandas as pd

LAT, LON = 18.52, 73.86  # Pune

url = "https://power.larc.nasa.gov/api/temporal/daily/point"
params = {
    "parameters": "PRECTOTCORR,T2M,RH2M",
    "community": "AG",
    "latitude": LAT,
    "longitude": LON,
    "start": "20190101",
    "end": "20251231",
    "format": "JSON",
}

response = requests.get(url, params=params, timeout=120)
response.raise_for_status()

data = response.json()["properties"]["parameter"]
df = pd.DataFrame(data)
df.index = pd.to_datetime(df.index, format="%Y%m%d")
df.index.name = "date"

df = df.rename(columns={
    "PRECTOTCORR": "rain_mm",
    "T2M": "temp_c",
    "RH2M": "humidity_pct",
})
df = df.replace(-999, pd.NA)

df.to_csv("data/weather_daily.csv")

print(df.head())
print("Rows:", len(df))
print(df.isna().sum())