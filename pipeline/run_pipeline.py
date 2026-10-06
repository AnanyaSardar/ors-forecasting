import subprocess
import sys

src = sys.argv[1]
steps = [
    ["pipeline/validate_data.py", src],
    ["pipeline/clean_data.py", src, "data/clean_data.csv"],
    ["pipeline/make_features.py", "data/clean_data.csv", "data/features.csv"],
    ["pipeline/backtest.py", "data/features.csv"],
    ["pipeline/forecast.py"],
]

for step in steps:
    print("\n>>>", step[0])
    if subprocess.run([sys.executable] + step).returncode != 0:
        sys.exit("Pipeline stopped at " + step[0])

print("\nDone. Forecast: data/forecast.csv")