import json
import pandas as pd
import streamlit as st

cfg = json.load(open("pipeline/config.json"))
D, T = cfg["date_column"], cfg["target_column"]
P, R = cfg["product_column"], cfg["region_column"]

st.set_page_config(page_title="Demand forecast", layout="wide")

clean = pd.read_csv("data/clean_data.csv", parse_dates=[D])
fc = pd.read_csv("data/forecast.csv", parse_dates=[D])
scores = pd.read_csv("data/backtest_scores.csv")

product = st.sidebar.selectbox("Product", sorted(fc[P].unique()))
region = st.sidebar.selectbox("Region", sorted(fc[fc[P] == product][R].unique()))

h = clean[(clean[P] == product) & (clean[R] == region)].set_index(D)
f = fc[(fc[P] == product) & (fc[R] == region)].set_index(D)
s = scores[(scores[P] == product) & (scores[R] == region)].iloc[0]

st.title(f"{product} demand forecast: {region}")
c1, c2, c3 = st.columns(3)
c1.metric("Model used", s["use"])
c2.metric("Baseline error", f"{s['baseline']:.1f}%")
c3.metric("Chosen model error", f"{s[s['use']]:.1f}%")
st.caption("Error = WAPE on the last 52 weeks. Lower is better.")

chart = pd.concat([h[T].tail(52).rename("history"), f["forecast"], f["stock_to_keep"]], axis=1)
st.line_chart(chart)

st.subheader("Next weeks: stock plan")
st.dataframe(f[["forecast", "stock_to_keep", "model"]])

st.subheader("Data checks")
st.write(
    f"Weeks filled: {int(h['filled'].sum())} | "
    f"Stock-out suspects: {int(h['stockout_suspect'].sum())} | "
    f"Outliers fixed: {int(h['outlier_check'].sum())} | "
    f"Still missing: {int(h[T].isna().sum())}"
)