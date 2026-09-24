# app_resolution_starter.py — Week 4: the resolution lesson as a widget
# Run with:  streamlit run app_resolution_starter.py
import streamlit as st
import pandas as pd
import statsmodels.api as sm

st.title("Resolution Changes the Story")

@st.cache_data
def load_series():
    co2 = sm.datasets.co2.load_pandas().data.dropna()
    return co2["co2"]

series = load_series()

# TODO: add a selectbox over resample rules ["W", "MS", "QS", "YS"]
# and resample `series` with the chosen rule before plotting:
rule = "MS"                              # <- replace with the selectbox
shown = series.resample(rule).mean()

window = st.slider("Rolling window (periods)", 1, 36, 12)

st.line_chart(pd.DataFrame({
    "raw": shown,
    f"rolling({window})": shown.rolling(window).mean(),
}))
st.caption(f"Resolution: {rule} · window: {window} — both choices belong in your caption.")
