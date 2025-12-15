# trading_bot.py
# Forex Trading Dashboard with Future Prediction (No TensorFlow)

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# -----------------------------
# STEP 1: PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Forex Trading Dashboard", layout="wide")

st.title("📊 Forex Trading Dashboard")
st.write("Simple & Professional Forex Analysis with Future Prediction")

# -----------------------------
# STEP 2: LOAD DATA
# -----------------------------
df = pd.read_csv("forex_data.csv")

# If Date column exists
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
else:
    df['Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')

# -----------------------------
# STEP 3: SIMPLE FUTURE PREDICTION
# -----------------------------
window = 5  # last 5 days average
df['Prediction'] = df['BC'].rolling(window=window).mean()

# -----------------------------
# STEP 4: KPIs
# -----------------------------
current_price = df['BC'].iloc[-1]
previous_price = df['BC'].iloc[-2]
trend = "UP 📈" if current_price > previous_price else "DOWN 📉"

col1, col2, col3 = st.columns(3)
col1.metric("Current Price (BC)", round(current_price, 2))
col2.metric("Market Trend", trend)
col3.metric("Prediction (Avg)", round(df['Prediction'].iloc[-1], 2))

# -----------------------------
# STEP 5: PRICE VS PREDICTION CHART
# -----------------------------
fig = px.line(
    df,
    x='Date',
    y=['BC', 'Prediction'],
    title="BC Price vs Future Prediction"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# STEP 6: SIMPLE BUY / SELL LOGIC
# -----------------------------
df['Signal'] = np.where(df['BC'] > df['Prediction'], 'BUY', 'SELL')

signal_counts = df['Signal'].value_counts().reset_index()
signal_counts.columns = ['Signal', 'Count']

fig2 = px.pie(
    signal_counts,
    names='Signal',
    values='Count',
    title="Buy vs Sell Signals"
)

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# END
# -----------------------------
st.success("Dashboard loaded successfully 🚀")
