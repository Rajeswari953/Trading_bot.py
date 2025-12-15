# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Forex Trading Dashboard", layout="wide")

st.title("📊 Forex Trading Dashboard")
st.write("Forex analysis with future trend prediction (no heavy ML)")

# LOAD DATA
df = pd.read_csv("forex_data.csv")

# SAFE DATE HANDLING
df['Date'] = np.arange(len(df))  # safe numeric index instead of calendar dates

# FUTURE PREDICTION (MOVING AVERAGE)
window = 5
df['Prediction'] = df['BC'].rolling(window=window).mean()

# KPIs
current_price = df['BC'].iloc[-1]
previous_price = df['BC'].iloc[-2]
trend = "UP 📈" if current_price > previous_price else "DOWN 📉"

col1, col2, col3 = st.columns(3)
col1.metric("Current BC Price", round(current_price, 2))
col2.metric("Market Trend", trend)
col3.metric("Predicted Avg", round(df['Prediction'].iloc[-1], 2))

# PRICE VS PREDICTION CHART
fig = px.line(
    df,
    x='Date',
    y=['BC', 'Prediction'],
    title="BC Price vs Future Prediction"
)
st.plotly_chart(fig, use_container_width=True)

# BUY / SELL SIGNALS
df['Signal'] = np.where(df['BC'] > df['Prediction'], 'BUY', 'SELL')
signal_count = df['Signal'].value_counts().reset_index()
signal_count.columns = ['Signal', 'Count']

fig2 = px.pie(
    signal_count,
    names='Signal',
    values='Count',
    title="Buy vs Sell Signals"
)
st.plotly_chart(fig2, use_container_width=True)

st.success("Dashboard loaded successfully 🚀")
