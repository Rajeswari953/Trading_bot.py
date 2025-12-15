import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Forex Trading Dashboard",
    layout="wide"
)

# ---------------- LIGHT GREEN STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f4fff7, #e8f8ee);
    color: #1b4332;
    font-family: Arial, sans-serif;
}

.card {
    min-height: 100px;
    padding: 16px;
    text-align: center;
    border-radius: 14px;
    background-color: #ffffff;
    border: 1px solid #cdebd7;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
}

h1 {
    text-align: center;
    color: #1b4332;
}

h3 {
    color: #2d6a4f;
    font-size: 15px;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LIGHT THEME FOR CHARTS ----------------
def light_theme(fig):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#1b4332",
        title_font_color="#1b4332"
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e6f2ea")
    fig.update_yaxes(showgrid=True, gridcolor="#e6f2ea")
    return fig

# ---------------- TITLE ----------------
st.markdown("<h1>Forex Trading Dashboard</h1>", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("forex_data.csv")

cols = ['BO','BH','BL','BC','AO','AH','AL','AC']
data = df[cols].values

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

pred = np.roll(data, -1, axis=0)

# ---------------- BUSINESS METRICS ----------------
last_price = data[-1, 3]
prev_price = data[-2, 3]
trend = "UP" if last_price > prev_price else "DOWN"

pnl = np.sum(data[-30:,3] - data[-31:-1,3])

trade_diff = data[1:,3] - data[:-1,3]
win_rate = np.sum(trade_diff > 0) / len(trade_diff) * 100

cumulative = np.cumsum(trade_diff)
max_drawdown = np.min(cumulative)

# ---------------- KPI ROW ----------------
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"<div class='card'><h3>Current Price</h3><b>{last_price:.4f}</b></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='card'><h3>Trend</h3><b>{trend}</b></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='card'><h3>Total P/L (30 days)</h3><b>{pnl:.4f}</b></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='card'><h3>Win Rate</h3><b>{win_rate:.2f}%</b></div>", unsafe_allow_html=True)
with k5:
    st.markdown(f"<div class='card'><h3>Max Drawdown</h3><b>{max_drawdown:.4f}</b></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- CHART DATA ----------------
N = 60
price = data[-N:,3]
pred_price = pred[-N:,3]

buy = np.where(pred_price > price, price, np.nan)
sell = np.where(pred_price < price, price, np.nan)

profit_loss = price[1:] - price[:-1]

# ---------------- PRICE & PREDICTION ----------------
c1, c2 = st.columns([2,3])

with c1:
    df_price = pd.DataFrame({
        "Price": price,
        "Predicted": pred_price
    })
    fig = px.line(
        df_price,
        y=["Price","Predicted"],
        title="Price vs Prediction",
        color_discrete_map={
            "Price":"#2d6a4f",
            "Predicted":"#95d5b2"
        }
    )
    fig = light_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- BUY / SELL ----------------
with c2:
    df_signal = pd.DataFrame({
        "Time": range(N),
        "Price": price,
        "Buy": buy,
        "Sell": sell
    })

    fig = px.line(df_signal, x="Time", y="Price", title="Buy / Sell Signals")
    fig.add_scatter(
        x=df_signal["Time"], y=df_signal["Buy"],
        mode="markers", marker=dict(color="#2d6a4f", size=10, symbol="triangle-up"),
        name="Buy"
    )
    fig.add_scatter(
        x=df_signal["Time"], y=df_signal["Sell"],
        mode="markers", marker=dict(color="#d00000", size=10, symbol="triangle-down"),
        name="Sell"
    )
    fig = light_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- BOTTOM CHARTS ----------------
c3, c4 = st.columns(2)

with c3:
    df_pl = pd.DataFrame({
        "Trade": range(len(profit_loss)),
        "Profit/Loss": profit_loss
    })
    fig = px.bar(
        df_pl,
        x="Trade",
        y="Profit/Loss",
        title="Profit / Loss per Trade",
        color_discrete_sequence=["#40916c"]
    )
    fig = light_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

with c4:
    wins = np.sum(trade_diff > 0)
    losses = np.sum(trade_diff <= 0)
    fig = px.pie(
        names=["Wins","Losses"],
        values=[wins, losses],
        title="Win vs Loss Distribution",
        color_discrete_map={
            "Wins":"#2d6a4f",
            "Losses":"#d00000"
        }
    )
    fig = light_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- BEST PAIRS ----------------
if "Pair" in df.columns:
    pair_profit = df.groupby("Pair")["BC"].apply(lambda x: x.diff().sum()).sort_values(ascending=False)
    fig = px.bar(
        x=pair_profit.index,
        y=pair_profit.values,
        title="Best Performing Currency Pairs",
        labels={"x":"Currency Pair","y":"Total Profit"},
        color_discrete_sequence=["#74c69d"]
    )
    fig = light_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
