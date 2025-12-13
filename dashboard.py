# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

st.title("Forex Trading Bot Dashboard")
st.write("Predicting Forex prices using LSTM!")

# Automatically load CSV from the folder
df = pd.read_csv("forex_data.csv")  # make sure this file is in the same folder
st.write("Data Preview:")
st.dataframe(df.head())  # show first 5 rows

# Select columns
features = df[['BO','BH','BL','BC','AO','AH','AL','AC']].values

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(features)

# Create sequences
look_back = 60
X, y = [], []
for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i - look_back:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

st.write("Data ready for LSTM")
st.write("X shape:", X.shape)
st.write("y shape:", y.shape)

# Build LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(50))
model.add(Dense(8))
model.compile(optimizer='adam', loss='mean_squared_error')

st.write("Training model... please wait a few seconds")
model.fit(X, y, epochs=5, batch_size=32, verbose=0)
st.success("Model trained!")

# Predict
predicted = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(y)

# Plot BC (Bid Close)
st.write("Prediction vs Actual (BC)")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(actual_prices[:,3], color='blue', label='Actual BC')
ax.plot(predicted_prices[:,3], color='red', label='Predicted BC')
ax.set_title('Forex BC Price Prediction')
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)
