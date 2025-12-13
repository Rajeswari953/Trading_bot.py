# trading-bot.py
# Full Forex Trading Bot Project

# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Step 2: Load your CSV
df = pd.read_csv("forex_data.csv")  # Make sure your CSV file is in the same folder

# Step 3: Select important columns
features = df[['BO','BH','BL','BC','AO','AH','AL','AC']].values

# Step 4: Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(features)

# Step 5: Create sequences (look back 60 steps)
look_back = 60
X = []
y = []

for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i - look_back:i])   # 60 previous rows
    y.append(scaled_data[i])                 # next row

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Step 6: Build LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(50))
model.add(Dense(8))  # Predict 8 price columns

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 7: Train the model
history = model.fit(X, y, epochs=10, batch_size=32)

# Step 8: Predict future prices
predicted = model.predict(X)

# Inverse scale to get actual price values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(y)

# Step 9: Plot results for BC (Bid Close) column
plt.figure(figsize=(14,5))
plt.plot(actual_prices[:, 3], color='blue', label='Actual BC Price')
plt.plot(predicted_prices[:, 3], color='red', label='Predicted BC Price')
plt.title('Forex BC Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
