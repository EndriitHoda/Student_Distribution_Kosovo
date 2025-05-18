import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
import json

# Load and preprocess data
data = pd.read_csv('cleaned_dataset.csv')

def range_to_avg(x):
    try:
        low, high = map(int, x.split('-'))
        return (low + high) / 2
    except:
        return np.nan

data["Mosha"] = data["Mosha"].apply(range_to_avg)
data = data.dropna()

categorical_cols = ["Komuna", "Niveli Akademik", "Gjinia"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(data[categorical_cols])

numerical_cols = ["Viti Akademik", "Mosha"]
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(data[numerical_cols])

X = np.hstack([scaled_nums, encoded_cats])
y = data["Numri i nxenesve"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for LSTM input (samples, time steps, features)
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build LSTM model
model = Sequential([
    LSTM(64, input_shape=(1, X_train.shape[1]), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
model.fit(X_train_lstm, y_train, epochs=100, batch_size=16, verbose=1)

# Evaluate
loss, mae_metric = model.evaluate(X_test_lstm, y_test, verbose=0)
y_pred = model.predict(X_test_lstm).flatten()

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print results
print(f"RÂ² Score: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

# Save results
results = {
    "r2": r2,
    "mae": mae,
    "mse": mse,
    "rmse": rmse
}
with open("AlgorithmResults/results_lstm.json", "w") as f:
    json.dump(results, f, indent=4)