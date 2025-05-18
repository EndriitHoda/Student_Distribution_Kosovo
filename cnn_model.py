import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import json
import os

# Load and preprocess data
data = pd.read_csv('cleaned_dataset.csv')

# Convert age range to average
def range_to_avg(x):
    try:
        low, high = map(int, x.split('-'))
        return (low + high) / 2
    except:
        return np.nan

data["Mosha"] = data["Mosha"].apply(range_to_avg)
data = data.dropna()

# One-hot encode categorical features
categorical_cols = ["Komuna", "Niveli Akademik", "Gjinia"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(data[categorical_cols])

# Scale numeric features
numerical_cols = ["Viti Akademik", "Mosha"]
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(data[numerical_cols])

# Combine features
X = np.hstack([scaled_nums, encoded_cats])
y = data["Numri i nxenesve"].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# log transform
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1)  # Output is log-transformed prediction
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train model on log-transformed target
model.fit(X_train, y_train_log, epochs=100, batch_size=16, verbose=1, validation_split=0.2)

# Predict (log scale)
y_pred_log = model.predict(X_test).flatten()

# Invert log transform to get predictions in original scale
y_pred = np.expm1(y_pred_log)

# Evaluate using original targets and predictions
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R2: {r2:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")

# Save results
os.makedirs("AlgorithmResults", exist_ok=True)
results = {
    "r2": r2,
    "mae": mae,
    "mse": mse,
    "rmse": rmse
}
with open("AlgorithmResults/results_cnn.json", "w") as f:
    json.dump(results, f, indent=4)
