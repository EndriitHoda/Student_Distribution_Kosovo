import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import json

# Step 1: Load the data from CSV
data = pd.read_csv('cleaned_dataset.csv')

# Step 2: Convert 'Mosha' from a range to an average number
def range_to_avg(x):
    try:
        low, high = map(int, x.split('-'))
        return (low + high) / 2
    except:
        return np.nan

data["Mosha"] = data["Mosha"].apply(range_to_avg)

# Drop rows with missing or invalid values
data = data.dropna()

# Step 3: One-hot encode categorical variables
categorical_cols = ["Komuna", "Niveli Akademik", "Gjinia"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(data[categorical_cols])

# Step 4: Normalize numerical features
numerical_cols = ["Viti Akademik", "Mosha"]
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(data[numerical_cols])

# Step 5: Combine features
X = np.hstack([scaled_nums, encoded_cats])
y = data["Numri i nxenesve"].values

# Analyze target variable scale
print(f"Mean target value: {y.mean():.2f}")
print(f"Median target value: {np.median(y):.2f}")

# Step 6: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Build the FNN model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # regression output
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Step 8: Train the model
model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

# Step 9: Evaluate the model
loss, mae_metric = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Mean Absolute Error (MAE) from evaluate(): {mae_metric:.2f}")

# Predict on test set
y_pred = model.predict(X_test).flatten()

# Calculate additional metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

results_nn = {
    "model": "NeuralNetwork",
    "r2": r2,
    "mae": mae,
    "mse": mse,
    "rmse": rmse
}

with open("AlgorithmResults/results_nn.json", "w") as f:
    json.dump(results_nn, f)

