import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from statsmodels.tsa.stattools import adfuller

result = adfuller(train)
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

df = pd.read_csv("cleaned_dataset.csv")
df_grouped = df.groupby("Viti Akademik")["Numri i nxenesve"].sum().reset_index()
df_grouped["Viti Akademik"] = pd.to_datetime(df_grouped["Viti Akademik"], format="%Y")

df_grouped.set_index("Viti Akademik", inplace=True)

t# Use last N points for test set
test_size = 4
train = df_grouped.iloc[:-test_size]
test = df_grouped.iloc[-test_size:]

# Perform walk-forward validation
history = list(train["Numri i nxenesve"])
predictions = []

for actual in test["Numri i nxenesve"]:
    model = ARIMA(history, order=(2, 1, 1))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    history.append(actual)

# Convert predictions to DataFrame with same index as test
pred_series = pd.Series(predictions, index=test.index)

# Evaluation
mae = mean_absolute_error(test, pred_series)
rmse = np.sqrt(mean_squared_error(test, pred_series))
r2 = r2_score(test, pred_series)

plt.figure(figsize=(12, 7))
plt.plot(train, label='Train Data', marker='o')
plt.plot(test, label='Test Data', marker='o', color='orange')
plt.plot(pred_test, label='Test Prediction', linestyle='--', marker='x', color='green')
plt.plot(forecast_index, forecast, label='Forecast (Next 5 Years)', linestyle='--', marker='o', color='red')

plt.title("Kosovo-Wide Student Enrollment Forecast with ARIMA", fontsize=14)
plt.xlabel("Academic Year")
plt.ylabel("Total Number of Students")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

mae = mean_absolute_error(test, pred_test)
rmse = np.sqrt(mean_squared_error(test, pred_test))
r2 = r2_score(test, pred_test)

print("ARIMA Model Evaluation on Test Data (2022–2023):")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.3f}")