import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("cleaned_dataset.csv")
df_grouped = df.groupby("Viti Akademik")["Numri i nxenesve"].sum().reset_index()
df_grouped["Viti Akademik"] = pd.to_datetime(df_grouped["Viti Akademik"], format="%Y")

df_grouped.set_index("Viti Akademik", inplace=True)

train = df_grouped.iloc[:-2]
test = df_grouped.iloc[-2:]

model = ARIMA(train, order=(2, 1, 1))
model_fit = model.fit()

pred_test = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, typ='levels')

forecast_steps = 5
forecast = model_fit.forecast(steps=forecast_steps)

forecast_index = pd.date_range(start=df_grouped.index[-1] + pd.DateOffset(years=1), periods=forecast_steps, freq='Y')

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