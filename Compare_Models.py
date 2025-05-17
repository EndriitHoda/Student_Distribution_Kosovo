import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Apply a professional theme
sns.set(style="whitegrid")

# Load all result files
model_files = {
    "Random Forest": "AlgorithmResults/results_rf.json",
    "Neural Network": "AlgorithmResults/results_nn.json",
    "XGBoost": "AlgorithmResults/results_xgb.json",
    "CNN": "AlgorithmResults/results_cnn.json",
    "LSTM": "AlgorithmResults/results_lstm.json"
}

# Rescale MSE for readability
scaling_factors = {
    'R² Score': 1,
    'MAE': 1,
    'MSE': 1000,
    'RMSE': 1
}

metrics = ['R² Score', 'MAE', 'MSE (÷1000)', 'RMSE']
model_names = []
metric_matrix = []

# Load and prepare data
for model_name, path in model_files.items():
    if os.path.exists(path):
        with open(path, 'r') as f:
            results = json.load(f)
            model_names.append(model_name)
            metric_matrix.append([
                results['r2'],
                results['mae'],
                results['mse'] / scaling_factors['MSE'],
                results['rmse']
            ])
    else:
        print(f"Warning: File {path} not found. Skipping {model_name}.")

metric_matrix = np.array(metric_matrix)
x = np.arange(len(metrics))
width = 0.15

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
palette = sns.color_palette("Set2", len(model_names))

for i, (model, values) in enumerate(zip(model_names, metric_matrix)):
    bars = ax.bar(x + i * width - width * (len(model_names) - 1) / 2, values, width, label=model, color=palette[i])

    # Label values on each bar
    for j, val in enumerate(values):
        ax.text(x[j] + i * width - width * (len(model_names) - 1) / 2, val + 1, f"{val:.2f}",
                ha='center', va='bottom', fontsize=8)

# Final formatting
ax.set_ylabel("Metric Values")
ax.set_title("Comparison of Regression Model Performance")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
sns.despine()
plt.tight_layout()
plt.show()
