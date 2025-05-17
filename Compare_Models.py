import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Apply a professional theme
sns.set(style="whitegrid")

# Load results from all models
with open("AlgorithmResults/results_rf.json", "r") as f:
    results_rf = json.load(f)

with open("AlgorithmResults/results_nn.json", "r") as f:
    results_nn = json.load(f)

with open("AlgorithmResults/results_xgb.json", "r") as f:
    results_xgb = json.load(f)

# Rescale MSE to make the chart readable
scaling_factors = {
    'R² Score': 1,
    'MAE': 1,
    'MSE': 1000,
    'RMSE': 1
}

# Prepare data
metrics = ['R² Score', 'MAE', 'MSE (÷1000)', 'RMSE']
rf_values = [
    results_rf['r2'],
    results_rf['mae'],
    results_rf['mse'] / scaling_factors['MSE'],
    results_rf['rmse']
]
nn_values = [
    results_nn['r2'],
    results_nn['mae'],
    results_nn['mse'] / scaling_factors['MSE'],
    results_nn['rmse']
]
xgb_values = [
    results_xgb['r2'],
    results_xgb['mae'],
    results_xgb['mse'] / scaling_factors['MSE'],
    results_xgb['rmse']
]

# Plotting setup
x = np.arange(len(metrics))
width = 0.25  # Slightly narrower bars to fit 3 side-by-side

fig, ax = plt.subplots(figsize=(12, 6))

palette = sns.color_palette("Set2")
bars1 = ax.bar(x - width, rf_values, width, label='Random Forest', color=palette[0])
bars2 = ax.bar(x, nn_values, width, label='Neural Network', color=palette[1])
bars3 = ax.bar(x + width, xgb_values, width, label='XGBoost', color=palette[2])

# Axis labeling
ax.set_ylabel('Metric Values')
ax.set_title('Comparison of Regression Model Performance')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Add numerical value labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# Final formatting
plt.tight_layout()
sns.despine()
plt.show()
