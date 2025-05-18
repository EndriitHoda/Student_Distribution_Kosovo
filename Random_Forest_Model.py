import pandas as pd
import warnings
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

warnings.filterwarnings('ignore')

df = pd.read_csv('cleaned_dataset.csv')

label_encoders = {}
for col in ['Komuna', 'Niveli Akademik', 'Mosha', 'Gjinia']:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le


def augment_data(df, n=2):
    df_aug = df.copy()
    for _ in range(n):
        aug = df.copy()
        aug['Viti Akademik'] += np.random.randint(-2, 3, size=len(df))
        noise = np.random.normal(0, 10, size=len(df))  # small noise
        aug['Numri i nxenesve'] = (aug['Numri i nxenesve'] + noise).clip(lower=0)
        df_aug = pd.concat([df_aug, aug], ignore_index=True)
    return df_aug


df_aug = augment_data(df, n=2)

X = df_aug[['Komuna_encoded', 'Viti Akademik', 'Niveli Akademik_encoded', 'Mosha_encoded', 'Gjinia_encoded']]
y = df_aug['Numri i nxenesve']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 100 root trees with depth 25 to cover all dataset with ~2500 rows
model = RandomForestRegressor(n_estimators=100, max_depth=25, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nRegression Model Evaluation on Test Set:")
print(f"R² Score (Accuracy-like): {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

results_rf = {
    "model": "RandomForest",
    "r2": r2,
    "mae": mae,
    "mse": mse,
    "rmse": rmse
}

train_sizes, train_scores, valid_scores = learning_curve(
    model, X_train, y_train,
    cv=5,
    scoring='r2',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
    random_state=42
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

plt.figure(figsize=(8,6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training score')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='blue')

plt.plot(train_sizes, valid_scores_mean, 'o-', color='green', label='Cross-validation score')
plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color='green')

plt.title('Learning Curve for RandomForestRegressor')
plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

with open("AlgorithmResults/results_rf.json", "w") as f:
    json.dump(results_rf, f)

cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("\nCross-Validated R² Scores:", np.round(cv_scores, 3))
print(f"Average R² (CV): {np.mean(cv_scores):.2f}")

def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(8,5))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='r', linestyles='dashed')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title(f'Residual Plot for {model_name}')
    plt.grid(True)
    plt.show()

plot_residuals(y_test, y_pred, 'Random Forest')
# Similarly for XGBoost, replace y_pred with xgb predictions


def safe_encode(encoder, value, column_name):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        print(f"\nError: '{value}' is not a valid value for {column_name}.")
        print(f"Valid options are: {list(encoder.classes_)}")
        exit()


if __name__ == "__main__":
    print("\nStudent Number Predictor (Regression Mode)\n")

    komuna = input("Enter Komuna (e.g., Prishtine): ").strip()
    year = input("Enter Viti Akademik (e.g., 2030): ").strip()
    niveli = input("Enter Niveli Akademik (e.g., Para-fillor): ").strip()
    mosha = input("Enter Mosha (e.g., 5-6): ").strip()
    gjinia = input("Enter Gjinia (e.g., Mashkull): ").strip()

    try:
        year = int(year)
    except ValueError:
        print("Viti Akademik must be a valid number like 2030.")
        exit()

    komuna_enc = safe_encode(label_encoders['Komuna'], komuna, "Komuna")
    niveli_enc = safe_encode(label_encoders['Niveli Akademik'], niveli, "Niveli Akademik")
    mosha_enc = safe_encode(label_encoders['Mosha'], mosha, "Mosha")
    gjinia_enc = safe_encode(label_encoders['Gjinia'], gjinia, "Gjinia")

    input_features = [[komuna_enc, year, niveli_enc, mosha_enc, gjinia_enc]]
    prediction = model.predict(input_features)[0]

    print(f"\nEstimated number of students in {komuna} for {year}: {int(prediction)}")
