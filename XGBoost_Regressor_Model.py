import pandas as pd
import numpy as np
import warnings
from xgboost import XGBRegressor, plot_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
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
        aug['Viti Akademik'] += np.random.randint(-2, 3, len(df))
        aug['Numri i nxenesve'] += np.random.normal(0, 10, len(df))
        aug['Numri i nxenesve'] = aug['Numri i nxenesve'].clip(lower=0)
        df_aug = pd.concat([df_aug, aug], ignore_index=True)
    return df_aug


df_aug = augment_data(df, n=2)

X = df_aug[['Komuna_encoded', 'Viti Akademik', 'Niveli Akademik_encoded', 'Mosha_encoded', 'Gjinia_encoded']]
y = df_aug['Numri i nxenesve']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb = XGBRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    verbose=1,
    n_jobs=-1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

final_model = XGBRegressor(**grid_search.best_params_, random_state=42)
final_model.fit(X_train, y_train)

# Plot R² over boosting rounds
train_r2_scores = []
test_r2_scores = []
cv_r2_scores = []

n_rounds = 200  # You can adjust based on how long you want to track

temp_model = XGBRegressor(
    n_estimators=1,  # we'll manually increase this
    max_depth=grid_search.best_params_['max_depth'],
    learning_rate=grid_search.best_params_['learning_rate'],
    subsample=grid_search.best_params_['subsample'],
    colsample_bytree=grid_search.best_params_['colsample_bytree'],
    random_state=42
)

# Choose steps to plot up to the best number of estimators
step_size = max(1, grid_search.best_params_['n_estimators'] // n_rounds)

for i in range(step_size, grid_search.best_params_['n_estimators'] + 1, step_size):
    model = XGBRegressor(
        **{**grid_search.best_params_, 'n_estimators': i},
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_r2_scores.append(train_r2)

    cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
    cv_r2_scores.append(cv_r2)

# Plot
plt.figure(figsize=(10, 6))
rounds = list(range(step_size, grid_search.best_params_['n_estimators'] + 1, step_size))
plt.plot(rounds, train_r2_scores, label='Train R²', color='blue', linewidth=2)
plt.plot(rounds, cv_r2_scores, label='CV R² (5-fold)', color='orange', linewidth=2)

plt.fill_between(
    rounds,
    train_r2_scores,
    cv_r2_scores,
    where=(np.array(train_r2_scores) > np.array(cv_r2_scores)),
    interpolate=True,
    color='red',
    alpha=0.1,
    label='Overfitting Gap'
)

plt.xlabel('Number of Estimators')
plt.ylabel('R² Score')
plt.title('XGBoost Learning Curve (Train vs CV R²)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# best_model = grid_search.best_estimator_

y_pred = final_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nBest Hyperparameters Found:")
print(grid_search.best_params_)

print("\nTuned XGBoost Model Evaluation on Test Set:")
print(f"🔹 R² Score: {r2:.2f}")
print(f"🔹 Mean Absolute Error: {mae:.2f}")
print(f"🔹 Mean Squared Error: {mse:.2f}")
print(f"🔹 Root Mean Squared Error: {rmse:.2f}")

results_xgb = {
    "model": "XGBoost",
    "r2": r2,
    "mae": mae,
    "mse": mse,
    "rmse": rmse
}

with open("AlgorithmResults/results_xgb.json", "w") as f:
    json.dump(results_xgb, f)

cv_scores = cross_val_score(final_model, X, y, cv=5, scoring='r2')
print("\nCross-Validated R² Scores:", np.round(cv_scores, 3))
print(f"Average R² (CV): {np.mean(cv_scores):.2f}")

plt.figure(figsize=(10, 6))
plot_importance(final_model, importance_type='gain', max_num_features=10, title="Feature Importance")
plt.tight_layout()
plt.show()

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

plot_residuals(y_test, y_pred, 'XGBoost')

def safe_encode(encoder, value, column_name):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        print(f"\nError: '{value}' is not valid for {column_name}.")
        print(f"Valid options: {list(encoder.classes_)}")
        exit()


if __name__ == "__main__":
    print("\nPredict Number of Students in Future (Tuned XGBoost)\n")

    komuna = input("Enter Komuna (e.g., Prishtine): ").strip()
    year = input("Enter Viti Akademik (e.g., 2030): ").strip()
    niveli = input("Enter Niveli Akademik (e.g., Para-fillor): ").strip()
    mosha = input("Enter Mosha (e.g., 5-6): ").strip()
    gjinia = input("Enter Gjinia (e.g., Mashkull): ").strip()

    try:
        year = int(year)
    except ValueError:
        print("Viti Akademik must be a number (e.g., 2030).")
        exit()

    komuna_enc = safe_encode(label_encoders['Komuna'], komuna, "Komuna")
    niveli_enc = safe_encode(label_encoders['Niveli Akademik'], niveli, "Niveli Akademik")
    mosha_enc = safe_encode(label_encoders['Mosha'], mosha, "Mosha")
    gjinia_enc = safe_encode(label_encoders['Gjinia'], gjinia, "Gjinia")

    input_features = [[komuna_enc, year, niveli_enc, mosha_enc, gjinia_enc]]
    prediction = final_model.predict(input_features)[0]

    print(f"\nEstimated number of students in {komuna} for {year}: {int(prediction)}")
