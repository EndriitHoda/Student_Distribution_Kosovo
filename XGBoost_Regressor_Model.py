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
