import pandas as pd
import numpy as np
import warnings
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

warnings.filterwarnings('ignore')

LABEL_COLUMNS = ['Komuna', 'Niveli Akademik', 'Mosha', 'Gjinia']
MODEL_FILENAME = 'student_model.pkl'
ENCODERS_FILENAME = 'label_encoders.pkl'


def augment_data(df, n=2):
    df_aug = df.copy()
    for _ in range(n):
        aug = df.copy()
        aug['Viti Akademik'] += np.random.randint(-2, 3, size=len(df))
        noise = np.random.normal(0, 10, size=len(df))
        aug['Numri i nxenesve'] = (aug['Numri i nxenesve'] + noise).clip(lower=0)
        df_aug = pd.concat([df_aug, aug], ignore_index=True)
    return df_aug


def retrain_model(csv_path='cleaned_dataset.csv', augment_n=2):
    if not os.path.exists(csv_path):
        print(f"Dataset '{csv_path}' not found.")
        return

    df = pd.read_csv(csv_path)

    label_encoders = {}
    for col in LABEL_COLUMNS:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le

    df_aug = augment_data(df, n=augment_n)

    X = df_aug[['Komuna_encoded', 'Viti Akademik', 'Niveli Akademik_encoded', 'Mosha_encoded', 'Gjinia_encoded']]
    y = df_aug['Numri i nxenesve']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=25, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nRegression Model Evaluation on Test Set:")
    print(f"R² Score (Accuracy-like): {r2_score(y_test, y_pred):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print("\nCross-Validated R² Scores:", np.round(cv_scores, 3))
    print(f"Average R² (CV): {np.mean(cv_scores):.2f}")

    # Save model and encoders
    joblib.dump(model, MODEL_FILENAME)
    joblib.dump(label_encoders, ENCODERS_FILENAME)
    print(f"\nModel saved to '{MODEL_FILENAME}'")
    print(f"Encoders saved to '{ENCODERS_FILENAME}'")


if __name__ == "__main__":
    retrain_model()
