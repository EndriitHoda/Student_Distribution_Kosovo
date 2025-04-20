import pandas as pd
import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress warnings
warnings.filterwarnings('ignore')

# === Load and preprocess data ===
df = pd.read_csv('cleaned_dataset.csv')  # Your CSV file

# Encode categorical columns
label_encoders = {}
for col in ['Komuna', 'Niveli Akademik', 'Mosha', 'Gjinia']:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

# === Data augmentation (light noise) ===
def augment_data(df, n=2):
    df_aug = df.copy()
    for _ in range(n):
        aug = df.copy()
        aug['Viti Akademik'] += np.random.randint(-2, 3, size=len(df))  # +/- 2 years
        noise = np.random.normal(0, 10, size=len(df))  # small noise
        aug['Numri i nxenesve'] = (aug['Numri i nxenesve'] + noise).clip(lower=0)
        df_aug = pd.concat([df_aug, aug], ignore_index=True)
    return df_aug

df_aug = augment_data(df, n=2)

# Feature matrix and target variable
X = df_aug[['Komuna_encoded', 'Viti Akademik', 'Niveli Akademik_encoded', 'Mosha_encoded', 'Gjinia_encoded']]
y = df_aug['Numri i nxenesve']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate the model on test set ===
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📈 Regression Model Evaluation on Test Set:")
print(f"🔹 R² Score (Accuracy-like): {r2:.2f}")
print(f"🔹 Mean Absolute Error (MAE): {mae:.2f}")
print(f"🔹 Mean Squared Error (MSE): {mse:.2f}")
print(f"🔹 Root Mean Squared Error (RMSE): {rmse:.2f}")

# === Cross-validation ===
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("\n📊 Cross-Validated R² Scores:", np.round(cv_scores, 3))
print(f"🔁 Average R² (CV): {np.mean(cv_scores):.2f}")

# === Predict using terminal input ===
def safe_encode(encoder, value, column_name):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        print(f"\n❌ Error: '{value}' is not a valid value for {column_name}.")
        print(f"✅ Valid options are: {list(encoder.classes_)}")
        exit()

if __name__ == "__main__":
    print("\n🧠 Student Number Predictor (Regression Mode)\n")

    # Inputs from user
    komuna = input("Enter Komuna (e.g., Prishtine): ").strip()
    year = input("Enter Viti Akademik (e.g., 2030): ").strip()
    niveli = input("Enter Niveli Akademik (e.g., Para-fillor): ").strip()
    mosha = input("Enter Mosha (e.g., 5-6): ").strip()
    gjinia = input("Enter Gjinia (e.g., Mashkull): ").strip()

    try:
        year = int(year)
    except ValueError:
        print("❌ Viti Akademik must be a valid number like 2030.")
        exit()

    # Encode user inputs
    komuna_enc = safe_encode(label_encoders['Komuna'], komuna, "Komuna")
    niveli_enc = safe_encode(label_encoders['Niveli Akademik'], niveli, "Niveli Akademik")
    mosha_enc = safe_encode(label_encoders['Mosha'], mosha, "Mosha")
    gjinia_enc = safe_encode(label_encoders['Gjinia'], gjinia, "Gjinia")

    # Prepare and predict
    input_features = [[komuna_enc, year, niveli_enc, mosha_enc, gjinia_enc]]
    prediction = model.predict(input_features)[0]

    print(f"\n🔮 Estimated number of students in {komuna} for {year}: {int(prediction)}")