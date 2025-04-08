# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
data = pd.read_csv('cleaned_dataset.csv')

# Preprocessing: Separating features and target variable
X = data.drop(columns=['Numri i nxenesve'])
y = data['Numri i nxenesve']

# Define categorical features
categorical_features = ['Komuna', 'Niveli Akademik', 'Mosha', 'Gjinia']

# Create a preprocessor pipeline to handle categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)  # One-hot encode categorical columns
    ],
    remainder='passthrough'  # Leave the numerical columns as they are
)

# Create a pipeline that first preprocesses the data, then fits a RandomForestRegressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))  # Random Forest with 100 trees
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate Root Mean Squared Error (RMSE) for better interpretability
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")
