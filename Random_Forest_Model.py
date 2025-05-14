import pandas as pd
import numpy as np
import warnings
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename=f'model_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

warnings.filterwarnings('ignore')

# Constants
LABEL_COLUMNS = ['Komuna', 'Niveli Akademik', 'Mosha', 'Gjinia']
MODEL_FILENAME = 'student_model.pkl'
ENCODERS_FILENAME = 'label_encoders.pkl'
REQUIRED_COLUMNS = LABEL_COLUMNS + ['Viti Akademik', 'Numri i nxenesve']

def validate_data(df):
    """Validate input dataframe for required columns and data quality"""
    try:
        # Check for required columns
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for null values
        if df.isnull().any().any():
            logging.warning("Null values detected. Filling with appropriate values...")
            for col in LABEL_COLUMNS:
                df[col] = df[col].fillna(df[col].mode()[0])
            df['Numri i nxenesve'] = df['Numri i nxenesve'].fillna(df['Numri i nxenesve'].median())
            df['Viti Akademik'] = df['Viti Akademik'].fillna(df['Viti Akademik'].median())

        # Check for negative values in Numri i nxenesve
        if (df['Numri i nxenesve'] < 0).any():
            raise ValueError("Negative values found in 'Numri i nxenesve'")

        logging.info("Data validation completed successfully")
        return df
    except Exception as e:
        logging.error(f"Data validation failed: {str(e)}")
        raise

def augment_data(df, n=2):
    """Augment dataset with controlled noise"""
    try:
        df_aug = df.copy()
        for _ in range(n):
            aug = df.copy()
            aug['Viti Akademik'] += np.random.randint(-2, 3, size=len(df))
            noise = np.random.normal(0, df['Numri i nxenesve'].std() * 0.1, size=len(df))
            aug['Numri i nxenesve'] = (aug['Numri i nxenesve'] + noise).clip(lower=0).round()
            df_aug = pd.concat([df_aug, aug], ignore_index=True)
        logging.info(f"Data augmented with {n} iterations")
        return df_aug
    except Exception as e:
        logging.error(f"Data augmentation failed: {str(e)}")
        raise

def plot_feature_importance(model, feature_names, output_path='feature_importance.png'):
    """Plot and save feature importance visualization"""
    try:
        importance = model.feature_importances_
        feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
        feature_imp = feature_imp.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_imp)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Feature importance plot saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to create feature importance plot: {str(e)}")

def retrain_model(csv_path='cleaned_dataset.csv', augment_n=2):
    """Retrain the model with enhanced features"""
    try:
        # Check if dataset exists
        if not os.path.exists(csv_path):
            error_msg = f"Dataset '{csv_path}' not found."
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Load and validate data
        df = pd.read_csv(csv_path)
        df = validate_data(df)
        logging.info(f"Dataset loaded successfully with {len(df)} records")

        # Encode categorical variables
        label_encoders = {}
        for col in LABEL_COLUMNS:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            label_encoders[col] = le

        # Augment data
        df_aug = augment_data(df, n=augment_n)

        # Prepare features and target
        X = df_aug[[col + '_encoded' if col in LABEL_COLUMNS else col 
                   for col in ['Komuna', 'Viti Akademik', 'Niveli Akademik', 'Mosha', 'Gjinia']]]
        y = df_aug['Numri i nxenesve']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info(f"Data split: {len(X_train)} training, {len(X_test)} testing")

        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 25, None],
            'min_samples_split': [2, 5]
        }
        base_model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        model = grid_search.best_estimator_
        logging.info(f"Best parameters: {grid_search.best_params_}")

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate model
        metrics = {
            'R2': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
        }

        print("\nRegression Model Evaluation on Test Set:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")
            logging.info(f"Test {metric}: {value:.2f}")

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        print("\nCross-Validated R² Scores:", np.round(cv_scores, 3))
        print(f"Average R² (CV): {np.mean(cv_scores):.2f}")
        logging.info(f"Cross-validation R²: {np.mean(cv_scores):.2f}")

        # Feature importance
        plot_feature_importance(model, X.columns)

        # Save model and encoders
        joblib.dump(model, MODEL_FILENAME)
        joblib.dump(label_encoders, ENCODERS_FILENAME)
        print(f"\nModel saved to '{MODEL_FILENAME}'")
        print(f"Encoders saved to '{ENCODERS_FILENAME}'")
        logging.info("Model and encoders saved successfully")

    except Exception as e:
        logging.error(f"Model training failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        retrain_model()
    except Exception as e:
        print(f"Error during model training: {str(e)}")
