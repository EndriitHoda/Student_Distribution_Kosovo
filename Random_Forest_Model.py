# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """Load and validate the dataset."""
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully. Shape: {data.shape}")

        # Check for missing values
        if data.isnull().sum().any():
            logger.warning("Dataset contains missing values")
            data = data.dropna()  # Simple handling - could be improved based on needs
        return data
    except FileNotFoundError:
        logger.error(f"File {file_path} not found")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def create_and_train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Pipeline, Dict]:
    """Create and train the model with hyperparameter tuning."""
    # Define categorical features
    categorical_features = ['Komuna', 'Niveli Akademik', 'Mosha', 'Gjinia']

    # Verify all categorical features exist in the dataset
    missing_features = [feat for feat in categorical_features if feat not in X.columns]
    if missing_features:
        raise ValueError(f"Features not found in dataset: {missing_features}")

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Define parameter grid for tuning
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5]
    }

    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def evaluate_model(model: Pipeline, split_data: Dict) -> Dict:
    """Evaluate the model and return performance metrics."""
    y_pred = model.predict(split_data['X_test'])

    # Calculate metrics
    mse = mean_squared_error(split_data['y_test'], y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(split_data['y_test'], y_pred)

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

    # Perform cross-validation
    cv_scores = cross_val_score(model, split_data['X_train'], split_data['y_train'],
                                cv=5, scoring='neg_mean_squared_error')
    metrics['CV_RMSE'] = np.sqrt(-cv_scores.mean())

    return metrics


def plot_feature_importance(model: Pipeline, feature_names: list):
    """Plot feature importance."""
    # Get feature importances from the random forest
    importances = model.named_steps['regressor'].feature_importances_

    # Get feature names after preprocessing
    preprocessor = model.named_steps['preprocessor']
    feature_names_transformed = (preprocessor
                                 .named_transformers_['cat']
                                 .get_feature_names_out(feature_names))

    # Create importance dataframe
    feat_imp_df = pd.DataFrame({
        'feature': feature_names_transformed,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feat_imp_df.head(10))
    plt.title('Top 10 Feature Importances')
    plt.show()


def main():
    try:
        # Load data
        data = load_and_validate_data('cleaned_dataset.csv')

        # Prepare features and target
        X = data.drop(columns=['Numri i nxenesve'])
        y = data['Numri i nxenesve']

        # Train model
        model, split_data = create_and_train_model(X, y)

        # Evaluate model
        metrics = evaluate_model(model, split_data)

        # Print results
        logger.info("Model Performance Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        # Plot feature importance
        categorical_features = ['Komuna', 'Niveli Akademik', 'Mosha', 'Gjinia']
        plot_feature_importance(model, categorical_features)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()