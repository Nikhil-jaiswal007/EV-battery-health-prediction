"""
train.py - ML Model Training Pipeline for EV Battery SoH Prediction

Trains 3 production ML models:
1. XGBoost Regressor
2. Random Forest Regressor
3. Neural Network (MLP Regressor)

Saves trained model artifacts and performance metrics.
"""

import os
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

from dataset import generate_battery_dataset

FEATURE_NAMES = ['cycle', 'soc', 'voltage', 'current', 'temperature', 'capacity', 'fast_charging']
TARGET_NAME = 'soh'


def train_and_save_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(models_dir, exist_ok=True)

    print("Generating simulated NASA battery degradation dataset...")
    df = generate_battery_dataset(num_samples=7000, random_seed=42)

    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit feature scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. XGBoost
    print("Training XGBoost Regressor...")
    xgb_model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    # 2. Random Forest
    print("Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    # 3. Neural Network (MLP)
    print("Training Neural Network (MLP) Regressor...")
    nn_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=400,
        random_state=42
    )
    nn_model.fit(X_train_scaled, y_train)
    nn_preds = nn_model.predict(X_test_scaled)

    # Evaluate models
    def get_metrics(y_true, y_pred):
        return {
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred))
        }

    metrics = {
        'xgboost': get_metrics(y_test, xgb_preds),
        'rf': get_metrics(y_test, rf_preds),
        'nn': get_metrics(y_test, nn_preds)
    }

    print("\nModel Training Evaluation Metrics:")
    for m_name, m_vals in metrics.items():
        print(f"[{m_name.upper()}] R2 Score: {m_vals['r2']:.4f} | RMSE: {m_vals['rmse']:.3f} | MAE: {m_vals['mae']:.3f}")

    # Serialize artifacts
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgb_model.joblib'))
    joblib.dump(rf_model, os.path.join(models_dir, 'rf_model.joblib'))
    joblib.dump(nn_model, os.path.join(models_dir, 'nn_model.joblib'))

    with open(os.path.join(models_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll models and scalers successfully serialized to: {models_dir}")


if __name__ == '__main__':
    train_and_save_models()
