"""
model_service.py - Battery Health Inference Engine & Service

Provides inference service for:
- SoH (State of Health %) prediction via XGBoost, Random Forest, or Neural Network
- RUL (Remaining Useful Life in cycles to 80% EOL threshold)
- Pack Valuation ($) based on battery condition ($120/kWh benchmark)
- Life-cycle degradation curve projection across cycles [0..600]
"""

import os
import joblib
import numpy as np
import pandas as pd

FEATURE_NAMES = ['cycle', 'soc', 'voltage', 'current', 'temperature', 'capacity', 'fast_charging']


class BatteryModelService:
    _instance = None

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, 'saved_models')
        self.models = {}
        self.scaler = None
        self.load_models()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_models(self):
        xgb_path = os.path.join(self.models_dir, 'xgb_model.joblib')
        rf_path = os.path.join(self.models_dir, 'rf_model.joblib')
        nn_path = os.path.join(self.models_dir, 'nn_model.joblib')
        scaler_path = os.path.join(self.models_dir, 'scaler.joblib')

        # If models don't exist yet, trigger training
        if not (os.path.exists(xgb_path) and os.path.exists(rf_path) and 
                os.path.exists(nn_path) and os.path.exists(scaler_path)):
            from .train import train_and_save_models
            train_and_save_models()

        self.models['xgboost'] = joblib.load(xgb_path)
        self.models['rf'] = joblib.load(rf_path)
        self.models['nn'] = joblib.load(nn_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, data: dict) -> dict:
        cycle = float(data.get('cycle', 20))
        soc = float(data.get('soc', 80))
        voltage = float(data.get('voltage', 17.5))
        current = float(data.get('current', 1.0))
        temperature = float(data.get('temperature', 20.0))
        capacity = float(data.get('capacity', 9.5))
        model_name = str(data.get('model', 'xgboost')).lower()
        fast_charging = 1 if data.get('fast_charging', False) in [True, 1, 'true', '1'] else 0

        if model_name not in self.models:
            model_name = 'xgboost'

        model = self.models[model_name]

        # Construct single input DataFrame
        input_df = pd.DataFrame([{
            'cycle': cycle,
            'soc': soc,
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'capacity': capacity,
            'fast_charging': fast_charging
        }])[FEATURE_NAMES]

        # Model inference
        if model_name == 'nn':
            features_scaled = self.scaler.transform(input_df)
            pred_soh = float(model.predict(features_scaled)[0])
        else:
            pred_soh = float(model.predict(input_df)[0])

        # Bounds clipping
        pred_soh = float(np.clip(pred_soh, 40.0, 99.9))

        # Remaining Useful Life (RUL) estimation to 80% EOL
        # Effective cycle fade rate per cycle under current stress factors
        temp_factor = 1.0 + max(0.0, temperature - 25.0) * 0.02
        fast_factor = 1.35 if fast_charging == 1 else 1.0
        c_rate_factor = 1.0 + (current - 1.0) * 0.1
        degradation_rate = 0.065 * temp_factor * fast_factor * c_rate_factor

        if pred_soh > 80.0:
            remaining_cycles = max(0, int(round((pred_soh - 80.0) / degradation_rate)))
        else:
            remaining_cycles = 0

        # Pack Valuation: $120/kWh base for 40kWh equivalent automotive pack ($4800 at 90% SoH)
        pack_value = int(round((pred_soh / 100.0) * 5333))

        # Lifecycle degradation curve projection across cycles [0, 100, 200, 300, 400, 500, 600]
        curve_cycles = [0, 100, 200, 300, 400, 500, 600]
        curve_records = []
        for c in curve_cycles:
            # Capacity slightly decreases along the projection
            projected_cap = max(6.0, capacity - ((c - cycle) * 0.005))
            curve_records.append({
                'cycle': float(c),
                'soc': soc,
                'voltage': voltage,
                'current': current,
                'temperature': temperature,
                'capacity': projected_cap,
                'fast_charging': fast_charging
            })
        
        curve_df = pd.DataFrame(curve_records)[FEATURE_NAMES]
        if model_name == 'nn':
            curve_scaled = self.scaler.transform(curve_df)
            curve_preds = model.predict(curve_scaled)
        else:
            curve_preds = model.predict(curve_df)

        curve = [round(float(np.clip(val, 40.0, 99.9)), 2) for val in curve_preds]

        # Determine battery condition status
        if pred_soh >= 85.0:
            status = "Healthy"
        elif pred_soh >= 75.0:
            status = "Moderate"
        else:
            status = "Degraded"

        return {
            'soh': round(pred_soh, 2),
            'remaining_cycles': remaining_cycles,
            'pack_value': pack_value,
            'curve': curve,
            'model_used': model_name,
            'status': status
        }
