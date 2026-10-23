import os
import django
import json

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ev_battery_health.settings')
django.setup()

from django.test import Client

def run_tests():
    client = Client()
    print("=== Running EV Battery Health Application Tests ===\n")

    # 1. Test GET /
    print("1. Testing GET / (Dashboard UI)...")
    res = client.get('/')
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert b"EV Battery Health Prediction Dashboard" in res.content
    assert b"Telemetry Inputs" in res.content
    print("   [PASS] Dashboard UI successfully rendered.")

    # 2. Test GET /api/health/
    print("2. Testing GET /api/health/...")
    res = client.get('/api/health/')
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data['status'] == 'healthy'
    assert 'xgboost' in data['available_models']
    print(f"   [PASS] Health check passed. Available models: {data['available_models']}")

    # 3. Test POST /api/predict/ with XGBoost
    print("3. Testing POST /api/predict/ with XGBoost...")
    payload_xgb = {
        'cycle': 20,
        'soc': 80,
        'voltage': 17.50,
        'current': 1.00,
        'temperature': 20.0,
        'capacity': 9.50,
        'model': 'xgboost',
        'fast_charging': False
    }
    res = client.post('/api/predict/', data=json.dumps(payload_xgb), content_type='application/json')
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert 'soh' in data
    assert 'remaining_cycles' in data
    assert 'pack_value' in data
    assert 'curve' in data
    assert len(data['curve']) == 7
    assert data['model_used'] == 'xgboost'
    print(f"   [PASS] XGBoost: SoH={data['soh']}%, RUL={data['remaining_cycles']} cycles, Value=${data['pack_value']}, Status={data['status']}")

    # 4. Test POST /api/predict/ with Random Forest
    print("4. Testing POST /api/predict/ with Random Forest...")
    payload_rf = dict(payload_xgb, model='rf')
    res = client.post('/api/predict/', data=json.dumps(payload_rf), content_type='application/json')
    assert res.status_code == 200
    data = res.json()
    assert data['model_used'] == 'rf'
    print(f"   [PASS] Random Forest: SoH={data['soh']}%, RUL={data['remaining_cycles']} cycles, Value=${data['pack_value']}")

    # 5. Test POST /api/predict/ with Neural Network (MLP)
    print("5. Testing POST /api/predict/ with Neural Network (MLP)...")
    payload_nn = dict(payload_xgb, model='nn')
    res = client.post('/api/predict/', data=json.dumps(payload_nn), content_type='application/json')
    assert res.status_code == 200
    data = res.json()
    assert data['model_used'] == 'nn'
    print(f"   [PASS] Neural Network: SoH={data['soh']}%, RUL={data['remaining_cycles']} cycles, Value=${data['pack_value']}")

    # 6. Test Stress Conditions (Thermal stress & Fast Charging)
    print("6. Testing Stress Conditions (48°C + Fast Charging + 350 cycles)...")
    payload_stress = {
        'cycle': 350,
        'soc': 40,
        'voltage': 14.80,
        'current': 2.50,
        'temperature': 48.0,
        'capacity': 7.20,
        'model': 'xgboost',
        'fast_charging': True
    }
    res = client.post('/api/predict/', data=json.dumps(payload_stress), content_type='application/json')
    assert res.status_code == 200
    data = res.json()
    print(f"   [PASS] Stress Test Result: SoH={data['soh']}%, RUL={data['remaining_cycles']} cycles, Status={data['status']}")
    assert data['soh'] < payload_xgb['capacity'] * 10 or data['status'] in ['Moderate', 'Degraded']

    print("\nALL BACKEND API & MODEL TESTS PASSED SUCCESSFULLY! (6/6)")

if __name__ == '__main__':
    run_tests()
