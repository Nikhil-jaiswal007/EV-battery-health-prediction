import json
import os
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie

from .ml.model_service import BatteryModelService


@ensure_csrf_cookie
def index(request):
    """
    Renders the EV Battery Health Prediction Dashboard.
    Ensures CSRF cookie is set for client-side API requests.
    """
    return render(request, 'predictor/index.html')


@csrf_exempt
def predict_api(request):
    """
    POST /api/predict/
    Accepts telemetry JSON payload, performs model inference,
    and returns State of Health (SoH), Remaining Useful Life (RUL),
    Pack Valuation ($), and degradation projection curve.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed.'}, status=405)

    try:
        if request.content_type == 'application/json' or request.body:
            try:
                payload = json.loads(request.body.decode('utf-8'))
            except json.JSONDecodeError:
                return HttpResponseBadRequest(json.dumps({'error': 'Invalid JSON body'}), content_type='application/json')
        else:
            payload = request.POST.dict()

        service = BatteryModelService.get_instance()
        result = service.predict(payload)

        return JsonResponse(result, status=200)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def health_api(request):
    """
    GET /api/health/
    Returns backend status, loaded models, and NASA benchmark metrics.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(base_dir, 'ml', 'saved_models', 'metrics.json')
    
    metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except Exception:
            pass

    return JsonResponse({
        'status': 'healthy',
        'framework': 'Django 5.2',
        'available_models': ['xgboost', 'rf', 'nn'],
        'metrics': metrics
    })
