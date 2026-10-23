from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/predict/', views.predict_api, name='predict_api'),
    path('api/health/', views.health_api, name='health_api'),
]
