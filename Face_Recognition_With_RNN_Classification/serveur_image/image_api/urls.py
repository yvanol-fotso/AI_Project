from django.urls import path
from image_api import views

urlpatterns = [
    path('upload', views.upload_image, name='upload_image'),
]
