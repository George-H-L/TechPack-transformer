from django.urls import path
from . import views 

app_name = 'techpack_generator'

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('list', views.techpack_list, name='list'),
    path('create/', views.create_techpack, name='create'),
    path('<int:pk>/', views.techpack_detail, name='detail'),
    path('<int:pk>/modify', views.modify_techpack, name='modify'),
    path('<int:pk>/download', views.download_svg, name='download_svg'),
    path('<int:pk>/delete', views.delete_techpack, name='delete'),
    path('<int:pk>/preview', views.preview_svg, name='preview_svg'),
]