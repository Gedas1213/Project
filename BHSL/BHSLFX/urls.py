from django.urls import path, re_path, include
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('register', views.register, name="register"),
    path('myaccounts', views.MyAccounts.as_view(), name='myaccounts'),
    path('myaccounts/new', views.UsersAccountListView.as_view(), name='add_account'),
    path('myaccounts/<int:pk>', views.Bot.as_view(), name='bot'),
]