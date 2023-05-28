from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('register', views.register, name="register"),
    path('myaccounts', views.MyAccounts.as_view(), name='myaccounts'),
    path('myaccounts/new', views.UsersAccountListView.as_view(), name='add_account'),
    path('myaccounts/<int:pk>', views.CurrencyData.as_view(), name='bot'),
    path('useraccount/<int:pk>/delete/', views.UserAccountDeleteView.as_view(), name='useraccount_delete'),
    path('close_all_positions/', views.close_all_positions, name='close_all_positions'),
    path("botbot", views.Bot.as_view(), name="next_page"),
]