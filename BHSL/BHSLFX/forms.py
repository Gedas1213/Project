from .models import UserAccount
from django import forms
import MetaTrader5 as mt


TIME_GAP_CHOICES = (
    (mt.TIMEFRAME_M15, "15 min"),
    (mt.TIMEFRAME_M20, "20 min"),
    (mt.TIMEFRAME_M30, "30 min"),
    (mt.TIMEFRAME_H1, "1 hour"),
    (mt.TIMEFRAME_H2, "2 hour"),
    (mt.TIMEFRAME_H3, "3 hour"),
    (mt.TIMEFRAME_H4, "4 hour"),
    (mt.TIMEFRAME_H6, "6 hour"),
    (mt.TIMEFRAME_H8, "8 hour"),
    (mt.TIMEFRAME_H12, "12 hour"),
    (mt.TIMEFRAME_D1, "1 day"),
)

class AccountCreateForm(forms.ModelForm):
    class Meta:
        model = UserAccount
        fields = ('account_number', 'account_password', 'server')
        widgets = {'owner': forms.HiddenInput()}
        
class CurrencyPair(forms.Form):
    currency_pair = forms.CharField(max_length=6)
    time_frame = forms.ChoiceField(choices=TIME_GAP_CHOICES)