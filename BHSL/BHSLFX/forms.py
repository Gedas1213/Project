from .models import UserAccount
from django import forms


TIME_GAP_CHOISES = (
    ("TIMEFRAME_M1", "1 min"),
    ("TIMEFRAME_M2", "2 min"),
    ("TIMEFRAME_M3", "3 min"),
    ("TIMEFRAME_M4", "4 min"),
    ("TIMEFRAME_M5", "5 min"),
    ("TIMEFRAME_M6", "6 min"),
    ("TIMEFRAME_M10", "10 min"),
    ("TIMEFRAME_M12", "12 min"),
    ("TIMEFRAME_M15", "15 min"),
    ("TIMEFRAME_M20", "20 min"),
    ("TIMEFRAME_M30", "30 min"),
    ("TIMEFRAME_H1", "1 hour"),
)

class AccountCreateForm(forms.ModelForm):
    class Meta:
        model = UserAccount
        fields = ('account_number', 'account_password', 'server')
        widgets = {'owner': forms.HiddenInput()}

class CurrencyPair(forms.Form):
    currency_pair = forms.CharField(max_length=7)
    time_frame = forms.ChoiceField(choices=TIME_GAP_CHOISES)
