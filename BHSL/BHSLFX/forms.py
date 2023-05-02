from .models import UserAccount
from django import forms

class AccountCreateForm(forms.ModelForm):
    class Meta:
        model = UserAccount
        fields = ('account_number', 'account_password', 'server')
        widgets = {'owner': forms.HiddenInput()}