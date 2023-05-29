from .models import UserAccount
from django import forms
import MetaTrader5 as mt
import os

path = 'data\model.pth'

TIME_GAP_CHOICES = (
    (mt.TIMEFRAME_M1, "1 min"),
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


CURRENCY_PAIRS = (
    ('EURUSD', 'EURUSD'), ('GBPUSD', 'GBPUSD'), ('USDCHF', 'USDCHF'), ('USDJPY', 'USDJPY'),
    ('USDCNH', 'USDCNH'), ('USDRUB', 'USDRUB'), ('AUDUSD', 'AUDUSD'), ('NZDUSD', 'NZDUSD'),
    ('USDCAD', 'USDCAD'), ('USDSEK', 'USDSEK'), ('USDHKD', 'USDHKD'), ('USDSGD', 'USDSGD'),
    ('USDNOK', 'USDNOK'), ('USDDKK', 'USDDKK'), ('USDTRY', 'USDTRY'), ('USDZAR', 'USDZAR'),
    ('USDCZK', 'USDCZK'), ('USDHUF', 'USDHUF'), ('USDPLN', 'USDPLN'), ('USDRUR', 'USDRUR'),
    ('AUDCAD', 'AUDCAD'), ('AUDCHF', 'AUDCHF'), ('AUDJPY', 'AUDJPY'), ('AUDNZD', 'AUDNZD'),
    ('CADCHF', 'CADCHF'), ('CADJPY', 'CADJPY'), ('CHFJPY', 'CHFJPY'), ('EURAUD', 'EURAUD'),
    ('EURCAD', 'EURCAD'), ('EURCHF', 'EURCHF'), ('EURCZK', 'EURCZK'), ('EURDKK', 'EURDKK'),
    ('EURGBP', 'EURGBP'), ('EURHKD', 'EURHKD'), ('EURHUF', 'EURHUF'), ('EURJPY', 'EURJPY'),
    ('EURNOK', 'EURNOK'), ('EURNZD', 'EURNZD'), ('EURPLN', 'EURPLN'), ('EURRUR', 'EURRUR'),
    ('EURSEK', 'EURSEK'), ('EURTRY', 'EURTRY'), ('EURZAR', 'EURZAR'), ('GBPAUD', 'GBPAUD'),
    ('GBPCHF', 'GBPCHF'), ('GBPJPY', 'GBPJPY'), ('XAUUSD', 'XAUUSD'), ('XAUEUR', 'XAUEUR'),
    ('XAUAUD', 'XAUAUD'), ('XAGUSD', 'XAGUSD'), ('XAGEUR', 'XAGEUR'), ('GBPCAD', 'GBPCAD'),
    ('USDCRE', 'USDCRE'), ('GBPNOK', 'GBPNOK'), ('GBPNZD', 'GBPNZD'), ('GBPPLN', 'GBPPLN'),
    ('GBPSEK', 'GBPSEK'), ('GBPSGD', 'GBPSGD'), ('GBPZAR', 'GBPZAR'), ('NZDCAD', 'NZDCAD'),
    ('NZDCHF', 'NZDCHF'), ('NZDJPY', 'NZDJPY'), ('NZDSGD', 'NZDSGD'), ('SGDJPY', 'SGDJPY'),
    ('XPDUSD', 'XPDUSD'), ('XPTUSD', 'XPTUSD'), ('USDGEL', 'USDGEL'), ('USDMXN', 'USDMXN'),
    ('EURMXN', 'EURMXN'), ('GBPMXN', 'GBPMXN'), ('CADMXN', 'CADMXN'), ('CHFMXN', 'CHFMXN'),
    ('MXNJPY', 'MXNJPY'), ('NZDMXN', 'NZDMXN'), ('USDCOP', 'USDCOP'), ('USDARS', 'USDARS'),
    ('USDCLP', 'USDCLP'),
)


TRAINING_TRUE = (
    (0, 'Atlikti modelio mokymą'),
    (1, 'Naudoti buvusį modelį')
)

path = 'data\model.pth'

class AccountCreateForm(forms.ModelForm):
    class Meta:
        model = UserAccount
        fields = ('account_number', 'account_password', 'server')
        widgets = {'owner': forms.HiddenInput()}
        
class CurrencyPair(forms.Form):
    currency_pair = forms.ChoiceField(choices=CURRENCY_PAIRS)
    time_frame = forms.ChoiceField(choices=TIME_GAP_CHOICES)
    if os.path.isfile(path) is None:
        pass
    else:
        make_training = forms.ChoiceField(choices=TRAINING_TRUE)
