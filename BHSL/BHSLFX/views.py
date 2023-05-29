from django.shortcuts import render, redirect
import tradingeconomics as te
te.login()
import MetaTrader5 as mt
from django.views.decorators.csrf import csrf_protect
from django.contrib import messages
from django.contrib.auth.forms import User
from django.http import HttpResponseRedirect
import datetime as datetime
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views import generic
from django.urls import reverse_lazy
from .models import UserAccount
from .forms import CurrencyPair
import pandas as pd
import torch
import pygad.torchga
import pygad
import torch.nn as nn
import numpy as np
import ta
from dateutil.relativedelta import relativedelta
import os
import logging


# Create your views here.

path = 'data\model.pth'

def index(request):
    us_calendar=te.getCalendarData(country=['united states'], initDate=datetime.datetime.today().strftime("%Y-%m-%d"), endDate=datetime.datetime.today().strftime("%Y-%m-%d"), importance='3', output_type='df')
    eco_calendar = us_calendar.drop(columns=['CalendarId', 'Event', 'Reference', 'ReferenceDate', 'Source', 'TEForecast', 'URL', 'DateSpan', 'Importance', 'LastUpdate', 'Revised', 'Currency', 'Unit', 'Ticker', 'Symbol'])
    df_conv=[]
    for i in range(eco_calendar.shape[0]):
        temp = eco_calendar.iloc[i]
        df_conv.append(dict(temp))
    update_time = datetime.datetime.today()
  

    context = {
        "df_conv": df_conv,
        "update_time": update_time,
    }
    return render(request, "index.html", context=context)

SpecialSym =['$', '@', '#', '%']


@csrf_protect
def register(request):
    if request.method == "POST":
       
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        password2 = request.POST['password2']
      
        if password != password2:
            messages.error(request, ('Slaptažodžiai nesutampa!'))
            return redirect('register')
      
        if User.objects.filter(username=username).exists():
            messages.error(
                request, ('Vartotojo vardas %s užimtas!') % username)
            return redirect('register')
           
        if len(password) < 8:
            messages.error(request, ('Slaptažodis turi būti ne trumpesnis kaip 8 simbolių'))
            return redirect('register')
        
        if not any(char.isdigit() for char in password):
            messages.error(request, ('Slaptažodyje turi būti bent vienas skaičius'))
            return redirect('register')
        
        if not any(char.isalpha() for char in password):
            messages.error(request, ('Slaptažodyje turi būti bent viena raidė'))
            return redirect('register')
        
        if not any(char in SpecialSym for char in password):
            messages.error(request, ('Slaptažodyje turi būti panaudotas bent vienas iš šių simbolių: $, @, #, %'))
            return redirect('register')
        
        if User.objects.filter(email=email).exists():
            messages.error(
                request, ('Vartotojas su el. paštu %s jau užregistruotas!') % email)
            return redirect('register')
        User.objects.create_user(
            username=username, email=email, password=password)
        messages.info(
            request, ('Vartotojas %s užregistruotas!') % username)
        return redirect('login')
    return render(request, 'register.html')

def close_all_positions(request):
    if request.method == 'POST':
        positions = mt.positions_get() 

        if positions is None or len(positions) == 0:
            logging.error("No positions found, error code = %s", mt.last_error())
        else:
            for position in positions:
                symbol = position.symbol
                volume = position.volume
                action_type = mt.TRADE_ACTION_DEAL
                position_type = mt.ORDER_TYPE_SELL if position.type == mt.ORDER_TYPE_BUY else mt.ORDER_TYPE_BUY

                close_order = {
                    "action": action_type,
                    "symbol": symbol,
                    "volume": volume,
                    "type": position_type,
                    "position": position.ticket,
                    "price": mt.symbol_info_tick(symbol).ask if position.type == mt.ORDER_TYPE_BUY else mt.symbol_info_tick(symbol).bid,
                    "magic": 234000,
                    "deviation": 20,
                    "comment": "python script close",
                    "type_time": mt.ORDER_TIME_GTC,
                    "type_filling": mt.ORDER_FILLING_IOC,
                }

                result = mt.order_send(close_order)

                if result.retcode != mt.TRADE_RETCODE_DONE:
                    logging.error("Failed to close position with ticket #:", position.ticket, ". Error code =", result.retcode)

        return redirect('index')

class UsersAccountListView(LoginRequiredMixin, generic.CreateView):
    model = UserAccount
    fields = ['account_number', 'account_password', 'server']
    success_url = 'http://127.0.0.1:8000/BHSLFX/myaccounts'
    template_name = "user_account_form.html"

    def form_valid(self, form):
        form.instance.owner = self.request.user
        return super().form_valid(form)
    
class MyAccounts(LoginRequiredMixin, generic.ListView):
    model = UserAccount
    template_name = "user_accounts.html"

    def get_queryset(self):
        mt.shutdown
        if not mt.shutdown():
            print("Shutting down failed")
        
        print('shoutdown completed successfully')
        return UserAccount.objects.filter(owner=self.request.user)
    
class UserAccountDeleteView(LoginRequiredMixin, generic.DeleteView):
    model = UserAccount
    success_url = reverse_lazy('myaccounts')
    template_name = 'useraccount_confirm_delete.html'

    def get_queryset(self):
        return UserAccount.objects.filter(owner=self.request.user)
    

class CurrencyData(LoginRequiredMixin, generic.DetailView, generic.edit.FormMixin):
    model = UserAccount
    template_name = 'currency_data.html'
    form_class = CurrencyPair
    scuscess_url = 'http://127.0.0.1:8000/BHSLFX/'
    
        

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        login = UserAccount.objects.first().account_number
        password = UserAccount.objects.first().account_password
        acc_server = UserAccount.objects.first().server

        mt.initialize(login, password, acc_server)
        if not mt.initialize():
             print("initialize() failed")
             mt.shutdown()

        mt.login(login, password)

        account_info = mt.account_info()
        balance = account_info.balance
        status = mt.terminal_info().connected
       
        context['balance'] = balance
        context['status'] = status
        return context
    
  
    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST)
        if form.is_valid():
            currency_pair = form.cleaned_data['currency_pair']
            time_frame = form.cleaned_data['time_frame']
            make_training = form.cleaned_data.get('make_training', 0)
            self.request.session['currency_pair'] = currency_pair
            self.request.session['time_frame'] = time_frame
            if os.path.isfile(path):
                self.request.session['make_training'] = make_training
            else:
                self.request.session['make_training'] = '0'
            
        return HttpResponseRedirect('http://127.0.0.1:8000/BHSLFX/botbot')


directory = 'data'
os.makedirs(directory, exist_ok=True)

class Bot(LoginRequiredMixin, generic.CreateView):
    model = UserAccount
    template_name = 'bot.html'

    def get(self, request):
        currency_pair = request.session.get('currency_pair', None)
        time_frame = request.session.get('time_frame', None)
        make_training = request.session.get('make_training', None)

        action, output, balance, margin_free = trading_logic(currency_pair, time_frame, make_training)

        time_frame_int = int(time_frame)

        if time_frame_int == 16385:
            time_frame_min = 1440
        elif time_frame_int == 16396:
            time_frame_min = 720
        elif time_frame_int == 16392:
            time_frame_min = 480
        elif time_frame_int == 16390:
            time_frame_min = 360
        elif time_frame_int == 16388:
            time_frame_min = 240
        elif time_frame_int == 16386:
            time_frame_min = 120 
        elif time_frame_int == 16385:
            time_frame_min = 60
        else:
            time_frame_min = time_frame_int

        positions = mt.positions_get()
        positions_active = positions if positions else []

        context = {
            'positions_active': positions_active,
            'currency_pair': currency_pair, 
            'time_frame': time_frame,
            'time_frame_min': time_frame_min,
            'output': output, 
            'balance': balance, 
            'margin_free': margin_free,
            'action': action,
            }
        return render(request, self.template_name, context)

def trading_logic(currency_pair, time_frame, make_training):
        
    time = time = relativedelta(years = 20)
    if time_frame == '16385':
        time = relativedelta(years = 11)
    elif time_frame == '30':
        time = relativedelta(years = 5)
    elif time_frame == '20':
        time = relativedelta(years = 3)
    elif time_frame == '15':
        time = relativedelta(years = 2)
    elif time_frame == '1':
        time = relativedelta(months = 1)
    

    if currency_pair is not None:
        get_data = pd.DataFrame(mt.copy_rates_range(currency_pair, 
                            int(time_frame), 
                            (datetime.datetime.now()-time), 
                            datetime.datetime.now()))

        get_data['returns'] = (get_data['close'] / get_data['close'].shift(1))-1
        df = get_data.drop(['real_volume', 'spread', 'time'], axis=1)
        df.to_csv('data\instument_data.csv', index=False)
        data = pd.read_csv('data\instument_data.csv')
        train_size = int(len(data)*0.8)
        df = pd.DataFrame(data[:train_size])
        X_test = pd.DataFrame(data[train_size:])
        

        log_directory = 'logs'
        os.makedirs(log_directory, exist_ok=True)
        current_datetime = datetime.datetime.today().strftime("%Y-%m-%d")
        filename = f"logs/{current_datetime}.txt"
        logging.basicConfig(filename=filename, encoding="UTF-8", level=logging.ERROR, format="%(asctime)s :%(filename)s: %(message)s")

        account_info = mt.account_info()
        balance = account_info.balance
        margin_free = account_info.margin_free

        if make_training =='0':
            ga_instance.run()
            np.save('data\\best_solution.npy', solution)
            torch.save(model.state_dict(), 'data\\model.pth')

        if os.path.isfile(path):
            model.load_state_dict(torch.load('data\model.pth'))
            p_model = model.eval()

        if os.path.isfile('data\\best_solution.npy'):
            p_solution = np.load('data\\best_solution.npy')

        print(X_test, "tokia vat")
        pred = make_predictions(p_model, X_test, p_solution)
        print(pred)
        result = pred.detach().numpy()
        list_result = result.tolist()
        output = pd.DataFrame(list_result, columns = ['Pikrti', 'Parduoti', 'Laikyti'])
        output=output.iloc[-1:].to_html(index=False, float_format="%.2f", col_space = 65)
        output = output.replace('<table', '<table style="margin: 0 auto;text-align:center;"')
        output = output.replace('<tr', '<tr style="text-align:center"')

        weekno = datetime.datetime.today().weekday()
        if weekno < 5:
            if pred[-1, 0] > pred[-1, 1] and pred[-1, 0] > pred[-1, 2] and pred[-1, 0] > 0.4:

                buy_request = {
                    "action": mt.TRADE_ACTION_DEAL,
                    "symbol": "currency_pair",
                    "volume": 0.1,
                    "type": mt.ORDER_TYPE_BUY,
                    "price": mt.symbol_info_tick(currency_pair).ask,
                    "sl": mt.symbol_info_tick(currency_pair).ask * 0.99,
                    "tp": mt.symbol_info_tick(currency_pair).ask *1.01,
                    "magic": 234000,
                    "type_time": mt.ORDER_TIME_DAY,
                    "type_filling": mt.ORDER_FILLING_FOK,
                    }

                open_positions = mt.positions_get()
                if open_positions is None:
                    logging.error("No positions found, error code =",mt.last_error())
                elif len(open_positions)==0:
                    logging.error("No positions found")
                else:
                    for position in open_positions:
                        if position.type == mt.ORDER_TYPE_SELL:
                            close_short_request = {
                                "action": mt.TRADE_ACTION_DEAL,
                                "symbol": position.symbol,
                                "volume": position.volume,
                                "type": mt.ORDER_TYPE_BUY,
                                "position": position.ticket,
                                "magic": position.magic,
                                "deviation": 20,
                                "comment": "closing short position",
                                "type_time": mt.ORDER_TIME_GTC,
                                "type_filling": mt.ORDER_FILLING_IOC,
                            }
                            result = mt.order_send(close_short_request)
                        
                            if result.retcode != mt.TRADE_RETCODE_DONE:
                                logging.error("order send failed, retcode={}".format(result.retcode))
                                action = "Klaida, nepavyko uždayti pozicijų"
                            else:
                                order = result.order
                                logging.error("short position #{} closed.".format(position.ticket))

                if margin_free >= 1000:
                    result = mt.order_send(buy_request)
                    action = "PERKA"
                    if result.retcode != mt.TRADE_RETCODE_DONE:
                        logging.error("order send failed, retcode={}".format(result.retcode))
                        action = "Klaida, nepavyko įvykdyti užsakymo?"
                    else:
                        order = result.order
                        if hasattr(order, 'ticket'):
                            logging.error("order #{} filled.".format(order.ticket))
                else:
                    logging.error('not sufficient amount of funds for a request')
                    action = "Per mažas pinigų likutis, papildykite sąskaitą"
                    order = None



            elif pred[-1, 1] > pred[-1, 0] and pred[-1, 1] > pred[-1, 2] and pred[-1, 1] > 0.4:

                sell_request = {
                "action": mt.TRADE_ACTION_DEAL,
                "symbol": currency_pair,
                "volume": 0.1,
                "type": mt.ORDER_TYPE_SELL,
                "price": mt.symbol_info_tick(currency_pair).bid,
                "sl": mt.symbol_info_tick(currency_pair).ask * 1.01,
                "tp": mt.symbol_info_tick(currency_pair).ask *0.99,
                "magic": 234000,
                "type_time": mt.ORDER_TIME_DAY,
                "type_filling": mt.ORDER_FILLING_FOK,
                }
                open_positions = mt.positions_get()
                if open_positions is None:
                    logging.error("No positions found, error code =",mt.last_error())
                elif len(open_positions)==0:
                    logging.error("No positions found")
                else:
                    for position in open_positions:
                        if position.type == mt.ORDER_TYPE_BUY:
                            close_long_request = {
                                "action": mt.TRADE_ACTION_DEAL,
                                "symbol": position.symbol,
                                "volume": position.volume,
                                "type": mt.ORDER_TYPE_SELL,
                                "position": position.ticket,
                                "magic": position.magic,
                                "deviation": 20,
                                "comment": "closing long position",
                                "type_time": mt.ORDER_TIME_GTC,
                                "type_filling": mt.ORDER_FILLING_IOC,
                            }
                            result = mt.order_send(close_long_request)
                        
                            if result.retcode != mt.TRADE_RETCODE_DONE:
                                logging.error("order_send failed, retcode={}".format(result.retcode))
                                action = "Klaida, nepavyko uždayti pozicijų"
                            else:
                                logging.error("long position #{} closed.".format(position.ticket))

                if margin_free >= 1000:
                    result = mt.order_send(sell_request)
                    action = "PARDUODA"
                    if result.retcode != mt.TRADE_RETCODE_DONE:
                        logging.error("order_send failed, retcode={}".format(result.retcode))
                        action = "Klaida, nepavyko įvykdyti užsakymo"
                    else:
                        order = result.order
                        if hasattr(order, 'ticket'):
                            logging.error("order #{} filled.".format(order.ticket))
                else:
                    logging.error('not sufficient amount of funds for a request')
                    action = "Per mažas pinigų likutis, papildykite sąskaitą"
                    order = None

            else:
                logging.error('no actions taken, HOLD')
                action = "LAIKO"
        else:
            action = "Biržos yra uždarytos, gražaus likusio savaitgalio!"
    
    return action, output, balance, margin_free


data = pd.read_csv('data\instument_data.csv')
train_size = int(len(data)*0.8)
df = pd.DataFrame(data[:train_size])
# X_test = pd.DataFrame(data[train_size:])


def fitness_func(ga_instance, solution, sol_idx):
    global df_copy, buy_treshold, sell_treshold, loss_function, model
    df_copy = df.copy()

    ema_period = max(int(solution[0]), 1)
    rsi_period = max(int(solution[1]), 1)
    buy_treshold = int(solution[2])
    sell_treshold = int(solution[3])

    df_copy['EMA'] = ta.trend.ema_indicator(df_copy['close'], ema_period)
    df_copy['RSI'] = ta.momentum.rsi(df_copy['close'], rsi_period)
    df_copy.fillna(0, inplace=True)

    buy_signals = (df_copy['close'] > df_copy['EMA']) & (df_copy['RSI'] < buy_treshold)
    sell_signals = (df_copy['RSI'] > sell_treshold)
    hold_signals = ~(buy_signals | sell_signals)
    buy_signals = buy_signals & ~hold_signals
    sell_signals = sell_signals & ~hold_signals

    trades = np.zeros(len(df))
    trades[buy_signals] = 1
    trades[sell_signals] = 2
    trades[hold_signals] = 0

    trades_long = torch.tensor(trades).long()  # Reshaping the tensor
    inputs = torch.tensor(df_copy[['EMA', 'RSI']].values).float()
    outputs = model(inputs)

    loss = loss_function(outputs, trades_long)


    return -loss.item()

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


class GaModel(nn.Module):
    def __init__(self):
        super(GaModel, self).__init__() 
        self.sequential = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 3),
        )

    def forward(self, x):
        return self.sequential(x)
    

model = GaModel()

loss_function = nn.CrossEntropyLoss()

def predict(model, X_test, best_solution):

    ema_period = max(int(best_solution[0]), 1)
    rsi_period = max(int(best_solution[1]), 1)
    buy_treshold = int(best_solution[2])
    sell_treshold = int(best_solution[3])

    X_test['EMA'] = ta.trend.ema_indicator(X_test['close'], ema_period)
    X_test['RSI'] = ta.momentum.rsi(X_test['close'], rsi_period)
    X_test.fillna(0, inplace=True)

    buy_signals = (X_test['close'] > X_test['EMA']) & (X_test['RSI'] < buy_treshold)
    sell_signals = (X_test['RSI'] > sell_treshold)
    hold_signals = ~(buy_signals | sell_signals)
    buy_signals = buy_signals & ~hold_signals
    sell_signals = sell_signals & ~hold_signals

    trades = np.zeros(len(X_test))
    trades[buy_signals] = 1
    trades[sell_signals] = 2
    trades[hold_signals] = 0

    inputs_test = torch.tensor(X_test[['EMA', 'RSI']].values).float()
    predictions = model(inputs_test)

    return predictions


torch_ga = pygad.torchga.TorchGA(model=model,
                                 num_solutions=4)


num_generations = 1 
num_parents_mating = 2 
sol_per_pop = 50
num_genes = 4
lower_bound = [0, 0, -100, -100]
upper_bound = [200, 50, 100, 100]
initial_population = np.random.uniform(lower_bound, upper_bound, (sol_per_pop, num_genes))
mutation_percent_genes = 50
mutation_by_replacement=True
parent_selection_type = 'rws'
crossover_type = "single_point"
mutation_type = "random"
save_best_solutions = True


ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes, 
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=callback_generation, 
                       mutation_percent_genes=mutation_percent_genes,
                       mutation_by_replacement=mutation_by_replacement,
                       parent_selection_type=parent_selection_type,
                       crossover_type = crossover_type,
                       mutation_type = mutation_type,
                       save_best_solutions=save_best_solutions)


# ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

solution, solution_fitness, solution_idx = ga_instance.best_solution()

print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))



def make_predictions(model, X_test, solution):
    predictions = predict(model, X_test, solution)
    predictions1 = torch.nn.functional.softmax(predictions, dim=1)
    return predictions1

