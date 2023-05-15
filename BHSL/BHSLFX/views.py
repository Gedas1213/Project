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
from .models import UserAccount
from .forms import CurrencyPair
import pandas as pd

# Create your views here.

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
        # pasiimame reikšmes iš registracijos formos
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        password2 = request.POST['password2']
        # tikriname, ar sutampa slaptažodžiai
        if password != password2:
            messages.error(request, ('Slaptažodžiai nesutampa!'))
            return redirect('register')
        # tikriname, ar neužimtas username
        if User.objects.filter(username=username).exists():
            messages.error(
                request, ('Vartotojo vardas %s užimtas!') % username)
            return redirect('register')
            # tikriname, ar nėra tokio pat email
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
            # jeigu viskas tvarkoje, sukuriame naują vartotoją
        User.objects.create_user(
            username=username, email=email, password=password)
        messages.info(
            request, ('Vartotojas %s užregistruotas!') % username)
        return redirect('login')
    return render(request, 'register.html')

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
            self.request.session['currency_pair'] = currency_pair
            self.request.session['time_frame'] = time_frame

            
            return HttpResponseRedirect('http://127.0.0.1:8000/BHSLFX/botbot')




from dateutil.relativedelta import relativedelta
import ta
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import pygad.torchga
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class Bot(generic.CreateView):
    model = UserAccount
    template_name = 'bot.html'

    def __init__(self):
        super().__init__()
        self.df = None

    def get(self, request):
        currency_pair = request.session.get('currency_pair', None)
        time_frame = request.session.get('time_frame', None)
        if currency_pair is not None:
            get_data = pd.DataFrame(mt.copy_rates_range(currency_pair, 
                               int(time_frame), 
                               (datetime.datetime.now()-relativedelta(years=20)), 
                                datetime.datetime.now()))


            get_data['time'] = pd.to_datetime(get_data['time'], unit="s")

            get_data['returns'] = (get_data['close'] / get_data['close'].shift(1))-1
            get_data.fillna(0, inplace=True)
            get_data.set_index('time', inplace=True)
            df = get_data.drop(['real_volume', 'spread'], axis=1)


            print((df))

            context = {'currency_pair': currency_pair, 'time_frame': time_frame, 'df':df}
            return render(request, self.template_name, context)
        
    def get_df(self):
        return self.df
    
bot = Bot()
df = bot.get_df()
        
#     #PyTorch modelis
class GaModel(nn.Module):
    def __init__(self):
        super(GaModel, self).__init__() 
        self.sequential = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.sequential(x)

model = GaModel()
# state_dict = model.state_dict()

#create an initial population of solutions to the PyTorch model
in_pop = pygad.torchga.TorchGA(model = model, num_solutions=4)



def fitness_function(solution: np.ndarray, df) -> Tuple[float, np.ndarray]:

    ema_period = int(solution[0])
    rsi_period = int(solution[1])
    buy_treshold = int(solution[2])
    sell_treshold = int(solution[3])

    df['EMA'] = ta.trend.ema_indicator(df['close'], ema_period)
    df['RSI'] = ta.momentum.rsi(df['close'], rsi_period)
    
    buy_signals = (df['close']>df['EMA'] & (df['RSI']<buy_treshold))
    sell_signals = (df['RSI']>sell_treshold)
    hold_signals = ~(buy_signals | sell_signals)
    buy_signals = buy_signals & ~hold_signals
    sell_signals = sell_signals & ~hold_signals


    trades = np.zeros(len(df))
    trades[buy_signals] = 1
    trades[sell_signals] = -1
    pnl = (trades * df['returns']).cumsum().iloc[-1]

    fitness = pnl / df['close'].iloc[0]

    signals = np.zeros(len(df))
    signals[buy_signals] = 1
    signals[sell_signals] = -1
    signals[hold_signals] = 0

    #prep data for the training and validating

    scailer = StandardScaler()
    X_df = df[['EMA', 'RSI']].values
    train_size = int(0.8 * len(X_df))

    X_train, X_val = X_df[:train_size], X_df[train_size:]
    Y_train, Y_val = signals[:train_size], signals[train_size:]
    
    X_train = scailer.fit_transform(X_train)
    X_val = scailer.transform(X_val)


    X_train, Y_train = torch.Tensor(X_train), torch.Tensor(Y_train)
    X_val, Y_val = torch.Tensor(X_val), torch.Tensor(Y_val)
        
    return fitness, signals, X_train, Y_train, X_val, Y_val

fitness, signals, X_train, Y_train, X_val, Y_val = fitness_function(solution, df)


class TradingDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)
        
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
train_dataset = TradingDataset(X_train, Y_train)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def train_function(model, train_loader, val_loader, optimizer, patience=10):
    
        #Loss funkcija
    def loss_function(outputs, signals):
        trades = torch.cumsum(outputs, dim=0)
        returns = torch.cumprod(1 + (trades * signals), dim=0)
        roi = returns[-1]
        return -roi

    best_val_loss = float('inf')
    for epoch in range(100):
        model.train()
        for i, (inputs, signals) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, signals)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for inputs, signals in val_loader:
                outputs = model(inputs)
                loss = loss_function(outputs, signals)
                total_loss += loss.item() * inputs

            # Check for early stopping
        if total_loss < best_val_loss:
            best_val_loss = total_loss
            num_epochs_no_improvement = 0
        else:
            num_epochs_no_improvement += 1
        
        if num_epochs_no_improvement >= patience:
            print(f'Validation loss did not improve for {patience} epochs. Stopping early.')
            break
    
    return best_val_loss

num_generations = 200 #can vary between 50 - 10 000
num_parents_mating = 2 #starting point 10 - 20% of the population
initial_population = np.random.randn(4, 4) #should be equal to the num_solutions parameter 
sol_per_pop = 4
num_genes = 4 #the number of parameters to be optimized in the solution
mutation_percent_genes = 10 #percentage of genes that will be randomly mutated in each offspring
parent_selection_type = 'rws'
crossover_type = "single_point"
mutation_type = "random"
keep_parents = True


ga_instance = pygad.GA(num_generations=num_generations,
                    num_parents_mating=num_parents_mating,
                    fitness_function=fitness_function,
                    sol_per_pop=sol_per_pop,
                    num_genes=num_genes,
                    mutation_percent_genes=mutation_percent_genes,
                    initial_population=initial_population,
                    parent_selection_type=parent_selection_type,
                    crossover_type=crossover_type,
                    mutation_type=mutation_type,
                    keep_parents=keep_parents)

ga_instance.run()

ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4) #for algo quality checking


solution, solution_fitness, solution_idx = ga_instance.best_solution()

predictions = pygad.torchga.predict(model=model,
                                solution=solution,
                                data=X_val)

print("Predictions : \n", predictions.detach().numpy())