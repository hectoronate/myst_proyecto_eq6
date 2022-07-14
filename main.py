
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Technical Analysis                                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: @Rub27182n | @if722399 | @hectoronate                                                       -- #
# -- license: TGNU General Public License v3.0                                                           -- #
# -- repository: https://github.com/Rub27182n/myst_proyecto_eq6.git                                      -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import ta
import data as dt
import numpy as np
import pandas as pd
import functions as fn
import pandas_ta as pta
import visualizations as vz
from backtesting.lib import crossover
from backtesting import Backtest, Strategy

import warnings
warnings.filterwarnings("ignore")


data = dt.BTCUSDT
data.set_index("Open Time", inplace=True)
df = data.copy()
train = data['2018-01-01 00:00:00': '2019-01-01 00:00:00']
test = data['2019-02-01 00:00:00': '2020-02-01 00:00:00']

"""Trading strategy test
In this code we test and show the concept of the trading strategy in the following order:
    1- Calculate Stochastic Relative Strenght Index K and D parameters.
    2- Calculate 3 levels of Exponencial Moving Average
    3- Calculate the sefault Average True Range
    4- Define SRSI K/D cross 
    5- Define Take Profit and Stop Loss with ATR
    6- Calculate buy signals when SRSI and EMA conditions are met
    7- Calculate sell signals 
    8- filter 2 times to exclude consecutive buys
    9- define points when a buy or sell is made
    10- calls a vz function to plot the result
"""

# Technical Indicators
# Stochastic Relative Strenght Index
data['K'] = fn.stochrsi_k(pd.Series(data.Close), 14, 3, 3)
data['D'] = fn.stochrsi_d(pd.Series(data.Close), 14, 3, 3)

# Exponential Weighted Moving Average
emas = [8, 14, 40]
for i in emas:
    data['EMA_'+str(i)] = fn.ema(data.Close, i)

# Average True Range
data['ATR'] = fn.atr(data.High, data.Low, data.Close)

data.dropna(inplace=True)

data['KD_Cross'] = (data['K'] > data['D']) & (data['K'] > data['D']).diff()
data.dropna(inplace=True)

data['TP'] = data.Close+(data.ATR*1.05)

data['SL'] = data.Close*.99

data['Buy_signal'] = np.where((data.KD_Cross) &
                                (data.Close > data['EMA_'+str(emas[0])]) &
                                (data['EMA_'+str(emas[0])] > data['EMA_'+str(emas[1])]) &
                                (data['EMA_'+str(emas[1])] > data['EMA_'+str(emas[2])]), 1, 0)

selldates = []
outcome = []
for i in range(len(data)):
    if data.Buy_signal.iloc[i]:
        k = 1
        SL = data.SL.iloc[i]
        TP = data.TP.iloc[i]
        in_position = True
        while in_position:
            if i + k ==len(data):
                break
            looping_high = data.High.iloc[i+k]
            looping_low = data.Low.iloc[i+k]
            if looping_high >= TP:
                selldates.append(data.iloc[i+k].name)
                outcome.append('TP')
                in_position = False
            elif looping_low <= SL:
                selldates.append(data.iloc[i+k].name)
                outcome.append('SL')
                in_position = False
            k += 1

data.loc[selldates, 'Sell_signal'] = 1
data.loc[selldates, 'Outcome'] = outcome

data.Sell_signal = data.Sell_signal.fillna(0).astype(int)

# filter 1
mask = data[(data.Buy_signal == 1) | (data.Sell_signal == 1)]

# filter 2
mask2 = mask[(mask.Buy_signal.diff() == 1) | (mask.Sell_signal.diff() == 1)]

data[['Buy_signal', 'Sell_signal']] = 0
data['Outcome'] = np.NaN

data.loc[mask2.index.values, 'Buy_signal'] = mask2['Buy_signal']
data.loc[mask2.index.values, 'Sell_signal'] = mask2['Sell_signal']
data.loc[mask2.index.values, 'Outcome'] = mask2['Outcome']

first_buy = data[data.Buy_signal == 1].first_valid_index()

data = data.loc[str(first_buy):].copy()

test = data['2018-01-01 00:00:00': '2018-01-04 00:00:00']

def pointpos(x, type):
    if type == 'Buy':
        if x['Buy_signal']== 1:
            return x['Close']
        else:
            return np.nan
    elif type == 'Sell':
        if x['Sell_signal']== 1:
            return x['Close']
        else:
            return np.nan

test.loc[:,'buys'] = test.apply(lambda row: pointpos(row, 'Buy'), axis=1)
test.loc[:,'sells'] = test.apply(lambda row: pointpos(row, 'Sell'), axis=1)

class TradePro(Strategy):
    initsize = .1
    srsi_len = 13
    k = 3
    d = 3
    ema1_len = 7
    ema2_len = 13
    ema3_len = 34
    tp_mult = 1.2
    sl_mult = 1.2
    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        self.stoch_k = self.I(ta.momentum.stochrsi_k, pd.Series(close), self.srsi_len, self.k, self.k)
        self.stoch_d = self.I(ta.momentum.stochrsi_d, pd.Series(close), self.srsi_len, self.d, self.d)
        self.EMA1 = self.I(pta.ema, pd.Series(close), self.ema1_len)
        self.EMA2 = self.I(pta.ema, pd.Series(close), self.ema2_len)
        self.EMA3 = self.I(pta.ema, pd.Series(close), self.ema3_len)
        self.atr = self.I(pta.atr, pd.Series(high), pd.Series(low), pd.Series(close))
    
    def next(self):
        price = self.data.Close
        if (crossover(self.stoch_k, self.stoch_d) and
            price > self.EMA1 and
            self.EMA1 > self.EMA2 and
            self.EMA2 > self.EMA3):
                sl = price - self.atr*self.sl_mult
                tp = price + self.atr*self.tp_mult
                self.buy(sl = sl, tp = tp, size = self.initsize)
        elif crossover(self.stoch_d, self.stoch_k):
            self.position.close()

train = data['2018-01-01 00:00:00': '2019-01-01 00:00:00']
test2 = data['2019-02-01 00:00:00': '2020-02-01 00:00:00']

bt_train = Backtest(train, TradePro, cash = 100000, exclusive_orders=True)
output_train = bt_train.run()

train_SR = output_train['Sharpe Ratio']
train_SoR = output_train['Sortino Ratio']
train_CR = output_train['Calmar Ratio']

bt_test = Backtest(test2, TradePro, cash = 100000, exclusive_orders=True)
output_test = bt_test.run()

# %%time
# stats, heatmap = bt_train.optimize(srsi_len = [8, 9, 10, 11], ema1_len = [6,7,8,9], 
#                                    ema2_len = [11,12,13,14], ema3_len = [35,37,38,44],
#                                    maximize='Return [%]',
#                                    return_heatmap= True)

test_SR = output_test['Sharpe Ratio']
test_SoR = output_test['Sortino Ratio']
test_CR = output_test['Calmar Ratio']

dic = {'MAD': ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'],
       'Train': [train_SR, train_SoR, train_CR],
       'Test': [test_SR, test_SoR, test_CR]}

MAD = pd.DataFrame(dic)

class TradePro(Strategy):
    initsize = .1
    srsi_len = 8
    k = 3
    d = 3
    ema1_len = 7
    ema2_len = 14
    ema3_len = 44
    tp_mult = 1.3
    sl_mult = 1.1
    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        self.stoch_k = self.I(ta.momentum.stochrsi_k, pd.Series(close), self.srsi_len, self.k, self.k)
        self.stoch_d = self.I(ta.momentum.stochrsi_d, pd.Series(close), self.srsi_len, self.d, self.d)
        self.EMA1 = self.I(pta.ema, pd.Series(close), self.ema1_len)
        self.EMA2 = self.I(pta.ema, pd.Series(close), self.ema2_len)
        self.EMA3 = self.I(pta.ema, pd.Series(close), self.ema3_len)
        self.atr = self.I(pta.atr, pd.Series(high), pd.Series(low), pd.Series(close))
    
    def next(self):
        price = self.data.Close
        if (crossover(self.stoch_k, self.stoch_d) and
            price > self.EMA1 and
            self.EMA1 > self.EMA2 and
            self.EMA2 > self.EMA3):
                sl = price - self.atr*self.sl_mult
                tp = price + self.atr*self.tp_mult
                self.buy(sl = sl, tp = tp, size = self.initsize)
        elif crossover(self.stoch_d, self.stoch_k):
            self.position.close()

bt_train = Backtest(train, TradePro, cash = 100000, exclusive_orders=True)
output_train = bt_train.run()

bt_test = Backtest(test2, TradePro, cash = 100000, exclusive_orders=True)
output_test = bt_test.run()
