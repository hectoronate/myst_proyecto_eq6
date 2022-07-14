"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Technical Analysis                                                                         -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: @Rub27182n | @if722399 | @hectoronate                                                       -- #
# -- license: TGNU General Public License v3.0                                                           -- #
# -- repository: https://github.com/Rub27182n/myst_proyecto_eq6.git                                      -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# Libraries
# from pathlib import Path
# from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import pandas as pd

#!pip install python-binance pandas mplfinance

# Get Binance data
# apikey = 'QddE1UndJCJ48XvSgC6xKZV2tR365sqFCvpdX6mT6xHOcB9a7Ykqu30qyNn8znOe'
# secret = 'kyArU6V1ejdtrZHFpk5CmLpJG1Lk4inAMnoC8hgX53RtK3zDjdUOwV6VH63d6a5B'
# client = Client(apikey, secret)

# due to high download times, we stored the info in csv's, here's how we get the data:
# 1m_data = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_1MINUTE, '1 Jan 2018')
# 1m_df = pd.DataFrame(1m_data)
# 1m_df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume',
#                  'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore']
# 1m_df['Open Time'] = pd.to_datetime(1m_df['Open Time']/1000, unit='s')
# filepath1 = Path('files/BTCUSDT_1d.csv')  
# filepath1.parent.mkdir(parents=True, exist_ok=True)  
# 1m_df.to_csv(filepath1)

BTCUSDT = pd.read_csv('files/BTCUSDT_15m.csv')
BTCUSDT.drop(columns=['Unnamed: 0', 'Ignore', 'Close Time','Quote Asset Volume',
                      'Number of Trades', 'TB Base Volume','TB Quote Volume'], inplace = True)

BTCUSDT['Open Time'] = BTCUSDT['Open Time'].apply(pd.to_datetime)

