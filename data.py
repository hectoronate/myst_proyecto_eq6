"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

"""
1) Descargar order books:
"""
# Importar Librerias
import pandas as pd
import json 
import numpy as np

def order_books(file_name):
    # Opening JSON file
    f = open(file_name)
    print(f)

    # Returns JSON object as a dictionary
    orderbooks_data = json.load(f)
    ob_data = orderbooks_data['bitfinex']

    # Drop Keys with none values
    ob_data = {i_key: i_value for i_key,i_value in ob_data.items() if i_value is not None}

    # Convert to DataFrame and rearange columns
    ob_data = {i_ob: pd.DataFrame(ob_data[i_ob])[['bid_size', 'bid', 'ask', 'ask_size']]
            if ob_data[i_ob] is not None else None for i_ob in list(ob_data.keys())}
    return ob_data

# Prueba
ob_data = order_books('files/orderbooks_05jul21.json')
#ob_data = order_books('orderbooks_05jul21.json')


def public_trades_metrics(pt_data):
    # -- Cantida de trades publicos que ocurren en 1 hora -- #
    #pt_data.drop('Unnamed: 0', inplace=True, axis = 1)
    a = pd.read_csv(pt_data)
    a.index = pd.to_datetime(a['timestamp']) # Convertir de str a timestamp

    return a



names = ['Open time','Open','High','Low','Close','Volume','Close time','Quote asset volume','Number of trades','Taker buy base asset volume','Taker buy quote asset volume','Ignore']
aave = pd.read_csv('files/AAVE/AAVEUSDT-15m-2022-06-01.csv', names = names)
aave2 = pd.read_csv('files/AAVE/AAVEUSDT-15m-2022-06-02.csv', names = names)