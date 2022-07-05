
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import data as dt
import pandas as pd
import numpy as np

# ob_ts = list(dt.ob_data.keys())
# l_ts = [pd.to_datetime(i_ts) for i_ts in ob_ts] # Modificar todos los valores de str a unidades de tiempo

# mid_prices = [(dt.ob_data[ob_ts[i]]['ask'][0] + dt.ob_data[ob_ts[i]]['bid'][0])/2 for i in range(len(ob_ts))]

#n_df = dt.ptm.drop(columns = ['Unnamed: 0', 'timestamp'])







# Leer datos
aave = dt.aave

# ----------- Promedio del volumen de los ultimos 14 precios: ----------- #

v = np.zeros(len(aave))
v[:len(aave)-14] = [np.mean(aave.iloc[0:,4][i:i+13]) for i in range(len(aave)-14)]
avg_volume = pd.DataFrame({'avg_volume':v})