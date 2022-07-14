"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Technical Analysis                                                                         -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: @Rub27182n | @if722399 | @hectoronate                                                       -- #
# -- license: TGNU General Public License v3.0                                                           -- #
# -- repository: https://github.com/Rub27182n/myst_proyecto_eq6.git                                      -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import ta
import data as dt
import numpy as np
import pandas as pd
import pandas_ta as pta
import visualizations as vz
from pandas_ta import Imports
from numpy import nan as npNaN
from pandas_ta.utils import get_drift, get_offset, verify_series

# ------------------------- Stochastic RSI ----------------------------------------------------------------------------
def stochrsi_d(close: pd.Series, 
               window: int = 14, 
               smooth1: int = 3, 
               smooth2: int = 3) -> pd.Series:
    
    """Stochastic Relative Strenght Index D (SRSId)
    The SRSI takes advantage of both momentum indicators in order to create a more 
    sensitive indicator that is attuned to a specific security's historical performance
    rather than a generalized analysis of price change.
    
    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period
        smooth1(int): moving average of Stochastic RSI
        smooth2(int): moving average of %K

    Returns:
            pandas.Series: New feature generated.

    References:
        [1] https://www.investopedia.com/terms/s/stochrsi.asp
    """

    return ta.momentum.StochRSIIndicator(
        close=close, 
        window=window, 
        smooth1=smooth1, 
        smooth2=smooth2).stochrsi_d()

def stochrsi_k(close: pd.Series,window: int = 14,smooth1: int = 3,smooth2: int = 3,fillna: bool = False,) -> pd.Series:
    
    """Stochastic Relative Strenght Index K (SRSId)
    The SRSI takes advantage of both momentum indicators in order to create a more 
    sensitive indicator that is attuned to a specific security's historical performance
    rather than a generalized analysis of price change.

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period
        smooth1(int): moving average of Stochastic RSI
        smooth2(int): moving average of %K

    Returns:
            pandas.Series: New feature generated.

    References:
        [1] https://www.investopedia.com/terms/s/stochrsi.asp
    """

    return ta.momentum.StochRSIIndicator(
        close=close, 
        window=window, 
        smooth1=smooth1, 
        smooth2=smooth2).stochrsi_k()

# --------------------------------ATR--------------------------------------------------------------
def atr(high, 
        low, 
        close, 
        length=None, 
        mamode=None, 
        talib=None, 
        drift=None, 
        offset=None, **kwargs):

    """Average True Range (ATR)
    Averge True Range is used to measure volatility, especially volatility caused by
    gaps or limit moves.

    Args:
        high (pd.Series): Series of 'high's
        low (pd.Series): Series of 'low's
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 14
        mamode (str): See ```help(ta.ma)```. Default: 'rma'
        talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
            version. Default: True
        drift (int): The difference period. Default: 1
        offset (int): How many periods to offset the result. Default: 0

    Returns:
        pd.Series: New feature generated.

    References:
        https://www.tradingview.com/wiki/Average_True_Range_(ATR)
"""
    # Validate arguments
    length = int(length) if length and length > 0 else 14
    mamode = mamode.lower() if mamode and isinstance(mamode, str) else "rma"
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if high is None or low is None or close is None: return

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import ATR
        atr = ATR(high, low, close, length)
    else:
        tr = pta.true_range(high=high, low=low, close=close, drift=drift)
        atr = pta.overlap.ma(mamode, tr, length=length)

    percentage = kwargs.pop("percent", False)
    if percentage:
        atr *= 100 / close

    # Offset
    if offset != 0:
        atr = atr.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        atr.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        atr.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    atr.name = f"ATR{mamode[0]}_{length}{'p' if percentage else ''}"
    atr.category = "volatility"

    return atr


#----------------------------- EMA ---------------------------------------------------------------------------------------------

def ema(close, 
        length=None, 
        talib=None, 
        offset=None, **kwargs):

    """Exponential Moving Average (EMA)
    The Exponential Moving Average is more responsive moving average compared to the
    Simple Moving Average (SMA).  The weights are determined by alpha which is
    proportional to it's length.  There are several different methods of calculating
    EMA.  One method uses just the standard definition of EMA and another uses the
    SMA to generate the initial value for the rest of the calculation.
    
    Args:
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 10
        talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
            version. Default: True
        offset (int): How many periods to offset the result. Default: 0

    Returns:
        pd.Series

    References:
        [1] https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_averages
        [2] https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
"""

    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    adjust = kwargs.pop("adjust", False)
    sma = kwargs.pop("sma", True)
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None: return

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import EMA
        ema = EMA(close, length)
    else:
        if sma:
            close = close.copy()
            sma_nth = close[0:length].mean()
            close[:length - 1] = npNaN
            close.iloc[length - 1] = sma_nth
        ema = close.ewm(span=length, adjust=adjust).mean()

    # Offset
    if offset != 0:
        ema = ema.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        ema.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        ema.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    ema.name = f"EMA_{length}"
    ema.category = "overlap"

    return ema


#---------------------------------VWAP--------------------------------------------------------
def vwap(high, low, close, volume, anchor=None, offset=None, **kwargs):
    """Indicator: Volume Weighted Average Price (VWAP)"""

    anchor = anchor.upper() if anchor and isinstance(anchor, str) and len(anchor) >= 1 else "D"
    offset = pta.utils.get_offset(offset)
    typical_price = pta.hlc3(high=high, low=low, close=close)

    # Calculate Result
    wp = typical_price * volume
    vwap  = wp.groupby(wp.index.to_period(anchor)).cumsum()
    vwap /= volume.groupby(volume.index.to_period(anchor)).cumsum()

    # Offset
    if offset != 0:
        vwap = vwap.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        vwap.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        vwap.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    vwap.name = f"VWAP_{anchor}"
    vwap.category = "overlap"

    return vwap


#-------------------------------PIVOTS--------------------------------------------------------
def pivots(_open, high, low, close, anchor=None, method=None):

    anchor = anchor.upper() if anchor and isinstance(anchor, str) and len(anchor) >= 1 else "D"
    method_list = ["traditional", "fibonacci", "woodie", "classic", "camarilla"]
    method = method if method in method_list else "traditional"
    date = close.index

    freq = pd.infer_freq(date)
    df = pd.DataFrame(
        index=date,
        data={"open": _open.values, "high": high.values, "low": low.values, "close": close.values},
    )

    if freq is not anchor:
        a = pd.DataFrame()
        a["open"] = df["open"].resample(anchor).first()
        a["high"] = df["high"].resample(anchor).max()
        a["low"] = df["low"].resample(anchor).min()
        a["close"] = df["close"].resample(anchor).last()
    else:
        a = df

    # Calculate the Pivot Points
    if method == "traditional":
        a["p"] = (a.high.values + a.low.values + a.close.values) / 3

        a["bc"] = (a.high.values + a.low.values ) / 2
        a["tc"] = (2 * a.p.values) - a.bc.values
        a["rng"] = abs(a.tc.values-a.bc.values)/a.p.values*100

        a["s1"] = (2 * a.p.values) - a.high.values
        a["s2"] = a.p.values - (a.high.values - a.low.values)
        a["s3"] = a.p.values - (a.high.values - a.low.values) * 2
        a["r1"] = (2 * a.p.values) - a.low.values
        a["r2"] = a.p.values + (a.high.values - a.low.values)
        a["r3"] = a.p.values + (a.high.values - a.low.values) * 2

    elif method == "fibonacci":
        a["p"] = (a.high.values + a.low.values + a.close.values) / 3
        a["pivot_range"] = a.high.values - a.low.values
        a["s1"] = a.p.values - 0.382 * a.pivot_range.values
        a["s2"] = a.p.values - 0.618 * a.pivot_range.values
        a["s3"] = a.p.values - 1 * a.pivot_range.values
        a["r1"] = a.p.values + 0.618 * a.pivot_range.values
        a["r2"] = a.p.values + 0.382 * a.pivot_range.values
        a["r3"] = a.p.values + 1 * a.pivot_range.values
        a.drop(["pivot_range"], axis=1, inplace=True)

    elif method == "woodie":
        a["pivot_range"] = a.high.values - a.low.values
        a["p"] = (a.high.values + a.low.values + a.open.values * 2) / 4
        a["s1"] = a.p.values * 2 - a.high.values
        a["s2"] = a.p.values - 1 * a.pivot_range.values
        a["s3"] = a.high.values + 2 * (a.p.values - a.low.values)
        a["s4"] = a.s3 - a.p.values
        a["r1"] = a.p.values * 2 - a.low.values
        a["r2"] = a.p.values + 1 * a.pivot_range.values
        a["r3"] = a.low.values - 2 * (a.high.values - a.p.values)
        a["r4"] = a.r3 + a.p.values
        a.drop(["pivot_range"], axis=1, inplace=True)

    elif method == "classic":
        a["p"] = (a.high.values + a.low.values + a.close.values) / 3
        a["pivot_range"] = a.high.values - a.low.values
        a["s1"] = a.p.values * 2 - a.high.values
        a["s2"] = a.p.values - 1 * a.pivot_range.values
        a["s3"] = a.p.values - 2 * a.pivot_range.values
        a["s4"] = a.p.values - 3 * a.pivot_range.values
        a["r1"] = a.p.values * 2 - a.low.values
        a["r2"] = a.p.values + 1 * a.pivot_range.values
        a["r3"] = a.p.values + 2 * a.pivot_range.values
        a["r4"] = a.p.values + 3 * a.pivot_range.values
        a.drop(["pivot_range"], axis=1, inplace=True)
    else:
        raise ValueError("Invalid method")

    if freq is not anchor:
        pivots_df = a.reindex(df.index, method="ffill")
    else:
        pivots_df = a

    pivots_df.drop(columns=["open", "high", "low", "close"], inplace=True)

    return pivots_df



