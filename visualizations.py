
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Technical Analysis                                                                         -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
# -- author: @Rub27182n | @if722399 | @hectoronate                                                       -- #
# -- license: TGNU General Public License v3.0                                                           -- #
# -- repository: https://github.com/Rub27182n/myst_proyecto_eq6.git                                      -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------- Equity Return[%] Visualization ---------------- #

def Equity_viz(x,y,plot = False)->'Scatter Plot':

    
    """
    Funtion to create the Equity Return[%] Visualization

    Params
    ----------
    'x': Time index
    'y': Series of the equity return

    Returns
    --------
    Scatter plot

    References
    ----------
    https://plotly.com/python/time-series/
    """


    fig = make_subplots(specs=[[{"secondary_y": True}]])


    fig.add_trace(go.Scatter(
        name="BTC/USDT 15min",
        mode="lines",x=x,y=y,
        marker_symbol="star"
    ))

    fig.update_layout(
            title_text="Equity Return[%]"
        )
        
    # Set x-axis title
    fig.update_xaxes(title_text="TimeFrame 15min")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>BTC/USD", secondary_y=False)

    if plot:
        fig.show()
    else:
        pass


def strategy_test_viz(data, emas, plot = False):

    """
    Function to plot the strategy test with EMA, SRSI and ATR

    Params:
        data: pandas.DataFrame with the following structure:
            Open:   float:  Open price
            High:   float:  highest price
            Low:    float:  lowest price
            Close:  float:  close price
            Volume: float:  volume
            K:      float:  SRSI K parameter
            D:      float:  SRSI D parameter
            EMA_n1: float:  EMA1 values
            EMA_n2: float:  EMA2 values
            EMA_n3: float:  EMA3 values
            ATR:    float:  ATR values
            KD_Cross:   Bool: indicates if there's a cross between K and D
            TP:     float:  Take Profit price
            SL:     float:  Stop Loss price
            Buy_signal:     Bool:   indicates buy signals
            Sell_signal:    Bool:   indicates sell signals
            Outcome:    str:    indicates if the buy was because of a TP or SL
            
        emas: list containing different values for EMA

    Returns:
        plotly graph that shows the following trading strategy:
            OHLC values in candlestick graph
            buy and entry points (green-red color)
            SRSI k/d (blue-red color)
            3 EMA levels (green-red-blue color)

    References:
        [1] https://plotly.com/python/financial-charts/
    """

    fig = go.Figure(
    data=[
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"]
        ),
        go.Scatter(x=data.index, y=data['K'], line=dict(color='blue', width=1.5),mode="lines", name="K", yaxis="y2"),
        go.Scatter(x=data.index, y=data['D'], line=dict(color='red', width=1.5),mode="lines", name="D", yaxis="y2"),
        go.Scatter(x=data.index, y=data['EMA_'+str(emas[0])], line=dict(color='blue', width=1), name='EMA_'+str(emas[0])),
        go.Scatter(x=data.index, y=data['EMA_'+str(emas[1])], line=dict(color='red', width=1), name='EMA_'+str(emas[1])),
        go.Scatter(x=data.index, y=data['EMA_'+str(emas[2])], line=dict(color='green', width=1), name='EMA_'+str(emas[2]))]
        ).update_layout(
        yaxis_domain=[0.3, 1],
        yaxis2={"domain": [0, 0.20]},
        xaxis_rangeslider_visible=False,
        showlegend=False,
        )

    fig.add_scatter(x=data.index, y=data['sells'], mode="markers",
                    marker=dict(size=10, color="red"),
                    name="sell")
    fig.add_scatter(x=data.index, y=data['buys'], mode="markers",
                    marker=dict(size=10, color="green"),
                    name="buy")
    if plot:
        fig.show()
    else:
        pass






