from turtle import width
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from app import globals_variable, yahoo_finance
from app.modules import statistical_models


def filter_test_df(test_df, ticker, model_id, pred_scope):

    test_df = test_df.query('Scope==@pred_scope & Ticker==@ticker & Model==@model_id')
    test_df = test_df[['Observed', 'Predicted']]

    return test_df.sort_index()


def plot_model_test(filtered_df: pd.DataFrame, px_theme: str ='plotly_dark'):

    fig = px.line(
            filtered_df,
            x=filtered_df.index, 
            y=filtered_df.columns,
            template=px_theme,
            labels={
                    'variable': ''
                },
            )
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    fig.update_yaxes(title_text = 'Close Price')
    fig.update_layout(
        yaxis_tickformat = '$',
        title={
            #'text': "Model Performance",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
            #family="Courier New, monospace",
            size=25)
            },
        )
    fig.update_traces(hovertemplate='%{y:$,.2f}')
    fig.update_layout(hovermode="x unified")

    fig.update_xaxes(
        title_text = 'Date',
        rangeslider_visible = True
        )
    fig.update_layout(width=750, height=500)
    return fig


def plot_importance(importance_df: pd.DataFrame, px_theme: str ='plotly_dark'):

    fig = go.Figure()
    fig.update_xaxes(showgrid=True, gridwidth=0.1, gridcolor='DarkGrey')

    metrics_list = list(importance_df['Metric'].unique())

    for metric in metrics_list:
        vis = metric == 'mae'
        mdf = importance_df[importance_df['Metric']==metric]

        mdf = mdf.sort_values('Importance')
        mdf = mdf.query('Importance > 0')
        show = min(mdf.shape[0], 10)
        mdf = mdf.tail(show)
        fig.add_trace(
            go.Bar(
                orientation='h', 
                x = mdf['Importance'],
                y = mdf['Feature'],
                name = metric, 
                visible=vis
            )
        )
                
    buttons = []

    for i, metric in enumerate(metrics_list):
        args = [False] * len(metrics_list)
        args[i] = True
        
        button = dict(
                    label = metric.upper(),
                    method = "update",
                    args=[{"visible": args}])
        
        buttons.append(button)
        
    fig.update_layout(
        updatemenus=[dict(
                        active=0,
                        type="buttons",
                        direction = "left",
                        buttons=buttons,
                        x = 1,
                        y = 1,
                        xanchor = 'left',
                        yanchor = 'bottom'
                    )], 
        autosize=True,
    )
    fig.update_layout(
        title={
            #'text': "Error Perturbation",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
            #family="Courier New, monospace",
            #size=25
            )
            },
        template=px_theme
    )

    fig.update_yaxes(tickmode='linear')
    #fig.update_layout(updatemenus=[dict(font=dict(color='gray',), bgcolor='black')])
    fig.update_traces(marker_color='rgba(50, 171, 96, 0.6)')
    #fig.update_layout(width=750, height=500)

    return fig




def plot_monitor_candle(ticker:str = globals_variable.COINS_SELECTION[-1]['ticker'], exchanges:str = "Dolar", interval: str ='1wk', variable:str='Close', models=[]):
    status, df=yahoo_finance.market_value(symbol=ticker,interval=interval)
    if not status:
        return None
    
    if exchanges in ['BTC-USD','COP=X']:
        status, df_exchanges=yahoo_finance.market_value(symbol=exchanges,interval=interval)
        if not status:
            return None
        for x in ['Close', 'Open','High','Low','Adj Close']: 
            df[x]=df[x]/df_exchanges['Close'] if exchanges == 'BTC-USD' else df[x]*df_exchanges['Close']
        

    layout = go.Layout(
                        margin=go.layout.Margin(
                                l=0, #left margin
                                r=0, #right margin
                                b=0, #bottom margin
                                t=0, #top margin
                            ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                        )
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df[variable])], layout=layout)
    if models:
        for x in models:
            if x == 'BBANDS':
                bands= getattr(statistical_models,x)(df,variable)
                bands_name=['upperband', 'middleband', 'lowerband']
                for index, band in enumerate(bands):
                    fig.add_trace(
                            go.Scatter(
                                        x = band.index,
                                        y = band,
                                        mode = 'lines',
                                        name = bands_name[index],
                                        line = dict(shape = 'linear', width = 3, dash = 'dashdot'),
                                        connectgaps = True
                                    ))
            elif x == 'MACD':
                lines= getattr(statistical_models,x)(df,variable)
                lines_name=['Convergence', 'Divergence']
                for index, line in enumerate(lines):
                    fig.add_trace(
                            go.Scatter(
                                        x = line.index,
                                        y = line,
                                        mode = 'lines',
                                        name = lines_name[index],
                                        line = dict(shape = 'linear', width =2, dash = 'dashdot'),
                                        connectgaps = True
                                    ))
            else:
                df_result=getattr(statistical_models,x)(df,variable)
                fig.add_trace(
                                go.Scatter(
                                            x = df.index,
                                            y = df_result,
                                            mode = 'lines',
                                            name = x,
                                            line = dict(shape = 'linear', width = 2,  dash = 'dash'),
                                            connectgaps = True
                                        ))
    

    fig.update_layout(clickmode='event+select')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightBlue')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightBlue')
    return fig


def plot_monitor_line(ticker:str = globals_variable.COINS_SELECTION[-1]['ticker'], exchanges:str = "Dolar", interval: str ='1wk', variable:str='Close', models=[]):
    status, df=yahoo_finance.market_value(symbol=ticker,interval=interval)
    if not status:
        return None
    if exchanges in ['BTC-USD','COP=X']:
        status, df_exchanges=yahoo_finance.market_value(symbol=exchanges,interval=interval)
        if not status:
            return None
        for x in ['Close', 'Open','High','Low','Adj Close']: 
            df[x]=df[x]/df_exchanges['Close'] if exchanges == 'BTC-USD' else df[x]*df_exchanges['Close']
        

    fig = px.line(
            df,
            x=df.index, 
            y=df[variable],
            template='plotly_white',
            labels={
                    'variable': ''
                },
            )
    if models:
        for x in models:
            if x == 'BBANDS':
                bands= getattr(statistical_models,x)(df,variable)
                bands_name=['upperband', 'middleband', 'lowerband']
                for index, band in enumerate(bands):
                    fig.add_trace(
                            go.Scatter(
                                        x = band.index,
                                        y = band,
                                        mode = 'lines',
                                        name = bands_name[index],
                                        line = dict(shape = 'linear', width = 3, dash = 'dashdot'),
                                        connectgaps = True
                                    ))
            elif x == 'MACD':
                lines= getattr(statistical_models,x)(df,variable)
                lines_name=['Convergence', 'Divergence']
                for index, line in enumerate(lines):
                    fig.add_trace(
                            go.Scatter(
                                        x = line.index,
                                        y = line,
                                        mode = 'lines',
                                        name = lines_name[index],
                                        line = dict(shape = 'linear', width =2, dash = 'dashdot'),
                                        connectgaps = True
                                    ))
            else:
                df_result=getattr(statistical_models,x)(df,variable)
                fig.add_trace(
                                go.Scatter(
                                            x = df.index,
                                            y = df_result,
                                            mode = 'lines',
                                            name = x,
                                            line = dict(shape = 'linear', width = 2,  dash = 'dash'),
                                            connectgaps = True
                                        ))
    
    fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        clickmode='event+select'
                        )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightBlue')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightBlue')
    fig.update_traces(hovertemplate='%{y:$,.2f}')
    fig.update_layout(hovermode="x unified")

    fig.update_xaxes(
        title_text = 'Date',
        rangeslider_visible = True
        )
    return fig



def plot_monitor_candle_volume(ticker:str = globals_variable.COINS_SELECTION[-1]['ticker'], exchanges:str = "Dolar", interval: str ='1wk', variable:str='Close',  models=[]):
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    status, df=yahoo_finance.market_value(symbol=ticker,interval=interval)
    if not status:
        return None

    if exchanges in ['BTC-USD','COP=X']:
        status, df_exchanges=yahoo_finance.market_value(symbol=exchanges,interval=interval)
        if not status:
            return None
        for x in ['Close', 'Open','High','Low','Adj Close']: 
            df[x]=df[x]/df_exchanges['Close'] if exchanges == 'BTC-USD' else df[x]*df_exchanges['Close']
    

    # Create subplots and mention plot grid size
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                vertical_spacing=0.03, subplot_titles=('OHLC', 'Volume'), 
                row_width=[0.2, 0.7])

    # Plot OHLC on 1st row
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                    low=df["Low"], close=df[variable], name="OHLC"), 
                    row=1, col=1
    )

    # Bar trace for volumes on 2nd row without legend
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], showlegend=False), row=2, col=1)

    if models:
        for x in models:
            if x == 'BBANDS':
                bands= getattr(statistical_models,x)(df,variable)
                bands_name=['upperband', 'middleband', 'lowerband']
                for index, band in enumerate(bands):
                    fig.add_trace(
                            go.Scatter(
                                        x = band.index,
                                        y = band,
                                        mode = 'lines',
                                        name = bands_name[index],
                                        line = dict(shape = 'linear', width = 3, dash = 'dashdot'),
                                        connectgaps = True
                                    ))
            elif x == 'MACD':
                lines= getattr(statistical_models,x)(df,variable)
                lines_name=['Convergence', 'Divergence']
                for index, line in enumerate(lines):
                    fig.add_trace(
                            go.Scatter(
                                        x = line.index,
                                        y = line,
                                        mode = 'lines',
                                        name = lines_name[index],
                                        line = dict(shape = 'linear', width =2, dash = 'dashdot'),
                                        connectgaps = True
                                    ))
            else:
                df_result=getattr(statistical_models,x)(df,variable)
                fig.add_trace(
                                go.Scatter(
                                            x = df.index,
                                            y = df_result,
                                            mode = 'lines',
                                            name = x,
                                            line = dict(shape = 'linear', width = 2,  dash = 'dash'),
                                            connectgaps = True
                                        ))
    
    
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        clickmode='event+select',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                        )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightBlue')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightBlue')
    return fig

def plot_monitor_line_volume(ticker:str = globals_variable.COINS_SELECTION[-1]['ticker'], exchanges:str = "Dolar", interval: str ='1wk', variable:str='Close', models=[]):
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    status, df=yahoo_finance.market_value(symbol=ticker,interval=interval)
    if not status:
        return None

    if exchanges in ['BTC-USD','COP=X']:
        status, df_exchanges=yahoo_finance.market_value(symbol=exchanges,interval=interval)
        if not status:
            return None
        for x in ['Close', 'Open','High','Low','Adj Close']: 
            df[x]=df[x]/df_exchanges['Close'] if exchanges == 'BTC-USD' else df[x]*df_exchanges['Close']
     

    # Create subplots and mention plot grid size
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                vertical_spacing=0.03, subplot_titles=('Line', 'Volume'), 
                row_width=[0.2, 0.7])

    
    fig.add_trace(
                    go.Scatter(
                                x = df.index,
                                y = df[variable],
                                mode ='lines',
                                name = 'line',
                                #line = dict(shape = 'linear', color = 'rgb(100, 10, 100)', width = 1, dash = 'dash'),
                                connectgaps = True
                            ), row=1, col=1
                )

    # Bar trace for volumes on 2nd row without legend
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], showlegend=False), row=2, col=1)

    if models:
        for x in models:
            if x == 'BBANDS':
                bands= getattr(statistical_models,x)(df,variable)
                bands_name=['upperband', 'middleband', 'lowerband']
                for index, band in enumerate(bands):
                    fig.add_trace(
                            go.Scatter(
                                        x = band.index,
                                        y = band,
                                        mode = 'lines',
                                        name = bands_name[index],
                                        line = dict(shape = 'linear', width = 3, dash = 'dashdot'),
                                        connectgaps = True
                                    ))
            elif x == 'MACD':
                lines= getattr(statistical_models,x)(df,variable)
                lines_name=['Convergence', 'Divergence']
                for index, line in enumerate(lines):
                    fig.add_trace(
                            go.Scatter(
                                        x = line.index,
                                        y = line,
                                        mode = 'lines',
                                        name = lines_name[index],
                                        line = dict(shape = 'linear', width = 2, dash = 'dashdot'),
                                        connectgaps = True
                                    ))
            else:
                df_result=getattr(statistical_models,x)(df,variable)
                fig.add_trace(
                                go.Scatter(
                                            x = df.index,
                                            y = df_result,
                                            mode = 'lines',
                                            name = x,
                                            line = dict(shape = 'linear', width = 2,  dash = 'dash'),
                                            connectgaps = True
                                        ))
    

    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        clickmode='event+select',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                        )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightBlue')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightBlue')
    return fig



def plot_lime(lime_df, px_theme='plotly_dark'):

    fig = px.bar(
        lime_df, 
        x='LIME Weight', 
        y='Feature',
        animation_frame='Date',
        orientation='h', 
        color='LIME Weight', 
        template=px_theme,
        title="",
        color_continuous_scale='viridis',
        )
    
    fig.update_traces(
        hovertemplate="<br>".join([
            "Feature: %{y}",
            "Contribution: $%{x:,.2f}"])
            )
    fig.update_layout(
            xaxis_tickformat = '$',
            title={
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(
                #family="Courier New, monospace",
                size=25)
                },
            )
    
    return fig


if '__main__'==__name__:
    pass
    # plot_monitor()