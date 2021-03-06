from turtle import width
from numpy import average
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from app import globals_variable, yahoo_finance
from app.modules import statistical_models
from plotly.subplots import make_subplots


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
    stats_vars = [
        round(df[variable].mean(),2), round(df[variable].std(),2), round(df[variable].max(),2), round(df[variable].min(),2)
        ]
    globals_variable.STATISTICS_VALUES =  ['${:,.2f}'.format(sv) for sv in stats_vars]
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
        

    fig = px.area(
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
    stats_vars=[round(df[variable].mean(),2), round(df[variable].std(),2), round(df[variable].max(),2), round(df[variable].min(),2)]
    globals_variable.STATISTICS_VALUES =  ['${:,.2f}'.format(sv) for sv in stats_vars]
    fig.update_layout(hovermode="x unified")
    return fig



def plot_monitor_candle_volume(ticker:str = globals_variable.COINS_SELECTION[-1]['ticker'], exchanges:str = "Dolar", interval: str ='1wk', variable:str='Close',  models=[]):


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
    stats_vars=[round(df[variable].mean(),2), round(df[variable].std(),2), round(df[variable].max(),2), round(df[variable].min(),2)]
    globals_variable.STATISTICS_VALUES =  ['${:,.2f}'.format(sv) for sv in stats_vars]
    fig.update_layout(hovermode="x unified")
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
                vertical_spacing=0.03, subplot_titles=('', 'Volume'), 
                row_width=[0.2, 0.7])

    
    fig.add_trace(
                    go.Scatter(
                                x = df.index,
                                y = df[variable],
                                mode ='lines',
                                fill='tozeroy',
                                name = 'Close',
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
    stats_vars = [round(df[variable].mean(),2), round(df[variable].std(),2), round(df[variable].max(),2), round(df[variable].min(),2)]
    globals_variable.STATISTICS_VALUES =  ['${:,.2f}'.format(sv) for sv in stats_vars]
    fig.update_layout(hovermode="x unified")
    return fig



def plot_lime(lime_df, px_theme='plotly_dark', contrib_var='LIME Weight'):

    fig = px.bar(
        lime_df, 
        x=contrib_var, 
        y='Feature',
        animation_frame='Date',
        orientation='h', 
        color=contrib_var, 
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


def error_bars(err_table, metric='RMSE'):

    fig = px.bar(
        err_table[err_table['Metric']==metric],
        x="Metric",
        y="Error",
        color="Model", 
        barmode="group",
        template='plotly_white',
        color_continuous_scale='viridis',
        )
    if metric in ['MAE', 'MSE', 'RMSE']:  
        fig.update_layout(yaxis_tickformat = '$')
    return fig



def plot_twt_wordcloud(text_df, sentiment='all', text_col='full_text', max_words=500, width=500, height=750):

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    import os
    from os import path
    import random
    from scipy.ndimage import gaussian_gradient_magnitude
    
    if sentiment == 'positive':
        path_to_logo = './sentiment_models/twt_green.png'
    elif sentiment == 'positive':
        path_to_logo = './sentiment_models/twt_red.png'
    else:
        path_to_logo = './sentiment_models/twt_logo.png'

    df2 = text_df.dropna(subset=[text_col], axis = 0)[text_col].copy()
    text = " ".join(s for s in df2)
    stopwords = set(STOPWORDS)
    stopwords.update(['t', 'https', 'co'])
    fura_color = np.array(Image.open(os.path.join(path_to_logo)))
    wordcloud = WordCloud(
            stopwords=stopwords,
            background_color="black",
            width=width, 
            height=height, 
            max_words=max_words,
            mask=fura_color,
            max_font_size=150,
            min_font_size=1).generate(text)
    # ploting it in a specific image
    image_colors = ImageColorGenerator(fura_color)

    # show
    fig, ax = plt.subplots(figsize=(10,10), dpi=60)
    ax.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
    ax.set_axis_off()

    img_path = './app/dashboard/assets/wcloud_' + sentiment + '.png'
    wordcloud.to_file(img_path)
    
    return 'wcloud_' + sentiment + '.png'


if '__main__'==__name__:
    pass
    # plot_monitor()