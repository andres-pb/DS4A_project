from dash import Output, Input, ctx
from app import globals_variable
from app.dashboard.crypto_plots import plot_monitor_candle, plot_monitor_candle_volume, plot_monitor_line, plot_monitor_line_volume
def register_monitor_callbacks(app):
    
    interval_options={
                        'btn_nclicks_3':'1d',
                        'btn_nclicks_4':'5d',
                        'btn_nclicks_5':'1wk',
                        'btn_nclicks_6':'1mo',
                        'btn_nclicks_7':'3mo',
                        'btn_nclicks_8':'6mo',
                        'btn_nclicks_9':'1y',
                             }
    variable_options={
                        'btn_nclicks_1':'Close',
                        'btn_nclicks_2':'Adj Close',
                    }
    statistical_models={
                        'Volume':'',
                        'Mean Price':'mean_price',
                        'Rolling Mean Price':'rolling_mean',
                        'Rolling Standard Deviation':'rolling_std',
                        'Exponential WMA':'exponential_wma',
                        'Bollinger Bands':'BBANDS',
                        'Average Directional Movement Index':'ADX',
                        'Average True Range':'ATR',
                        'Moving Average Convergence/Divergence':'MACD',
                         }
    right_button_values=dict()
    left_button_values=dict()
    
    
    def update_class_right(btn_nclicks_1, btn_nclicks_2):
        list_btn=['btn_nclicks_1', 'btn_nclicks_2']
        result=[None for x in range(len(list_btn))]
        variable='Close'
        for index, value in enumerate(list_btn):
            if ctx.triggered_id == value:
                variable=variable_options[value]
                if right_button_values.get(value,None):right_button_values[value]=None
                else:
                    result[index]=right_button_values[value]={'background-color': 'white'}
                    right_button_values[list_btn[index-1]]=None
        if not any(result):
            result[0]={'background-color': 'white'}
        return result, variable


    

    def update_class_left(btn_nclicks_3, btn_nclicks_4, btn_nclicks_5, btn_nclicks_6, btn_nclicks_7, btn_nclicks_8, btn_nclicks_9):
        list_btn=['btn_nclicks_3', 'btn_nclicks_4', 'btn_nclicks_5', 'btn_nclicks_6', 'btn_nclicks_7', 'btn_nclicks_8', 'btn_nclicks_9']
        result=[None for x in range(len(list_btn))]
        interval='1wk'
        for index, value in enumerate(list_btn):
            if ctx.triggered_id == value:
                interval=interval_options[value]
                if left_button_values.get(value,None):
                    left_button_values[value]=None
                    break
                else:
                    for k, v in left_button_values.items():
                        if v:
                            left_button_values[k]=None
                            break
                    result[index] =left_button_values[value]={'background-color': 'white'}
        if not any(result):
            result[2]={'background-color': 'white'}

        return result, interval


    @app.callback([
                        Output('btn_nclicks_1', 'style'),
                        Output('btn_nclicks_2', 'style'),
                        Output('btn_nclicks_3', 'style'), 
                        Output('btn_nclicks_4', 'style'), 
                        Output('btn_nclicks_5', 'style'), 
                        Output('btn_nclicks_6', 'style'), 
                        Output('btn_nclicks_7', 'style'), 
                        Output('btn_nclicks_8', 'style'), 
                        Output('btn_nclicks_9', 'style'),
                        Output('monitor_plot', 'figure')],
                    [
                        Input('dropdown_coins', 'value'), 
                        Input('dropdown_exchanges', 'value'), 
                        Input('dropdown-chart', 'value'), 
                        Input('checklist_view', 'value'),
                        Input('btn_nclicks_1', 'n_clicks'),
                        Input('btn_nclicks_2', 'n_clicks'),
                        Input('btn_nclicks_3', 'n_clicks'), 
                        Input('btn_nclicks_4', 'n_clicks'), 
                        Input('btn_nclicks_5', 'n_clicks'), 
                        Input('btn_nclicks_6', 'n_clicks'), 
                        Input('btn_nclicks_7', 'n_clicks'), 
                        Input('btn_nclicks_8', 'n_clicks'), 
                        Input('btn_nclicks_9', 'n_clicks')
                    ]
                )
    def monitor_plot_callback(
                                ticker, exchanges, chart, view,
                                btn_nclicks_1, btn_nclicks_2, 
                                btn_nclicks_3, btn_nclicks_4, btn_nclicks_5, btn_nclicks_6, btn_nclicks_7, btn_nclicks_8, btn_nclicks_9):
        print(view)
        models=list()
        for i in view:
            if i != 'Volume':
                models.append(statistical_models[i])
        print(models)
        right_result, variable =update_class_right(btn_nclicks_1, btn_nclicks_2)
        result, interval = update_class_left(btn_nclicks_3, btn_nclicks_4, btn_nclicks_5, btn_nclicks_6, btn_nclicks_7, btn_nclicks_8, btn_nclicks_9)
        if chart=='line':
            if 'Volume' in view:
                return *right_result, *result, plot_monitor_line_volume(ticker=ticker, exchanges=exchanges, interval=interval, models=models)
            return *right_result, *result, plot_monitor_line(ticker=ticker, exchanges=exchanges, interval=interval, models=models)
        if chart =='candle':
            if 'Volume' in view:
                return *right_result, *result, plot_monitor_candle_volume(ticker=ticker, exchanges=exchanges, interval=interval, models=models)
            return *right_result, *result, plot_monitor_candle(ticker=ticker, exchanges=exchanges, interval=interval, models=models)
        return None

  





    """left_button_values=dict()
    @app.callback([
                        Output('btn_nclicks_3', 'style'), 
                        Output('btn_nclicks_4', 'style'), 
                        Output('btn_nclicks_5', 'style'), 
                        Output('btn_nclicks_6', 'style'), 
                        Output('btn_nclicks_7', 'style'), 
                        Output('btn_nclicks_8', 'style'), 
                        Output('btn_nclicks_9', 'style')
                    ], 
                    [
                        Input('btn_nclicks_3', 'n_clicks'), 
                        Input('btn_nclicks_4', 'n_clicks'), 
                        Input('btn_nclicks_5', 'n_clicks'), 
                        Input('btn_nclicks_6', 'n_clicks'), 
                        Input('btn_nclicks_7', 'n_clicks'), 
                        Input('btn_nclicks_8', 'n_clicks'), 
                        Input('btn_nclicks_9', 'n_clicks')
                ])
    def update_class_left(btn_nclicks_3, btn_nclicks_4, btn_nclicks_5, btn_nclicks_6, btn_nclicks_7, btn_nclicks_8, btn_nclicks_9):
        list_btn=['btn_nclicks_3', 'btn_nclicks_4', 'btn_nclicks_5', 'btn_nclicks_6', 'btn_nclicks_7', 'btn_nclicks_8', 'btn_nclicks_9']
        result=[None for x in range(len(list_btn))]
        for index, value in enumerate(list_btn):
            if ctx.triggered_id == value:
                interval=interval_options[value]
                if left_button_values.get(value,None):
                    left_button_values[value]=None
                    break
                else:
                    for k, v in left_button_values.items():
                        if v:
                            left_button_values[k]=None
                            break
                    result[index] =left_button_values[value]={'background-color': 'white'}
        return result

    global ticker_store
    global view_store
    global chart_store
    global exchanges_store
    exchanges_store= None
    chart_store=None
    view_store=None
    ticker_store = None

    @app.callback(Output('monitor_plot', 'figure'), 
                    [
                        Input('dropdown_coins', 'value'), 
                        Input('dropdown_exchanges', 'value'), 
                        Input('dropdown-chart', 'value'), 
                        Input('checklist_view', 'value')
                    ]
                )
    def monitor_plot_callback(ticker, exchanges, chart, view):
        interval=globals()['interval'] if globals()['interval'] else '1wk'
        print(interval)
        if chart=='line':
            if 'Volume' in view:
                return plot_monitor_line_volume(ticker=ticker, exchanges=exchanges, interval=interval)
            return plot_monitor_line(ticker=ticker, exchanges=exchanges, interval=interval)
        if chart =='candle':
            if 'Volume' in view:
                return plot_monitor_candle_volume(ticker=ticker, exchanges=exchanges, interval=interval)
            return plot_monitor_candle(ticker=ticker, exchanges=exchanges, interval=interval)
        return None

  """
        