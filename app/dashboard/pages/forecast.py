import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, dash_table
from matplotlib.pyplot import plot, text
import pandas as pd
import datetime as dt
import sqlite3 as sql
from app.dashboard.crypto_plots import plot_model_test, plot_importance, plot_lime, error_bars
from dash.dependencies import Input, Output
from app.modules.models_meta import pred_models
from app.api import yahoo_finance, GoogleTrends
from app.modules.lstm import get_prediction
from app.modules.neural_prophet import get_npro_prediction

dash.register_page(__name__)


# padding for the page content
CONTENT_STYLE = {
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

# Connect to db
conn = sql.connect('database.db', detect_types=sql.PARSE_DECLTYPES)

# Read predictions obtained during model testing
preds_df = pd.read_sql('SELECT * FROM test_set_predictions', conn, index_col='Date')


layout = html.Div([
        # Menus and controls row
        dbc.Row([
                dbc.Col(
                    html.Div(
                        className='dropdown-fc',
                        children=[
                            html.H3('Cryptocurrency:'),
                            dcc.Dropdown(
                                id='coin-dropdown',
                                options=[{'label': c, 'value': c} for c in sorted(preds_df['Coin'].unique())],
                                value='BTC - Bitcoin',
                                clearable=False,                             
                            )
                        ],
                    ),
                    width={"size": 4, "offset": 0, 'order': 'first'},
                ),
                dbc.Col(
                    html.Div(
                        className='dropdown-fc',
                        children=[
                            html.H3('Model:'),
                            dcc.Dropdown(
                                id='model-dropdown',
                                options=[{'label': m, 'value': m} for m in sorted(preds_df['Model'].unique())],
                                value='Deep Learning LSTM',
                                clearable=False,
                            )
                        ],
                    ),
                    width={"size": 4, "offset": 0, 'order': 2},
                ),
                dbc.Col(
                    html.Div(
                        className='dropdown-fc',
                        children=[
                            html.H3('Time Scope:'),
                            dcc.Dropdown(
                                id='time-dropdown',
                                options=['1 day ahead'],
                                value='1 day ahead',
                                clearable=False,
                            )
                        ],
                    ),
                    width={"size": 4, "offset": 0, 'order': 'last'},
                )
        ],
        style={'margin-right': '20px','margin-top': '20px', 'margin-bottom': '50px'}),
        # Main content starts here in two columns        
        dbc.Row([
                # Left column of main content
                dbc.Col(
                    children=[
                        html.Div(
                            className='black-container',
                            children=[
                                dbc.Row([
                                    dbc.Col(
                                        children=[
                                            dcc.Loading(
                                                children=[
                                                    html.H6('CURRENT PRICE', className='white-subtitle'),
                                                    html.P(id='last-closing-price', children=[]),
                                                    dcc.Interval(id='update-price-interval', interval=180*1000, n_intervals=0),
                                                ],
                                                type='circle',
                                                color='#A0A0A0'
                                            ),
                                        ],
                                        className = 'quote-container',  
                                        width={"size": 6, "offset": 0, 'order': 'first'} ),
                                    dbc.Col(
                                        [dbc.Button(
                                            children=[
                                                html.P(className="fa fa-forward-fast"), 
                                                ' Forecast',
                                            ], 
                                            id='predict-btn',
                                            color='success',
                                            className="button-37 lg-block",
                                            size='large'
                                        ),], 
                                        className='main-btn-container',
                                         width={"size": 6, "offset": 0, 'order': 'last'} 
                                    ),
                                 ]),                              
                                html.Div(
                                    id='prediction-container',
                                    className='silver-container',
                                    children=[                                        
                                        html.H3('PREDICTION', className='white-subtitle'),
                                        dcc.Loading(
                                            [
                                                html.P(id='prediction-result', children=[]),
                                                html.P(id='prediction-validthru', children=[])
                                            ],
                                            type='circle',
                                            color='#A0A0A0'
                                        )
                                    ],
                                ),
                                html.Hr(),
                                html.H3('About the Model', style={'color': 'white'}),
                                html.P(
                                    id='about-model',
                                    className='text-just',
                                    children=['Loading model description...']
                                    ),
                                html.Hr(),
                                html.H3('Features', style={'color': 'white'}),
                                dbc.Row(
                                    children=[
                                        dbc.Col(
                                            html.Img(className='feature-img-on', 
                                                src=dash.get_asset_url('autoreg_iconxs.png'),
                                                id='autoreg-feature'),
                                            className='feature-container',
                                            width={"size": 3, "offset": 0, 'order': 'first'}    
                                        ),
                                        dbc.Tooltip(
                                            'Closing Price - Autoregressive',
                                            target="autoreg-feature",
                                            placement='bottom',
                                        ),
                                        dbc.Col(
                                            html.Img(className='feature-img-on', src=dash.get_asset_url('volume.png'),
                                                id='volume-feature'),
                                            className='feature-container',
                                            width={"size": 3, "offset": 0, 'order': 2}
                                        ),
                                        dbc.Tooltip(
                                            'Cryptocurrency Traded Volume',
                                            target="volume-feature",
                                            placement='bottom',
                                        ),
                                        dbc.Col(
                                            html.Img(className='feature-img-on', src=dash.get_asset_url('int_rate.png'),
                                                id='rf-feature'),
                                            className='feature-container',
                                            width={"size": 3, "offset": 0, 'order': 3}
                                        ),
                                        dbc.Tooltip(
                                            'Risk Free Rate - Treasury Bond Yield',
                                            target="rf-feature",
                                            placement='bottom',
                                        ),
                                        dbc.Col(
                                            html.Img(className='feature-img-on', src=dash.get_asset_url('google_logo.png'),
                                                id='gt-feature'),
                                            className='feature-container',
                                            width={"size": 3, "offset": 0, 'order': 'last'}
                                        ),
                                        dbc.Tooltip(
                                            'Google Trends interest level index',
                                            target="gt-feature",
                                            placement='bottom',
                                        ),
                                    ],
                                    className='features-container'
                                ),
                                html.Br(),
                                html.H6('Data Sources', style={'color': 'white'}),
                                dbc.Row(    
                                    children=[
                                        dbc.Col(
                                            html.Img(className='feature-img-on',
                                                src=dash.get_asset_url('yfinance_icon.jpg'),
                                                id='yahoo-source'),
                                            className='feature-container',
                                            width={"size": 3, "offset": 0, 'order': 'first'}    
                                        ),
                                        dbc.Tooltip(
                                            'Yahoo Finance - Live market data',
                                            target="yahoo-source",
                                            placement='bottom',
                                        ),
                                        dbc.Col(
                                            html.Img(className='feature-img-on', src=dash.get_asset_url('google_logo.png'),
                                                id='gt-source'),
                                            className='feature-container',
                                            width={"size": 3, "offset": 0, 'order': 2}
                                        ),
                                        dbc.Tooltip(
                                            'Google Trends - daily',
                                            target="gt-source",
                                            placement='bottom',
                                        ),
                                        dbc.Col(
                                            width={"size": 6, "offset": 0, 'order': 3}
                                        ),  
                                    ],
                                    className='features-container'
                                ),
                                html.Br(),
                                html.P(
                                    '*Disclaimer: this is an academic exercise, do not take it as official investment advise.',
                                    className='footnote'
                                )
                            ]
                        ),
                    ],
                    width={"size": 5, "offset": 0, 'order':'first'},
                ),
                # Right column of main content
                dbc.Col(
                    children = [
                        html.Div(
                            className='graph-cointainer',
                            children=[
                                html.H3('Model Performance', className='graph-title'),
                                html.P(
                                    """We set apart around 1 year of data during training and used it for model validation. 
                                       Below, you can see how the model prediction compared to the true price of the cryptocurrency.
                                    """,
                                       className='text-just graph-info'),
                                dcc.Loading(
                                    children=[dcc.Graph(id='test-plot',)],
                                    type='circle',
                                    color='#A0A0A0'
                                ),
                            ],
                            style={'padding': '1rem 1rem 1rem 1rem'}
                        ),
                        html.Div(
                            className='graph-cointainer',
                            children=[
                                html.H3('Model Comparison', className='graph-title'),
                                html.P(
                                    """Not sure what model to trust the most? Below you can compare them in terms of their error metrics
                                       when tested for the same dates.
                                    """,
                                       className='text-just graph-info'),
                                html.Br(),
                                html.Div(
                                    className='dropdown-fc-md',
                                    children=[
                                        html.H4('Error Metrics:'),
                                        dcc.Dropdown(
                                            id='error-dropdown',
                                            options=[
                                                {'label': 'Root Mean Square Error', 'value': 'RMSE'},
                                                {'label': 'Mean Absolute Percentage Error', 'value': 'MAPE'},
                                                {'label': 'Mean Square Error', 'value': 'MSE'},
                                                {'label': 'Mean Absolute Error', 'value': 'MAE'},
                                            ],
                                            value='RMSE',
                                            clearable=False                           
                                        ),
                                    ],
                                ),
                                dbc.Row(
                                    children = [
                                        dbc.Col(width={"size": 2, "offset": 0, 'order':'first'},),
                                        dbc.Col(
                                            dcc.Loading(
                                                children=[dcc.Graph(id='error-plot')],
                                                type='circle',
                                                color='#A0A0A0'
                                            ),
                                            width={"size": 7, "offset": 0, 'order':2},
                                        ),
                                        dbc.Col(width={"size": 3, "offset": 0, 'order':'last'},),   
                                    ]
                                ),
                                
                            ],
                            style={'padding': '1rem 1rem 1rem 1rem'}
                        ),

                    ],
                    width={"size": 7, "offset": 0, 'order':'last'},
                )
            ]
        ),
        html.Br(),
        html.Hr(),
        html.H1('EXPLAINABLE MODELS', className='header-extra'),
        dcc.Interval(id='update-question', interval=5*1000, n_intervals=0),
        html.H4(children=['What drives the price of crypto?'], id='questions-header', style={'text-align': 'center'}),
        dbc.Row([
            dbc.Col([], width={"size": 5, "offset": 0, 'order':'first'},className='background-box'),
            dbc.Col(
                children=[html.H2('We Love Black. Just Not BLACK BOX!', className='subheader-extra')],
                className='highlight-container',
                width={"size": 2, "offset": 0, 'order':2},
                ),
            dbc.Col([], width={"size": 5, "offset": 0, 'order':'last'}, className='background-box'),
            ]),         
        dbc.Row(
            children=[],
            style={'padding': '1rem 1rem 1rem 1rem'},
            id='err-pert-row'
        ),
        dbc.Row(
            children=[
                dbc.Col(
                    children=[],
                    id='lime-descr',
                    width={"size": 4, "offset": 0, 'order':'last'},
                ),
                dbc.Col(
                    children=[
                        dcc.Loading(
                            children=[dcc.Graph(id='lime-plot', className='graph-container',)],
                            type='circle',
                            color='#A0A0A0'
                        )],
                    width={"size": 8, "offset": 0, 'order':'first'},
                )
            ],
            style={'padding': '1rem 1rem 1rem 1rem'}
        ),
    ],
    style=CONTENT_STYLE
)


# -------------------------------------------- CALLBACKS ----------------------------------------------

@callback(
    Output('model-dropdown', 'options'),
    Output('model-dropdown', 'value'),
    [Input('coin-dropdown', 'value')],
)
def populate_mdl_ddown(sel_coin):
    pred_dff = preds_df.query('(Coin == @sel_coin)')
    mdls_list = sorted(pred_dff['Model'].unique())
    mdl_opts = [{'label': t, 'value': t} for t in mdls_list]
    if 'Neural Prophet' in mdls_list:
        mdl_value = 'Neural Prophet'
    else:
        mdl_value = mdls_list[0]
    return mdl_opts, mdl_value


@callback(
    Output('test-plot', 'figure'),
    [Input('coin-dropdown', 'value'), Input('model-dropdown', 'value'), Input('time-dropdown', 'value')],
)
def update_testing_plots(sel_coin, sel_model, sel_time):
    
    # First plot
    model_preds = preds_df.query('(Coin == @sel_coin) & (Model == @sel_model) & (Scope == @sel_time)')
    model_preds = model_preds[['Observed', 'Predicted']]
    fig_test = plot_model_test(model_preds, px_theme='plotly_white')  
 
    return fig_test


@callback(
    Output('error-plot', 'figure'),
    [Input('coin-dropdown', 'value'), Input('error-dropdown', 'value')],
)
def update_error_plots(sel_coin, sel_metric):
    
    conn = sql.connect('database.db', detect_types=sql.PARSE_DECLTYPES)

    # Error comparison plot
    query_metrics = """
        SELECT 
            Model,
            Metric,
            Error
        FROM error_metrics
        WHERE Coin = "{}"
    """.format(sel_coin)
    err_long = pd.read_sql(query_metrics, conn)
    err_long['Error'] = err_long['Error'].round(2)
    fig_err = error_bars(err_long, sel_metric)
    fig_err.update_layout(
        autosize=False,
        width=500,
        height=300,
    )
    
    return fig_err


@callback(
    Output('lime-descr', 'children'),
    [Input('model-dropdown', 'value')],
)
def update_lime_descr(sel_model):
    if sel_model == 'Neural Prophet':
        text = [
            html.H3('Feature Contribution Across Time', className='graph-title'),
            html.P(
                ["""
                    Neural Prophet summarizes the contribution of distinct features in an autorregressive term (ar), a trend component (trend) 
                    and the lags of the additional multivariate components that we have included (Google trends, risk free interest rate and volume).
                """],
                className='graph-info'
            ),
            html.P(
                ["""
                    Use the play and stop buttons to visualize how feature importance changes across time for this model.
                """],
                className='graph-info'
            ),
            html.P(
                ["""
                    Notice how, despite having a small contribution, sometimes negative, the additional features we have included are
                    not the main contributors to the daily price. The autoregressive head of the model prevails across time.
                """],
                className='graph-info'
            )
        ]
    else:
        text = [
            html.H3('Feature Contribution Across Time', className='graph-title'),
            html.P(
                ["""
                    Error perturbation might give us a general idea of those features
                    that are important to preserve an overall minimum error metrics. Using LIME - 
                    Local Interpretable Agnostic Explanations -, we can obtain a robust feature contribution measure
                    for each individual observation in the test set.
                """],
                className='graph-info'
            ),
            html.P(
                ["""
                    Use the play and stop buttons to visualize how feature importance changes across time for this model.
                """],
                className='graph-info'
            ),
            html.P(
                ["""
                    Notice how the most accurate models start to percieve higher contributions of features such as the interest rate, 
                    google trends and traded volume in times of negative price trends like in the the most recent history.
                """],
                className='graph-info'
            )
        ]
    return text


@callback(
    Output('lime-plot', 'figure'),
    [Input('coin-dropdown', 'value'), Input('model-dropdown', 'value'), Input('time-dropdown', 'value')],
)
def update_feature_plots(sel_coin, sel_model, sel_time):

    conn = sql.connect('database.db', detect_types=sql.PARSE_DECLTYPES)

    if sel_model == 'Neural Prophet':

        # feature contribution animation   
        query_fimp = """
            SELECT *
            FROM nprophet_ft_contribution
            WHERE Model = "{}"
                AND Ticker = "{}"
                AND Scope = "{}"
            ORDER BY Contribution
        """.format(sel_model, sel_coin[:3], sel_time)
        sublime_df = pd.read_sql(query_fimp, conn)
        fig_lime = plot_lime(sublime_df, contrib_var='Contribution', px_theme='plotly_white')

    else:
        # lime animation   
        query_lime = """
            SELECT *
            FROM lime_weigths
            WHERE Model = "{}"
                AND Coin = "{}"
                AND Scope = "{}"
            ORDER BY LIME_Weight
        """.format(sel_model, sel_coin, sel_time)

        sublime_df = pd.read_sql(query_lime, conn)
        sublime_df.rename(columns={col: col.replace('_', ' ') for col in sublime_df.columns.to_list()}, inplace=True)
        fig_lime = plot_lime(sublime_df, px_theme='plotly_white')

    return fig_lime


@callback(
    Output('about-model', 'children'),
    [Input('coin-dropdown', 'value'), Input('model-dropdown', 'value'), Input('time-dropdown', 'value')],
)
def update_about_model(sel_coin, sel_model, sel_time):

    if sel_model == 'Neural Prophet':
        text = [
            html.Div([
            html.P([
                html.A('Neural Prophet',
                    href='https://neuralprophet.com/html/index.html',
                    target="_blank",
                    style={'color': 'white'}),
                """
                    is a hybrid open source framework developed by Facebook in 2021 as the successor of Facebook Prophet.
                    It offers the flexibility of building model specifications that combine standard autoregressive linear models and/or
                    neural networks. Based on Pytorch, it offers incredible training speed, demonstrated accuracy across different types of forecasting problems,
                    and model explainability.
                """], 
                ),
            html.P("""We have leveraged the power of Neural Prophet to train, test and select a series of customized models that
                      accurately forecast the closing price of cryptocurrencies one day ahead, and considers 1 lag of the multivariate series
                      defined by our available features: the closing price itself, traded volume, interest risk free rates and the index of
                      internet interest on the cryptocurrency offered by Google Trends.
                """)
            ])
        ]
    else:
        try:
            text = pred_models[sel_coin][sel_model][sel_time]['about']
        except KeyError:
            text = html.P('Sorry, we could not find a description for this model.')
    return text


@callback(
    Output('prediction-result', 'children'),
    Output('prediction-validthru', 'children'),
    Output('predict-btn', 'n_clicks'),
    [Input('predict-btn', 'n_clicks'), Input('coin-dropdown', 'value'), Input('model-dropdown', 'value'), Input('time-dropdown', 'value')],
)
def predict_price(n, sel_coin, sel_model, sel_time):

    if n == 1:

        now = dt.datetime.today()
        pred_val = (now + dt.timedelta(1)).replace(hour=5, minute=0, second=0)
        pred_val_str = dt.datetime.strftime(pred_val, '%Y-%m-%d %H:%M')

        print('>>>> ATTEMPTING PRICE PREDICTION')

        if sel_model == 'Neural Prophet':
            forecast, ret = get_npro_prediction(
                sel_coin, 
                './app/dashboard/prod_models/',
                )
        elif sel_model in ['Deep Learning LSTM', 'Bidirectional LSTM']:   
            forecast, ret = get_prediction(
                pred_models, 
                sel_coin, 
                sel_model,
                sel_time,
                './app/dashboard/prod_models/',
                './app/dashboard/test_models/scalers/'
                )

        if ret > 0:
            clname, icon = 'price-quote', 'profit'  
        else:
            clname, icon = 'price-quote', 'loss'
        
        return [
            html.Div(
                [
                    html.P('${:,.2f}'.format(forecast), className=clname + ' ' + icon),
                ], 
                className='forecast-container'
            )], html.P(['As of {} local time UTC-5'.format(pred_val_str)], className='footnote'), None
    
    else:
        return html.P('Click the Forecast Button.'), '', None


@callback(
    Output('last-closing-price', 'children'),
    [Input('coin-dropdown', 'value'),  Input('update-price-interval', 'n_intervals')]
)
def update_close(sel_coin, n):
    today = dt.datetime.today() - dt.timedelta(1)
    usd_ticker = str(sel_coin)[:3] + '-USD'
    status, yahoo_df = yahoo_finance.market_value(usd_ticker, hist=today, interval='1d')
    if status:
        time_upd = dt.datetime.strftime(dt.datetime.now(), '%H:%M')
        close = yahoo_df['Close'].values[-1]
        return [
            html.P('${:,.2f}'.format(close), className='price-quote',), 
            html.Span('Last updated at {}'.format(time_upd), className='footnote')
            ]
    else:
        return html.P('Price not available...')


@callback(
    Output('questions-header', 'children'),
    [Input('update-question', 'n_intervals'), Input('questions-header', 'children')]
)
def update_close(n, previous):
    questions_list = [
        'What drives the price of crypto?',
        'How Relevant is Price History?',
        'Do Main Drivers of Price Change Across Time?',
        'Do Internet Searches Affect Cryptocurrencies?',
        'Are Drivers The Same Across Cryptourrencies?',
        'How Does Crypto React to Macroeconomic Factors?',
    ]
    next_i = questions_list.index(previous[0]) + 1
    
    if next_i >= len(questions_list):
        return [questions_list[0]]
    else:
        return [questions_list[next_i]]


@callback(
    Output('err-pert-row', 'children'),
    [Input('coin-dropdown', 'value'), Input('model-dropdown', 'value'), Input('time-dropdown', 'value')],
)
def populate_ep_row(sel_coin, sel_model, sel_time):

    if sel_model == 'Neural Prophet':
        return []
    else:
        # Feature importance error perturb plot
        query_ftimp = """
            SELECT
                Metric,
                Feature,
                Importance
            FROM feature_importance_ep
            WHERE Model = "{}"
                AND Coin = "{}"
                AND Scope = "{}"
            ORDER BY Metric, Importance
        """.format(sel_model, sel_coin[:3], sel_time)

        conn = sql.connect('database.db', detect_types=sql.PARSE_DECLTYPES)
        ft_importance = pd.read_sql(query_ftimp, conn)
        fig_imp = plot_importance(ft_importance, px_theme='plotly_white')

        return [
                dbc.Col(
                    className='side-graph-desc',
                    children=[
                        html.H3('Error Perturbation', className='graph-title'),
                        html.P(
                            ["""
                            ERROR PERTURBATION as a measure of feature importance for Recurrent Neural Networks. 
                            It tells us by how much the model error increased when
                            the true values on each one of the feature-lag combinations were shuffled. 
                            A greater error perturbation, suggests a greater feature importance. 
                            We offer you the top feature-lag combinations for the deep learning 
                            models following this criteria. Toggle the buttons to change the error metric.
                            """],
                            className='graph-info'
                        )
                    ],
                    width={"size": 4, "offset": 0, 'order':'first'},
                ),
                dbc.Col(
                    children=[
                        dcc.Loading(
                            children=[dcc.Graph(id='importance-plot', className='graph-container', figure=fig_imp)],
                            type='circle',
                            color='#A0A0A0'
                        )
                    ],
                    width={"size": 8, "offset": 0, 'order':'last'},
                )
            ]