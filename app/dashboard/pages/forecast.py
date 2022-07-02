import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback
from matplotlib.pyplot import plot, text
import pandas as pd
import datetime as dt
from app.dashboard.crypto_plots import plot_model_test, plot_importance, plot_lime
from dash.dependencies import Input, Output
from app.modules.models_meta import pred_models
from app.api import yahoo_finance, GoogleTrends
from app.modules.lstm import get_prediction

dash.register_page(__name__)


# padding for the page content
CONTENT_STYLE = {
    "margin-left": "12rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

# Read predictions obtained during model testing
preds_df = pd.read_csv('./app/dashboard/test_models/predictions.csv', parse_dates=['Date'], index_col='Date')
ft_importance_df = pd.read_csv('./app/dashboard/test_models/ft_importance.csv')
lime_df = pd.read_csv('./app/dashboard/test_models/lime.csv')

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
                                options=[],
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
                                html.Div(
                                    id='prediction-container',
                                    className='silver-container',
                                    children=[
                                        html.H3('The last close is: '),
                                        html.P(id='last-closing-price', children=[]),
                                        dcc.Interval(id='update-price-interval', interval=60*1000, n_intervals=0),
                                        html.P(id='prediction-test', children=[]),
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
                                            html.Img(className='feature-img-on', src=dash.get_asset_url('autoreg_iconxs.png')),
                                            className='feature-container',
                                            width={"size": 3, "offset": 0, 'order': 'first'}    
                                        ),
                                        dbc.Col(
                                            html.Img(className='feature-img-on', src=dash.get_asset_url('volume.png')),
                                            className='feature-container',
                                            width={"size": 3, "offset": 0, 'order': 2}
                                        ),
                                        dbc.Col(
                                            html.Img(className='feature-img-on', src=dash.get_asset_url('int_rate.png')),
                                            className='feature-container',
                                            width={"size": 3, "offset": 0, 'order': 3}
                                        ),
                                        dbc.Col(
                                            html.Img(className='feature-img-on', src=dash.get_asset_url('google_logo.png')),
                                            className='feature-container',
                                            width={"size": 3, "offset": 0, 'order': 'last'}
                                        ),
                                        ],
                                        #width={"size": 2, "offset": 0, 'order': 'first'},
                                        className='features-container'
                            ),
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
                    ],
                    width={"size": 7, "offset": 0, 'order':'last'},
                )
            ]
        ),
        html.Hr(),
        html.H1('EXPLAINABLE MODELS'),
        html.H2('We Love Black. Just Not BLACK BOX!'),
        dbc.Row(
            children=[
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
                            children=[dcc.Graph(id='importance-plot', className='graph-container')],
                            type='circle',
                            color='#A0A0A0'
                        )
                    ],
                    width={"size": 8, "offset": 0, 'order':'last'},
                )
            ],
            style={'padding': '1rem 1rem 1rem 1rem'}
        ),
        dbc.Row(
            children=[
                dbc.Col(
                    children=[
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
                    ],
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
    Output('time-dropdown', 'options'),
    Output('time-dropdown', 'value'),
    [Input('coin-dropdown', 'value'), Input('model-dropdown', 'value')],
)
def populate_time_ddown(sel_coin, sel_model):
    pred_dff = preds_df.query('(Coin == @sel_coin) & (Model == @sel_model)')
    times_list = sorted(pred_dff['Scope'].unique())
    time_opts = [{'label': t, 'value': t} for t in times_list]
    time_value = times_list[0]
    return time_opts, time_value

@callback(
    Output('test-plot', 'figure'),
    Output('importance-plot', 'figure'),
    Output('lime-plot', 'figure'),
    [Input('coin-dropdown', 'value'), Input('model-dropdown', 'value'), Input('time-dropdown', 'value')],
)
def update_models_plots(sel_coin, sel_model, sel_time):
    
    model_preds = preds_df.query('(Coin == @sel_coin) & (Model == @sel_model) & (Scope == @sel_time)')
    model_preds = model_preds[['Observed', 'Predicted']]

    ft_importance = ft_importance_df.query('(Coin == @sel_coin) & (Model == @sel_model) & (Scope == @sel_time)')
    ft_importance = ft_importance[['Feature', 'Importance', 'Metric']]
    ft_importance.sort_values(by=['Metric', 'Importance'], inplace=True)

    sublime_df = lime_df.query('(Coin == @sel_coin) & (Model == @sel_model) & (Scope == @sel_time)')

    fig_test = plot_model_test(model_preds, px_theme='plotly_white')
    fig_imp = plot_importance(ft_importance, px_theme='plotly_white')
    fig_lime = plot_lime(sublime_df, px_theme='plotly_white')

    return fig_test, fig_imp, fig_lime


@callback(
    Output('about-model', 'children'),
    [Input('coin-dropdown', 'value'), Input('model-dropdown', 'value'), Input('time-dropdown', 'value')],
)
def update_about_model(sel_coin, sel_model, sel_time):
    try:
        text = pred_models[sel_coin][sel_model][sel_time]['about']
    except KeyError:
        text = html.P('No information was found on this model.')
    return text


""" @callback(
    Output('prediction-test', 'children'),
    [Input('coin-dropdown', 'value'), Input('model-dropdown', 'value'), Input('time-dropdown', 'value')],
)
def predict_price(sel_coin, sel_model, sel_time):
    print('ATTEMPTING PRICE PREDICTION !!!')
    text = get_prediction(pred_models, sel_coin, sel_model,sel_time,'./app/dashboard/test_models/', './models/')
    import time
    time.sleep(60)
    return text """


@callback(
    Output('last-closing-price', 'children'),
    [Input('coin-dropdown', 'value'),  Input('update-price-interval', 'n_intervals')]
)
def update_close(sel_coin, n):
    today = dt.datetime.today() - dt.timedelta(1)
    usd_ticker = str(sel_coin)[:3] + '-USD'
    status, yahoo_df = yahoo_finance.market_value(usd_ticker, hist=today, interval='1d')
    if status:
        close = yahoo_df['Close'].values[-1]
        return close
    else:
        return 'Unable to retrieve current price. Retrying...'



    """
    Cryptocurrency prices might be affected by many yet unknown factors. 
    Some of our models are purely statistical and they provide us with valuable 
    insights about the autoregressive nature of crypto prices, for which we report 
    an autocorrelation plot below. On the other hand, for our deep learning models, 
    we wanted to consider both the time structure of the data, and other 
    sources of information that the team identified as potential predictors.
    It is difficult to measure feature importance in Recurrent Neural Networks,
    such as the ones we have used. However, we built our own measure of feature importance as
    the error perturbation. That is, we measured by how much the model error increased when
    the data on each one of the feature-lag combinations was shuffled. A greater error perturbation,
    suggests a greater feature importance. Below we report top 20 feature-lag combinations for the
    deep learning models following this criteria.
    """,