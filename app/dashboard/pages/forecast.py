import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback
import pandas as pd
from crypto_plots import plot_model_test, plot_importance
from dash.dependencies import Input, Output

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

layout = html.Div([
        # Menus and controls row
        dbc.Row([
                dbc.Col(
                    html.Div(
                        className='dropdown',
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
                        className='dropdown',
                        children=[
                            html.H3('Model:'),
                            dcc.Dropdown(
                                id='model-dropdown',
                                options=[{'label': m, 'value': m} for m in sorted(preds_df['Model'].unique())],
                                value='Deep Learning LSTM'
                            )
                        ],
                    ),
                    width={"size": 3, "offset": 0, 'order': 2},
                ),
                dbc.Col(
                    html.Div(
                        className='dropdown',
                        children=[
                            html.H3('Time Scope:'),
                            dcc.Dropdown(
                                id='time-dropdown',
                                options=[],
                                value='1 day ahead'
                            )
                        ],
                    ),
                    width={"size": 3, "offset": 0, 'order': 'last'},
                )
        ],
        style={'margin-right': '20px','margin-top': '5px', 'margin-bottom': '20px'}),
        # Main content starts here in two columns                 
        dbc.Row([
                dbc.Col(
                    'About model here',
                    width={"size": 5, "offset": 0, 'order':'first'},
                ),
                dbc.Col(
                    children = [
                        dcc.Loading(
                            children=[dcc.Graph(id='test-plot',)],
                            type='circle',
                            color='#A0A0A0'
                         ),
                        dcc.Loading(
                            children=[dcc.Graph(id='importance-plot',)],
                            type='circle',
                            color='#A0A0A0'
                        ),
                    ],
                    width={"size": 7, "offset": 0, 'order':'last'},
                )
            ]
        ),
    ],
    style=CONTENT_STYLE
)

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
    [Input('coin-dropdown', 'value'), Input('model-dropdown', 'value'), Input('time-dropdown', 'value')],
)
def update_models_plots(sel_coin, sel_model, sel_time):
    model_preds = preds_df.query('(Coin == @sel_coin) & (Model == @sel_model) & (Scope == @sel_time)')
    model_preds = model_preds[['Observed', 'Predicted']]

    ft_importance = ft_importance_df.query('(Coin == @sel_coin) & (Model == @sel_model) & (Scope == @sel_time)')
    ft_importance = ft_importance[['Feature', 'Importance', 'Metric']]

    fig_test = plot_model_test(model_preds, px_theme='plotly_white')
    fig_imp = plot_importance(ft_importance, px_theme='plotly_white')

    return fig_test, fig_imp
