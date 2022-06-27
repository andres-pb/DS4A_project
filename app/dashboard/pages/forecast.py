import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback
import pandas as pd
from crypto_plots import plot_model_test
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

layout = html.Div([

    dbc.Row([
            dbc.Col(
                children=[
                    html.H3('Model:'),
                    dbc.DropdownMenu(
                        id='model_dropdown',

                        label='Select a Model',
                        children=[
                            dbc.DropdownMenuItem(
                                'LSTM',
                                id = 'BTC_LSTM_VGC_1D'
                                ),
                            dbc.DropdownMenuItem(
                                'Bidirectional LSTM',
                                id = 'BTC_BLSTM_VGC_1D'
                                )
                            ]
                        ),
                ],
                width={"size": 5, "offset": 0, 'order': 'first'}, 
            ),
            dbc.Col(
            dcc.Graph(
                id='test_plot',
                figure=plot_model_test(preds_df, ticker='BTC-USD', model_id='BTC_LSTM_VGC_1D', pred_scope='1D', px_theme='plotly_white')
                ),
                width={"size": 7, "offset": 0, 'order':'last'}, 
            )
        ]
    ),
],
style=CONTENT_STYLE)

@callback(
    Output("model_dropdown", "label"),
    [Input('BTC_LSTM_VGC_1D', "n_clicks"), Input('BTC_BLSTM_VGC_1D', "n_clicks")],
)
def update_label(n1, n2):
    # use a dictionary to map ids back to the desired label
    id_lookup = {
        'BTC_LSTM_VGC_1D': 'LSTM',
        'BTC_BLSTM_VGC_1D': 'Bidirectional LSTM'
    }

    ctx = dash.callback_context

    if (n1 is None and n2 is None) or not ctx.triggered:
        # if neither button has been clicked, return "Not selected"
        return "Select a Model"

    # this gets the id of the button that triggered the callback
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return id_lookup[button_id]
