import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback
from matplotlib.pyplot import plot, text
import pandas as pd
import datetime as dt
from app.dashboard.crypto_plots import plot_model_test, plot_importance, plot_lime
from dash.dependencies import Input, Output
from app.modules.models_meta import COINS_SELECTION
from app.api import yahoo_finance, GoogleTrends
from app.modules.lstm import get_prediction

dash.register_page(__name__)


# padding for the page content
CONTENT_STYLE = {
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

# Read predictions obtained during model testing


layout = html.Div(
    [
        # Menus and controls row
        dbc.Row([
                dbc.Col(
                    html.Div(
                        className='dropdown-fc',
                        children=[
                            html.H3('Cryptocurrency:'),
                            dcc.Dropdown(
                                id='coin-dropdown',
                                options=[{'label': c, 'value': c[:3]} for c in COINS_SELECTION],
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
                            html.H3('Source:'),
                            dcc.Dropdown(
                                id='src-dropdown',
                                options=[{'label': s, 'value': s} for s in ['Twitter', 'News']],
                                value='Twitter',
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
                            html.H3('Time Frame:'),
                            dcc.Dropdown(
                                id='timef-dropdown',
                                options=['Last 24H', 'Max'],
                                value='Last 24H',
                                clearable=False,
                            )
                        ],
                    ),
                    width={"size": 4, "offset": 0, 'order': 'last'},
                )
            ],
            style={'margin-right': '20px','margin-top': '20px', 'margin-bottom': '50px'}
        ),

    ],
    style=CONTENT_STYLE
)

#============================================CALLBACKS=================================================#
"""@callback
(
    
)"""