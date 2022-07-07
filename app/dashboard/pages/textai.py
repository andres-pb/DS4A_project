import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback
from matplotlib.pyplot import plot, text
import pandas as pd
import datetime as dt
from dash.dependencies import Input, Output
from app.modules.models_meta import COINS_SELECTION
from app.api import yahoo_finance, Twitter
from app import Sentiment_predict

dash.register_page(__name__)


# padding for the page content
CONTENT_STYLE = {
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

# Read tweets and news
#sp = Sentiment_predict()
#news_df = sp.sentiment_df(source = "News", model = "FinBERT")
# tweets_df = sp.sentiment_df(
#     source = "Tweets", 
#     model = "FinBERT", 
#     tick = "ALL", 
#     quer = "crypto OR blockchain OR cryptocurrency OR coin bitcoin OR BTC OR Ethereum OR ETH OR Litecoin OR LTC"
# )


tweets_df = pd.read_csv('./tweets_all.csv', parse_dates=['created_at'])

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
                                id='coin-dropdown-ta',
                                options=[
                                    {'label': 'All', 'value': 'ALL'},
                                    {'label': 'BTC - Bitcoin', 'value': 'Bitcoin'},
                                    {'label': 'ETH - Ethereum', 'value': 'Ethereum'},
                                    {'label': 'LTC - Litecoin', 'value': 'Litecoin'}
                                ],
                                value='ALL',
                                clearable=False,
                                persistence=False                          
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
                    children=[
                        html.H3('Sentiment:'),
                            dcc.Dropdown(
                                id='src-dropdown',
                                options=[{'label': s, 'value': s} for s in ['Positive', 'Negative', 'All']],
                                value='All',
                                clearable=False,
                            )
                    ],
                    width={"size": 4, "offset": 0, 'order': 'last'},
                )
            ],
            style={'margin-right': '20px','margin-top': '20px', 'margin-bottom': '50px'}
        ),
        dbc.Row([
            dbc.Col(
                html.Div(
                    children=[
                        html.H3('COUNT OVER TIME'),
                        dcc.Graph(id='count-bar-plot')
                        ],
                    className='graph-container',
                ),
                width={"size": 6, "offset": 0, 'order': 2},
            ),
        dbc.Col(
                html.Div(
                    children=[
                        html.H3('TOTAL COUNT'),
                        html.Div(id='total-count')
                        ],
                    className='graph-container',
                ),
                width={"size": 2, "offset": 0, 'order': 2},
            ),
        dbc.Col([
                html.Div(
                    [dcc.Loading(
                        children=[html.Img(
                                    id='wcloud', 
                                    src='./assets/wcloud_all.png',
                                    className='wcloud-image'
                                    )]
                        ),],
                    id='wcloud-container',
                )],
                style={'padding': '1rem 1rem 1rem 1rem'},
                width={"size": 4, "offset": 0, 'order': 2},
            )
        ],
        ),
    ],
    style=CONTENT_STYLE
)

#============================================CALLBACKS=================================================#
# @callback(
#    Output('wcloud', 'src'),
#    [Input('src-dropdown', 'value'), Input('coin-dropdown-ta', 'value')] 
# )
# def plot_words(sel_src, sel_coin):

#     if sel_src == 'Twitter': 
#         if sel_coin == 'ALL':
#             wcfig = plot_twt_wordcloud(tweets_df)
#             dash.get_asset_url(wcfig)
#         else:
#             tdf = tweets_df[tweets_df['Ticker']==sel_coin[:3]]
#             wcfig = plot_twt_wordcloud(tdf)
#             return dash.get_asset_url(wcfig)