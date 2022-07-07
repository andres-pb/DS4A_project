import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback
from matplotlib.pyplot import plot, text
import pandas as pd
import sqlite3 as sql
import datetime as dt
from dash.dependencies import Input, Output
from app.modules.models_meta import COINS_SELECTION
from app.api import yahoo_finance, Twitter
import plotly.express as px
import plotly.graph_objects as go

#from app import Sentiment_predict

dash.register_page(__name__)


# padding for the page content
CONTENT_STYLE = {
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

# Read tweets and news
news_df = pd.read_csv('./sentiment_models/polygon_df_01_07_2022.csv', parse_dates=['published_utc'])
tweets_df = pd.read_csv('./tweets_all.csv', parse_dates=['created_at']).sort_values('created_at', ascending=False)


layout = html.Div([
        # Menus and controls row
        dbc.Row([
                dbc.Col(
                    [
                    html.H3('Read the Content:'),
                    html.Br(),
                    dcc.Dropdown(
                        options=[{'label':s.capitalize(), 'value': s} for s in ['positive', 'negative', 'neutral']],
                        value=['positive', 'negative', 'neutral'],
                        multi=True,
                        id='sent-dropdown',
                        className='white-selector'
                    ),
                    html.Br(),
                    dcc.Loading(
                        html.Div(
                            children=[],
                            id='text-explorer',
                            className='cards-container mb-4',
                        ),
                        type='circle',
                        color='#A0A0A0'
                    ) 
                    ],
                    className='ly-margins',
                    width={"size": 6, "offset": 0, 'order': 'first'},
                ),
                dbc.Col(
                    [html.Div(
                        className='dropdown-fc',
                        children=[
                            html.H3('Source:'),
                            html.Br(),
                            dcc.Dropdown(
                                id='src-dropdown',
                                options=[{'label': s, 'value': s} for s in ['Twitter', 'News']],
                                value='News',
                                clearable=False,
                            ),
                            html.Br(),
                        ],
                    ),
                    dcc.Loading(
                        [
                            html.Div(
                            children=[
                                html.H3('Distribution by Sentiment'),
                                dcc.Graph(id='sentiment-pie')
                                ],
                                className='graph-container',
                            ),    
                        ],
                        type='circle',
                        color='#A0A0A0',
                    ),
                ],
                    className='ly-margins',
                    width={"size": 6, "offset": 0, 'order': 'last'},
                ),
            ]),
        html.H1('Words By Sentiment', style={'text-align': 'center'}),
        html.Br(),
        dbc.Row(
            [],
            id='wordcloud-row',
        )
    ],
    style = CONTENT_STYLE
)
       

#============================================CALLBACKS=================================================#
@callback(
   Output('wordcloud-row', 'children'),
   [Input('src-dropdown', 'value')] 
)
def plot_words(sel_src):

    if sel_src == 'Twitter':

        return [dbc.Col(
                children=[
                    html.Div(
                    [dcc.Loading(
                        children=[html.Img(
                                    id='wcloud-neg', 
                                    src='./assets/wcloud_negative.png',
                                    className='wcloud-image'
                                    )]
                        ),],
                    id='wcloud-container',
                    )
                ],
                width={"size": 4, "offset": 0, 'order': 'first'}
            ),
            dbc.Col(
                children=[
                    html.Div(
                    [dcc.Loading(
                        children=[html.Img(
                                    id='wcloud-neutral', 
                                    src='./assets/wcloud_all.png',
                                    className='wcloud-image'
                                    )]
                        ),],
                    id='wcloud-container',
                    )
                ],
                width={"size": 4, "offset": 0, 'order': 2}
            ),
            dbc.Col(
                children=[
                    html.Div(
                    [dcc.Loading(
                        children=[html.Img(
                                    id='wcloud-positive', 
                                    src='./assets/wcloud_positive.png',
                                    className='wcloud-image'
                                    )]
                            ),],
                        id='wcloud-container',
                        )
                    ],
                    width={"size": 4, "offset": 0, 'order': 2}
                )
            ]

    else:
        return [dbc.Col(
                children=[
                    html.Div(
                    [dcc.Loading(
                        children=[html.Img(
                                    id='wcloud-neg', 
                                    src='./assets/news_wcloud_negative.png',
                                    className='wcloud-image'
                                    )]
                        ),],
                    id='wcloud-container',
                    )
                ],
                width={"size": 4, "offset": 0, 'order': 'first'}
            ),
            dbc.Col(
                children=[
                    html.Div(
                    [dcc.Loading(
                        children=[html.Img(
                                    id='wcloud-neutral', 
                                    src='./assets/news_wcloud_all.png',
                                    className='wcloud-image'
                                    )]
                        ),],
                    id='wcloud-container',
                    )
                ],
                width={"size": 4, "offset": 0, 'order': 2}
            ),
            dbc.Col(
                children=[
                    html.Div(
                    [dcc.Loading(
                        children=[html.Img(
                                    id='wcloud-positive', 
                                    src='./assets/news_wcloud_positive.png',
                                    className='wcloud-image'
                                    )]
                            ),],
                        id='wcloud-container',
                        )
                    ],
                    width={"size": 4, "offset": 0, 'order': 2}
                )
            ]
   


@callback(
    Output('sentiment-pie', 'figure'),
    [Input('src-dropdown', 'value')]
)
def update_count_bars(sel_src):

    print('SELECTED SOURCEEEE', sel_src)
    if sel_src == 'Twitter':
        total = tweets_df['id'].count() 
        sub_tweets_df = tweets_df[['id', 'Sentiment']].groupby(['Sentiment']).nunique().reset_index().sort_values('Sentiment')
        labels=sub_tweets_df['Sentiment']
        values=sub_tweets_df['id']
    else:
        total = news_df['id'].count()
        sub_df = news_df[['id', 'description_FinBERT']].groupby(['description_FinBERT']).nunique().reset_index().sort_values('description_FinBERT')
        sub_df.rename(columns={'description_FinBERT': 'Sentiment'}, inplace=True)
        labels = sub_df['Sentiment']
        values = sub_df['id']

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=.5,
            sort=False,
            marker_colors=['#d9534f', '#1f9bcf', '#50bc74']
        )
    )

    fig.add_annotation(
        x= 0.5, y = 0.5,
        text = str(total),
        showarrow = False,
        font = dict(
            size=40,family='Verdana', 
            color='darkgray'),
        )
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),)
   
    return fig


@callback(
    Output('text-explorer', 'children'),
    [Input('src-dropdown', 'value'), Input('sent-dropdown', 'value'),]
)
def populate_cards(sel_src, sel_sents):

    if sel_src == 'Twitter':
        tweets_df_cards = tweets_df[tweets_df['Sentiment'].isin(sel_sents)]
        # Define cards
        twt_cards = []

        for idx, tweet in tweets_df_cards.iterrows():
            footer = dt.datetime.strftime(tweet['created_at'], '%Y-%m-%d %H:%M')
            header = tweet['user_name']
            ftext = tweet['full_text']
            sent = tweet['Sentiment']
            
            if sent == 'positive':
                color = 'success'
            elif sent == 'negative':
                color = 'danger'
            else:
                color = 'info'

            card = dbc.Card([
                    dbc.Row(
                        children=[
                            dbc.Col([
                                dbc.CardImg(
                                    src=dash.get_asset_url("profile.png"),
                                    className="img-fluid rounded-start",
                                )],
                                className="col-md-4"
                            ),
                            dbc.Col(
                            [
                                dbc.CardBody([
                                    html.H4(header, className='card-title'),
                                    html.P(ftext, className="card-text",),
                                ],),
                            ] 
                            ),            
                            dbc.CardFooter(footer)
                        ],
                        className="g-0 d-flex align-items-center",
                    )
                ],
                color=color,
                className="post-card mb-3",
            )

            twt_cards.append(card)

        return twt_cards
    
    else:
        ndf_cards = news_df[news_df['description_FinBERT'].isin(sel_sents)]
        ncards = []

        for idx, article in ndf_cards.iterrows():

            title=article['title']
            footer = dt.datetime.strftime(article['published_utc'], '%Y-%m-%d %H:%M')
            ftext = article['description']
            header = article['author']
            sent = article['description_FinBERT']
            link = article['article_url']
            image_url = article['image_url']
            
            if sent == 'positive':
                color = 'success'
            elif sent == 'negative':
                color = 'danger'
            else:
                color = 'info'
            
            card = dbc.Card([
                    dbc.CardHeader(html.H4(title)),
                    dbc.Row(
                        children=[
                            dbc.Col([
                                dbc.CardImg(
                                    src=image_url,
                                    className="img-fluid rounded-start",
                                )],
                                className="col-md-4"
                            ),
                            dbc.Col(
                            [
                                dbc.CardBody([
                                    html.H6(header, className='card-title'),
                                    html.P(ftext, className="card-text",),
                                    html.A('Read...', href=link, target="_blank",)
                                ],),
                            ] 
                            ),            
                            dbc.CardFooter(footer)
                        ],
                        className="g-0 d-flex align-items-center",
                    )
                ],
                color=color,
                className="post-card mb-3",
            )

            ncards.append(card)

        return ncards



            

