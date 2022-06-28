import dash
from dash import html, dcc
from app import globals_variable
dash.register_page(__name__)

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "12rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

layout = html.Div(
                    className="flex-container",
                    children=[
                                 html.Div( 
                                    className="row",
                                    children=[
                                        html.Div( 
                                        className="column",
                                        children=[ 
                                                    html.Div( 
                                                            className="item monitor-title",
                                                            children=[ 
                                                                            html.P( children='BTC-USD'),
                                                                            ],
                                                                    ),
                                                    html.Div( 
                                                            className="item",
                                                            children=[ 
                                                                        html.Div(
                                                                                className="dropdown",
                                                                                children=[
                                                                                            html.P( className="dropdown-title", children='Crypto Currency'),
                                                                                            dcc.Dropdown(
                                                                                                className="dropdown-item", 
                                                                                                options=[{'label': x['name'], 'value': x['ticker']} for x in globals_variable.COINS_SELECTION],
                                                                                                value= globals_variable.COINS_SELECTION[0]['name']
                                                                                            ),
                                                                                        ]),
                                                                        html.Div(
                                                                            className="dropdown",
                                                                            children=[
                                                                                    html.P( className="dropdown-title", children='Currency'),
                                                                                    dcc.Dropdown(
                                                                                        className="dropdown-item", 
                                                                                        options=[{'label': x['name'], 'value': x['ticker']} for x in globals_variable.EXCHANGES],
                                                                                        value=globals_variable.EXCHANGES[0]['name']
                                                                                    ),
                                                                                ])
                                                                            ],
                                                                    )
                                                ]
                                                        ),
                                                html.Div( 
                                                        className="column summary",
                                                        children=[
                                                                    html.Div( 
                                                                                className="column monitor-container",
                                                                                children=[
                                                                                    html.P(children=['Average Value'], className="monitor-output-text monitor-A"),
                                                                                    html.Div(id='output-average', className="item monitor-output monitor-B")
                                                                                    ]
                                                                            ),
                                                                    html.Div( 
                                                                                className="column monitor-container",
                                                                                children=[
                                                                                    html.P(children=['Standard Deviation'], className="monitor-output-text monitor-A"),
                                                                                    html.Div(id='output-deviation', className="item monitor-output monitor-b")
                                                                                    ]
                                                                            ),
                                                                    html.Div( 
                                                                                className="column monitor-container",
                                                                                children=[
                                                                                    html.P(children=['Maximum Value'], className="monitor-output-text monitor-A"),
                                                                                    html.Div(id='output-max', className="item monitor-output monitor-B")
                                                                                    ]
                                                                            ),
                                                                    html.Div( 
                                                                                className="column monitor-container",
                                                                                children=[
                                                                                    html.P(children=['Minimal Value'], className="monitor-output-text monitor-A"),
                                                                                    html.Div(id='output-min', className="item monitor-output monitor-B")
                                                                                    ]
                                                                            )
                                                                ]
                                                        ),   
                                    html.Div( 
                                                                                className="monitor-container2",
                                                                                children=[
                                                                                    dcc.Graph(
                                                                                                    id='test_plot',
                                                                                                    className="monitor-AA",
                                                                                                    #figure=plot_model_test(preds_df, ticker='BTC-USD', model_id='BTC_LSTM_VGC_1D', pred_scope='1D'),
                                                                                                    style={'right': 0}
                                                                                                ),
                                                                                    html.Div( 
                                                                                               className="monitor-BB",
                                                                                               children=["asdasdfadsfsdfdsf assdgfdfs"]
                                                                                            ),
                                                                                    html.Div( 
                                                                                               className="monitor-CC"
                                                                                            )
                                                                                    
                                                                                ]
                                    )
                                    ]
                                 ),
                            ],
)