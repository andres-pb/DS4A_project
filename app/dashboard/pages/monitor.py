import dash
from dash import Input, Output, html, dcc,  callback
from app import globals_variable
from app.dashboard.crypto_plots import plot_monitor_candle, plot_monitor_candle_volume, plot_monitor_line, plot_monitor_line_volume
dash.register_page(__name__, path='/')

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "2rem",
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
                                                            id='title-coin',
                                                            className="item monitor-title",
                                                            children=[ 
                                                                html.P(children=['BTC - Bitcoin - USD']),
                                                                    ],
                                                            ),
                                                    html.Div( 
                                                            className="item",
                                                            children=[ 
                                                                        html.Div(
                                                                                className="dropdown",
                                                                                children=[
                                                                                            html.H3( className="dropdown-title", children='Cryptocurrency'),
                                                                                            dcc.Dropdown(
                                                                                                id="dropdown_coins",
                                                                                                className="dropdown-item", 
                                                                                                options=[{'label': x['name'], 'value': x['ticker']} for x in globals_variable.COINS_SELECTION],
                                                                                                value= 'BTC-USD',
                                                                                                persistence=True,
                                                                                                persistence_type='session'  
                                                                                            ),
                                                                                        ]),
                                                                        html.Div(
                                                                            className="dropdown",
                                                                            children=[
                                                                                    html.H3(className="dropdown-title", children='Currency'),
                                                                                    dcc.Dropdown(
                                                                                        id="dropdown_exchanges",
                                                                                        className="dropdown-item", 
                                                                                        options=[{'label': x['name'], 'value': x['ticker']} for x in globals_variable.EXCHANGES],
                                                                                        value='USD',
                                                                                        persistence=True,
                                                                                        persistence_type='session'
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
                                                                                    html.Div(id='output-average', className="item monitor-output monitor-B",children=[''])
                                                                                    ]
                                                                            ),
                                                                    html.Div( 
                                                                                className="column monitor-container",
                                                                                children=[
                                                                                    html.P(children=['Standard Deviation'], className="monitor-output-text monitor-A"),
                                                                                    html.P(id='output-deviation', className="item monitor-output monitor-b",children=[''])
                                                                                    ]
                                                                            ),
                                                                    html.Div( 
                                                                                className="column monitor-container",
                                                                                children=[
                                                                                    html.P(children=['Maximum Value'], className="monitor-output-text monitor-A"),
                                                                                    html.Div(id='output-max', className="item monitor-output monitor-B",children=[''])
                                                                                    ]
                                                                            ),
                                                                    html.Div( 
                                                                                className="column monitor-container",
                                                                                children=[
                                                                                    html.P(children=['Minimal Value'], className="monitor-output-text monitor-A"),
                                                                                    html.Div(id='output-min', className="item monitor-output monitor-B",children=[''])
                                                                                    ]
                                                                            )
                                                                ]
                                                        ),   
                                    html.Div( 
                                                                                className="monitor-container2",
                                                                                children=[
                                                                                html.Div( 
                                                                                               className="monitor-AA monitor-container-graph",
                                                                                               children=[
                                                                                                            html.Div( 
                                                                                                                        className="monitor-GRAPH-A",
                                                                                                                        children=[
                                                                                                                                        html.Div( 
                                                                                                                                            className="monitor-left-group",
                                                                                                                                            children=[
                                                                                                                                                html.Button('Close', className="graph-button", id='btn_nclicks_1', n_clicks=0),
                                                                                                                                                html.Button('Adj Close', className="graph-button", id='btn_nclicks_2', n_clicks=0),
                                                                                                                                            ]
                                                                                                                                        ),
                                                                                                                                        html.Div( 
                                                                                                                                            className="monitor-right-group",
                                                                                                                                            children=[
                                                                                                                                                html.Button('1D', className="graph-button", id='btn_nclicks_3', n_clicks=0),
                                                                                                                                                html.Button('5D', className="graph-button", id='btn_nclicks_4', n_clicks=0),
                                                                                                                                                html.Button('1W', className="graph-button", id='btn_nclicks_5', n_clicks=0, style={'background-color': 'white'}),
                                                                                                                                                html.Button('1M', className="graph-button", id='btn_nclicks_6', n_clicks=0),
                                                                                                                                                html.Button('3M', className="graph-button", id='btn_nclicks_7', n_clicks=0),
                                                                                                                                            ]
                                                                                                                                        )
                                                                                                                        ]
                                                                                                                    ),
                                                                                                            dcc.Graph(
                                                                                                                        id='monitor_plot',
                                                                                                                        className="monitor-GRAPH-B",
                                                                                                                        figure=plot_monitor_line()
                                                                                                                        )
                                                                                                        ]
                                                                                                ),
                                                                                    html.Div( 
                                                                                               className="monitor-BB",
                                                                                               children=[
                                                                                                            html.Div( 
                                                                                                                        className="radio-title",
                                                                                                                        children=['View:']
                                                                                                                    ),
                                                                                                            dcc.Checklist(
                                                                                                                            className="radio-item",
                                                                                                                            id="checklist_view",
                                                                                                                            options=[x['name'] for x in globals_variable.STATISTICAL_MODELS],
                                                                                                                            value=['Volume']
                                                                                                                            )
                                                                                                        ]
                                                                                            ),
                                                                                    html.Div( 
                                                                                               className="monitor-CC",
                                                                                               children=[
                                                                                                            html.Div( 
                                                                                                                        className="radio-title",
                                                                                                                        children=['Chart Type:']
                                                                                                                    ),
                                                                                                            html.Div(
                                                                                                                        className="dropdown",
                                                                                                                        children=[
                                                                                                                                dcc.Dropdown(
                                                                                                                                    className="dropdown-item", 
                                                                                                                                    id="dropdown-chart",
                                                                                                                                    options=[
                                                                                                                                                {'label': 'Line', 'value': 'line'},
                                                                                                                                                {'label': 'Candle', 'value': 'candle'}
                                                                                                                                            ],
                                                                                                                                    value='line',
                                                                                                                                    persistence=True,
                                                                                                                                    persistence_type='memory'
                                                                                                                                ),
                                                                                                                            ])
                                                                                                        ]
                                                                                            )
                                                                                    
                                                                                ]
                                    )
                                    ]
                                 ),
                            ],
                            style=CONTENT_STYLE,
)

@callback(Output('title-coin', 'children'),
            [
                Input('dropdown_coins', 'value'), 
                Input('dropdown_exchanges', 'value'), 
            ]
        )
def update_title(coin, exchange):
    return coin[:3] + ' - ' + exchange[:3]