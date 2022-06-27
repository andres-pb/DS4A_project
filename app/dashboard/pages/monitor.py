import dash
from dash import html, dcc

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
                                                                className="",
                                                                children=[
                                                                            html.Div( 
                                                                            className="item",
                                                                            children=[ 
                                                                                            html.P(className="monitor-title", children='BTC-USD'),
                                                                                            ],
                                                                                        style=CONTENT_STYLE
                                                                                    )
                                                                            ]
                                                            ),

                                                    html.Div( 
                                                                className="",
                                                                children=[
                                                                            html.Div( 
                                                                            className="item",
                                                                            children=[ 
                                                                                            html.P( children='BTC-USD sdfdsfsdfsd'),
                                                                                            ],
                                                                                        style=CONTENT_STYLE
                                                                                    ),
                                                                            html.Div( 
                                                                            className="item",
                                                                            children=[ 
                                                                                            html.P(children='BTC-USD sdfsdfdsfdsfdsfdsfsdfdsfdsfsdf'),
                                                                                            ],
                                                                                        style=CONTENT_STYLE
                                                                                    )
                                                                            ]
                                                            )
                                                    ]
                                                    )
                                                ]
                                            ),
                                html.Div( 
                                    className="row",
                                    children=[
                                        html.Div( 
                                                className="column",
                                                children=[
                                                html.Div( 
                                                            className="item",
                                                            children=[ 'hola']
                                                        )
                                                        ]
                                                 )   
                                            ]
                                         )
                            ],
)