import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

dashboard_app = Dash(__name__, use_pages=True, 
    external_stylesheets=[
            dbc.themes.LUX,
            dbc.icons.FONT_AWESOME
            ])


NAVBUTTON_STYLE = {
    'textAlign':'center', 
    'padding-':'0.2em 0.2em 0.2em 0.2em', 
    'color': '#FFFFFF',
    "cursor": "pointer"
    }

sidebar = html.Div(
    className="A",
    children=[html.Div(
            className="sidebar_style",
            children=[
                        html.P("Menu: "),
                        html.Hr(),
                        dbc.Nav(
                            [   
                                dbc.NavLink(
                                    html.Div(
                                        [html.Span(className="fa fa-line-chart fa-3x" ), "Monitor"], 
                                        style=NAVBUTTON_STYLE), 
                                        href="/", 
                                        active='exact',
                                        id='monitor-tooltip-tg',
                                        className='menu-item'
                                        ),
                                 dbc.Tooltip(
                                    'Live market data and more',
                                    target="monitor-tooltip-tg",
                                    placement='right',
                                    style={'opacity': 0.96, 'background-color': 'gray'}
                                ),
                                html.Hr(),
                                dbc.NavLink(html.Div(
                                    [html.Span(className="fa fa-align-center fa-3x"), "Text AI"], 
                                    style=NAVBUTTON_STYLE), 
                                    href="/textai", 
                                    active="exact",
                                    id='textai-tooltip-tg',
                                    className='menu-item'
                                    ),
                                dbc.Tooltip(
                                    'Read beyond the news and tweets',
                                    target="textai-tooltip-tg",
                                    placement='right',
                                    style={'opacity': 0.96, 'background-color': 'gray'}
                                ),
                                html.Hr(),
                                dbc.NavLink(
                                    html.Div(
                                        [html.Span(className="fa fa-fast-forward fa-3x"), "Forecast"], 
                                        style=NAVBUTTON_STYLE,
                                        className='menu-item'
                                        ), 
                                    href="/forecast", 
                                    active="exact",
                                    id='forecast-tooltip-tg',
                                    
                                ),
                                dbc.Tooltip(
                                    'Use our AI models to make price predictions with one click',
                                    target="forecast-tooltip-tg",
                                    placement='right',
                                    style={'opacity': 0.96, 'background-color': 'gray'}
                                ),
                            ],
                            vertical=True,
                            pills=True,
                        ),
                ],
            )]
)

dashboard_app.layout = html.Div(
    className="container-layout",
    children=[
    sidebar,
    html.Div(
                className="C",
                children=[
                html.H1(children=[html.P(className='fa fa-coins'),'Crypto for All']),
                html.H6('The future of Crypto in one click')
                ]),
    html.Div(
                className="B",
                children=[
                    dash.page_container
                ])
])

dashboard_app.config.suppress_callback_exceptions = True

from .callbacks import register_monitor_callbacks
register_monitor_callbacks(dashboard_app)
#app.config.suppress_callback_exceptions = True

if __name__ == '__main__':
	dashboard_app.run_server(debug=True)