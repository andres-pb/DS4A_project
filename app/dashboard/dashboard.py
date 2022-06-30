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
    'color': '#FFFFFF'
    }

sidebar = html.Div(
    className="A",
    children=[html.Div(
            className="sidebar_style",
            children=[
                        html.P("Market: "),
                        html.Hr(),
                        dbc.Nav(
                            [   
                                dbc.NavLink(html.Div([html.Span(className="fa fa-line-chart fa-3x" ), "Monitor"], style=NAVBUTTON_STYLE), href="/monitor", active='exact'),
                                html.Hr(),
                                dbc.NavLink(html.Div([html.Span(className="fa fa-align-center fa-3x"), "Text AI"], style=NAVBUTTON_STYLE), href="/textai", active="exact"),
                                html.Hr(),
                                dbc.NavLink(html.Div([html.Span(className="fa fa-fast-forward fa-3x"), "Forecast"], style=NAVBUTTON_STYLE), href="/forecast", active="exact"),
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
                html.H1(children=[html.P(className='fa fa-coins'),'Crypto Trading Bot']),
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
if __name__ == '__main__':
	dashboard_app.run_server(debug=True)