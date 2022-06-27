import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

app = Dash(__name__, use_pages=True, 
    external_stylesheets=[
            dbc.themes.LUX,
            dbc.icons.FONT_AWESOME
            ])

APP_TITLE = html.Div(
            [
                html.H1('Crypto AI',
                        className="display-6",
                        style={'textAlign':'right'}),
                html.Hr(),
            ],
            style={'top': 0, 'right': 0}
    )


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
                                dbc.NavLink(html.Div([html.Span(className="fa fa-line-chart fa-3x" ), "Monitor"], style=NAVBUTTON_STYLE), href="/monitor", active="exact"),
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

app.layout = html.Div(
    className="container",
    children=[
    sidebar,
    html.Div(
                className="C",
                children=[
                html.H1(children='Crypto Trading Bot'),
                ]),
    html.Div(
                className="B",
                children=[
                dash.page_container
                ])
])

if __name__ == '__main__':
	app.run_server(debug=True)