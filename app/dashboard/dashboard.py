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

# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": "2rem",
    "bottom": 0,
    "width": "8rem",
    "padding": "2rem 1rem",
    "background-color": "#000000",
}



NAVBUTTON_STYLE = {
    'textAlign':'center', 
    'padding-':'0.2em 0.2em 0.2em 0.2em', 
    'color': '#FFFFFF'
    }

sidebar = html.Div(
    [
        html.P("Market: ", style={'textAlign':'center'}),
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
    style=SIDEBAR_STYLE,
)

app.layout = html.Div([
    sidebar,
	dash.page_container
])

if __name__ == '__main__':
	app.run_server(debug=True)