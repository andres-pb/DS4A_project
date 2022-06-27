"""
Dash app server. Runs with:
>> py crypto_dash.py
"""

from re import S
import dash
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Input, Output, html, dcc

from crypto_plots import plot_model_test, filter_test_df

external_style = ['/assets/fontAwesome/font-awesome.css']

app = dash.Dash(
    __name__, 
    external_stylesheets=[
            dbc.themes.LUX,
            dbc.icons.FONT_AWESOME
            ]
    )

preds_df = pd.read_csv('./app/dash/test_models/predictions.csv',  parse_dates=['Date'], index_col='Date')


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

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "12rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
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

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/monitor":
        return [
                APP_TITLE,

                ]
    elif pathname == "/textai":
        return [
                APP_TITLE,
                ]
    elif pathname == "/forecast":
        return [
                APP_TITLE,
                dbc.Row([
                    dbc.Col(
                        html.Div("Left column here"),
                        width={"size": 5, "offset": 0, 'order': 'first'}, 
                    ),
                    dbc.Col(
                    dcc.Graph(
                        id='test_plot',
                        figure=plot_model_test(preds_df, ticker='BTC-USD', model_id='BTC_LSTM_VGC_1D', pred_scope='1D', px_theme='plotly_white')
                        ),
                        width={"size": 7, "offset": 0, 'order':'last'}, 
                    )
                ]),

                ]
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == '__main__':

    app.run_server(debug=True)