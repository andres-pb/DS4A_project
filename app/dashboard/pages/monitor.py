import dash
from dash import html, dcc

dash.register_page(__name__)

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "12rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

layout = html.Div(children=[
    html.H1(children=' IVAN GOMEZ'),

    html.Div(children='''
        This is our Archive page content.
    '''),

],
style=CONTENT_STYLE)