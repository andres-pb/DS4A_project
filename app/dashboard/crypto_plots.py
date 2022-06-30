from turtle import width
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# To load filtered test predictions
def filter_test_df(test_df, ticker, model_id, pred_scope):

    test_df = test_df.query('Scope==@pred_scope & Ticker==@ticker & Model==@model_id')
    test_df = test_df[['Observed', 'Predicted']]

    return test_df.sort_index()


def plot_model_test(filtered_df: pd.DataFrame, px_theme: str ='plotly_dark'):

    fig = px.line(
            filtered_df,
            x=filtered_df.index, 
            y=filtered_df.columns,
            template=px_theme,
            labels={
                    'variable': ''
                },
            )
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    fig.update_yaxes(title_text = 'Close Price')
    fig.update_layout(
        yaxis_tickformat = '$',
        title={
            #'text': "Model Performance",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
            #family="Courier New, monospace",
            size=25)
            },
        )
    fig.update_traces(hovertemplate='%{y:$,.2f}')
    fig.update_layout(hovermode="x unified")

    fig.update_xaxes(
        title_text = 'Date',
        rangeslider_visible = True
        )
    fig.update_layout(width=750, height=500)
    return fig


def plot_importance(importance_df: pd.DataFrame, px_theme: str ='plotly_dark'):

    fig = go.Figure()
    fig.update_xaxes(showgrid=True, gridwidth=0.1, gridcolor='DarkGrey')

    metrics_list = list(importance_df['Metric'].unique())

    for metric in metrics_list:
        vis = metric == 'mae'
        mdf = importance_df[importance_df['Metric']==metric]

        mdf = mdf.sort_values('Importance').tail(5)
        fig.add_trace(
            go.Bar(
                orientation='h', 
                x = mdf['Importance'],
                y = mdf['Feature'],
                name = metric, 
                visible=vis
            )
        )
                
    buttons = []

    for i, metric in enumerate(metrics_list):
        args = [False] * len(metrics_list)
        args[i] = True
        
        button = dict(
                    label = metric.upper(),
                    method = "update",
                    args=[{"visible": args}])
        
        buttons.append(button)
        
    fig.update_layout(
        updatemenus=[dict(
                        active=0,
                        type="buttons",
                        direction = "left",
                        buttons=buttons,
                        x = 1,
                        y = 1,
                        xanchor = 'left',
                        yanchor = 'bottom'
                    )], 
        autosize=True,
    )
    fig.update_layout(
        title={
            'text': "Error Perturbation",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
            #family="Courier New, monospace",
            size=25
            )
            },
        template=px_theme
    )

    fig.update_yaxes(tickmode='linear')
    #fig.update_layout(updatemenus=[dict(font=dict(color='gray',), bgcolor='black')])
    fig.update_traces(marker_color='rgba(50, 171, 96, 0.6)')
    fig.update_layout(width=750, height=500)

    return fig