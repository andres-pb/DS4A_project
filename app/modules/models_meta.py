from app.modules.lstm import build_LSTM, build_BLSTM
from dash import html, dcc


COINS_SELECTION = [
    'BTC-USD',
    'NMC-USD', 
    'FTC-USD', 
    'PPC-USD', 
    'LTC-USD',
]

pred_models = {

    'BTC - Bitcoin': {
            'Deep Learning LSTM' : {
                    '1 day ahead': {
                        'model_id': 'BTC_LSTM_VGC_1D',
                        'test_days': 365,
                        'lags': 60,
                        'n_features': 4,
                        'scaler_path': '',
                        'test_weights_path': './app/dash/test_models/BTC_LSTM_VGC_1D.h5',
                        'final_weights_path': '',
                        'builder_func': build_LSTM,
                        'builder_kwargs': dict(
                                        num_rnns=2,
                                        dim_rnn=50, 
                                        dense_units=100, 
                                        drop=False, 
                                        drop_rate=0.5),
                        'about': html.Div([
                            html.P([
                                html.A('Long Short Term Memory Neural Network',
                                 href='https://en.wikipedia.org/wiki/Long_short-term_memory',
                                 target="_blank",
                                 style={'color': 'white'}),
                                """
                                    using 60 lags of 
                                    multiple features to forecast the closing price of the cryptocurrency.
                                """]),
                            html.P(
                                """
                                It consumes public data on four features until the last known daily close. 
                                Historical Closing Price values constitute the autoregressive nature of the model, 
                                historical traded volume is used as an additional market signal, 
                                the historical U.S. Treasury Yield at the most relevant maturity among 5 (FVX), 10 (TNX) 
                                or 30 (TYX) years is included as a way of capturing international inflation and market risk signals. 
                                Finally, the daily social interest in the cryptocurrency as measured by Google Trends takes 
                                into account potential investors' and general public perception on the asset.
                                """),
                            html.P("LSTM layers capture the sequential structure of the data to predict next day's closing price.")
                       ])
                    },
                    # another time scope
            },
            # another model type for btc
    },
    # another coin
        
}
