from app.modules.lstm import build_LSTM, build_BLSTM
from dash import html, dcc


COINS_SELECTION_TICKERS = [
    'BTC-USD',
    'NMC-USD', 
    'FTC-USD', 
    'PPC-USD', 
    'LTC-USD',
]

COINS_SELECTION = [
    'BTC - Bitcoin',
    'ETH - Ethereum',
    'LTC - Litecoin', 
    'NMC - Namecoin', 
]

pred_models = {

    'BTC - Bitcoin': {
            'Bidirectional LSTM' : {
                    '1 day ahead': {
                        'ticker': 'BTC',
                        'coin_name': 'Bitcoin',
                        'model_id': 'BTC_Bidirectional_LSTM_60_lags_4_fts_prod',
                        'test_days': 365,
                        'lags': 60,
                        'n_features': 4,
                        'features': ['Volume', 'Close'],
                        'treasury_ticker': '^TYX',
                        'google_trend': True,
                        'ordered_features': ['Volume', 'Gtrend', 'TYX', 'Close'],
                        'builder_func': build_BLSTM,
                        'builder_kwargs': dict(
                                        num_rnns=2,
                                        dim_rnn=16, 
                                        dense_units=50, 
                                        drop=False,
                                        drop_rate=0.5
                                        ),
                        'about': html.Div([
                            html.P([
                                "Bidirectional ",
                                html.A('Long Short Term Memory Neural Network',
                                 href='https://en.wikipedia.org/wiki/Long_short-term_memory',
                                 target="_blank",
                                 style={'color': 'white'}),
                                """
                                    using 60 lags of multiple features to forecast the closing 
                                    price of the cryptocurrency.
                                """], ),
                            html.P(
                                """
                                It consumes public data on four features until the last known daily close. 
                                Historical Closing Price values constitute the autoregressive nature of the model, 
                                historical traded volume is used as an additional market signal, 
                                the historical U.S. Treasury Yield at the most relevant maturity among 5 (FVX), 10 (TNX) 
                                or 30 (TYX) years is included as a way of capturing international inflation and market risk signals. 
                                Finally, the daily social interest in the cryptocurrency as measured by Google Trends (Gtrend) takes 
                                into account potential investors' and general public perception on the asset.
                                """),
                            html.P("""Bidirectional LSTM layers capture the sequential structure of the data to predict next day's 
                                    closing price. But they also read the sequences backwards, 
                                    so that they are able to capture how the future shapes the past of a series.
                                    """
                                    )
                       ]),
                       'importance_insights':html.Div([
                            html.P([
                                """
                                The forecast for BTC-Bitcoin next day price is mainly explained by the most recent lags of the price itself. 
                                However, when we consider
                                periods of decreasing prices, like the last few weeks, 
                                LIME shows how other factors such as risk free rates, inflation and internet
                                searches become more and more relevant, having, in some cases, a negative impact on the predicted price.
                                """], ),
                            ])
                    },
                    # another time scope
            },
            # another model type for btc
            'Deep Learning LSTM' : {
                    '1 day ahead': {
                        'ticker': 'BTC',
                        'coin_name': 'Bitcoin',
                        'model_id': 'BTC_LSTM_60_lags_4_fts_prod',
                        'test_days': 365,
                        'lags': 60,
                        'n_features': 4,
                        'features': ['Volume', 'Close'],
                        'treasury_ticker': '^TYX',
                        'google_trend': True,
                        'ordered_features': ['Volume', 'Gtrend', 'TYX', 'Close'],
                        'builder_func': build_LSTM,
                        'builder_kwargs': dict(
                                    num_rnns=2,
                                    dim_rnn=100, 
                                    dense_units=100, 
                                    drop=False, 
                                    drop_rate=0.1,
                                ),
                        'about': html.Div([
                            html.P([
                                html.A('Long Short Term Memory Neural Network',
                                 href='https://en.wikipedia.org/wiki/Long_short-term_memory',
                                 target="_blank",
                                 style={'color': 'white'}),
                                """
                                    using 60 lags of multiple features to forecast the closing 
                                    price of the cryptocurrency.
                                """], ),
                            html.P(
                                """
                                It consumes public data on four features until the last known daily close. 
                                Historical Closing Price values constitute the autoregressive nature of the model, 
                                historical traded volume is used as an additional market signal, 
                                the historical U.S. Treasury Yield at the most relevant maturity among 5 (FVX), 10 (TNX) 
                                or 30 (TYX) years is included as a way of capturing international inflation and market risk signals. 
                                Finally, the daily social interest in the cryptocurrency as measured by Google Trends (Gtrend) takes 
                                into account potential investors' and general public perception on the asset.
                                """),
                            html.P("LSTM layers capture the sequential structure of the data to predict next day's closing price.")
                       ]),
                       'importance_insights':html.Div([
                            html.P([
                                """
                                The price forecast for BTC Bitcoin is mainly explained by the most recent lags. However, when we consider
                                periods of decreasing prices, LIME shows how other factors such as risk free rates, inflation and internet
                                searches become more and more relevant, having, in some cases, a negative impact on the predicted price.
                                """], ),
                            ])
                    },
                    # another time scope            
            } # another model type
    },
    'ETH - Ethereum': {
        'Bidirectional LSTM' : {
                '1 day ahead': {
                    'ticker': 'ETH',
                    'coin_name': 'Ethereum',
                    'model_id': 'ETH_Bidirectional_LSTM_60_lags_5_fts_prod',
                    'test_days': 120,
                    'lags': 60,
                    'n_features': 4,
                    'features': ['Volume', 'Close'],
                    'treasury_ticker': '^FVX',
                    'google_trend': True,
                    'ordered_features': ['Volume', 'Gtrend', 'FVX', 'BTC', 'Close'],
                    'builder_func': build_BLSTM,
                    'builder_kwargs': dict(
                                    num_rnns=2,
                                    dim_rnn=32, 
                                    dense_units=50, 
                                    drop=False,
                                    drop_rate=0.5
                                    ),
                    'about': html.Div([
                        html.P([
                            "Bidirectional ",
                            html.A('Long Short Term Memory Neural Network',
                                href='https://en.wikipedia.org/wiki/Long_short-term_memory',
                                target="_blank",
                                style={'color': 'white'}),
                            """
                                using 60 lags of multiple features to forecast the closing 
                                price of the cryptocurrency.
                            """], ),
                        html.P(
                            """
                                It consumes public data on five features until the last known daily close. 
                                Historical Closing Price values constitute the autoregressive nature of the model, 
                                historical traded volume is used as an additional market signal, 
                                the historical U.S. Treasury Yield at the most relevant maturity among 5 (FVX), 10 (TNX) 
                                or 30 (TYX) years is included as a way of capturing international inflation and market risk signals. 
                                The daily social interest in the cryptocurrency as measured by Google Trends (Gtrend) takes 
                                into account potential investors' and general public perception on the asset. Finally, the price of Bitcoin (BTC)
                                is also included as the most representative asset of this kind.
                            """),
                        html.P("""Bidirectional LSTM layers capture the sequential structure of the data to predict next day's 
                                closing price. But they also read the sequences backwards, 
                                so that they are able to capture how the future shapes the past of a series.
                                """
                                )
                    ]),
                    'importance_insights':html.Div([
                        html.P([
                            """
                            The forecast for ETH-Ethereum next day price is mainly explained by the 10 most recent lags of the price itself. 
                            However, when we consider more recent periods, like the last few weeks, 
                            LIME shows how other factors such as risk free rates, inflation and internet
                            searches have started to contribute to the price prediction.
                            """], ),
                        ])
                },
                # another time scope
        },
        # another model type for btc
        'Deep Learning LSTM' : {
                '1 day ahead': {
                    'ticker': 'ETH',
                    'coin_name': 'Ethereum',
                    'model_id': 'ETH_LSTM_60_lags_5_fts_prod',
                    'test_days': 120,
                    'lags': 60,
                    'n_features': 4,
                    'features': ['Volume', 'Close'],
                    'treasury_ticker': '^FVX',
                    'google_trend': True,
                    'ordered_features': ['Volume', 'Gtrend', 'FVX', 'BTC', 'Close'],
                    'builder_func': build_LSTM,
                    'builder_kwargs': dict(
                                num_rnns=2,
                                dim_rnn=200, 
                                dense_units=100, 
                                drop=False, 
                                drop_rate=0.1,
                            ),
                    'about': html.Div([
                        html.P([
                            html.A('Long Short Term Memory Neural Network',
                                href='https://en.wikipedia.org/wiki/Long_short-term_memory',
                                target="_blank",
                                style={'color': 'white'}),
                            """
                                using 60 lags of multiple features to forecast the closing 
                                price of the cryptocurrency.
                            """], ),
                        html.P(
                            """
                                It consumes public data on five features until the last known daily close. 
                                Historical Closing Price values constitute the autoregressive nature of the model, 
                                historical traded volume is used as an additional market signal, 
                                the historical U.S. Treasury Yield at the most relevant maturity among 5 (FVX), 10 (TNX) 
                                or 30 (TYX) years is included as a way of capturing international inflation and market risk signals. 
                                The daily social interest in the cryptocurrency as measured by Google Trends (Gtrend) takes 
                                into account potential investors' and general public perception on the asset. Finally, the price of Bitcoin (BTC)
                                is also included as the most representative asset of this kind.
                            """),
                        html.P("LSTM layers capture the sequential structure of the data to predict next day's closing price.")
                    ]),
                    'importance_insights': ''
                },
                # another time scope            
        } # another model type
    },
    'LTC - Litecoin': {
            'Deep Learning LSTM' : {
                    '1 day ahead': {
                        'ticker': 'LTC',
                        'coin_name': 'Litecoin',
                        'model_id': 'LTC_LSTM_60_lags_5_fts_prod',
                        'test_days': 365,
                        'lags': 60,
                        'n_features': 4,
                        'features': ['Volume', 'Close'],
                        'treasury_ticker': '^TYX',
                        'google_trend': True,
                        'ordered_features': ['Volume', 'Gtrend', 'TYX',	'BTC', 'Close'],
                        'builder_func': build_LSTM,
                        'builder_kwargs': dict(
                                            num_rnns=2,
                                            dim_rnn=200, 
                                            dense_units=100, 
                                            drop=True, 
                                            drop_rate=0.3,
                                        ),
                        'about': html.Div([
                            html.P([
                                html.A('Long Short Term Memory Neural Network',
                                    href='https://en.wikipedia.org/wiki/Long_short-term_memory',
                                    target="_blank",
                                    style={'color': 'white'}),
                                """
                                    using 60 lags of multiple features to forecast the closing 
                                    price of the cryptocurrency.
                                """], ),
                            html.P(
                                """
                                It consumes public data on five features until the last known daily close. 
                                Historical Closing Price values constitute the autoregressive nature of the model, 
                                historical traded volume is used as an additional market signal, 
                                the historical U.S. Treasury Yield at the most relevant maturity among 5 (FVX), 10 (TNX) 
                                or 30 (TYX) years is included as a way of capturing international inflation and market risk signals. 
                                The daily social interest in the cryptocurrency as measured by Google Trends (Gtrend) takes 
                                into account potential investors' and general public perception on the asset. Finally, the price of Bitcoin (BTC)
                                is also included as the most representative asset of this kind.
                                """),
                            html.P("LSTM layers capture the sequential structure of the data to predict next day's closing price.")
                        ]),
                       'importance_insights': ''
                    },
                    # another time scope
            },
            # another model type for LTC
            'Bidirectional LSTM' : {
                    '1 day ahead': {
                        'ticker': 'LTC',
                        'coin_name': 'Litecoin',
                        'model_id': 'LTC_Bidirectional_LSTM_60_lags_4_fts_prod',
                        'test_days': 365,
                        'lags': 60,
                        'n_features': 4,
                        'features': ['Volume', 'Close'],
                        'treasury_ticker': '^TYX',
                        'google_trend': True,
                        'ordered_features': ['Volume', 'Gtrend', 'TYX',	'BTC', 'Close'],
                        'builder_func': build_BLSTM,
                        'builder_kwargs': dict(
                                            num_rnns=2,
                                            dim_rnn=16, 
                                            dense_units=50, 
                                            drop=True,
                                            drop_rate=0.3
                                        ),
                        'about': html.Div([
                            html.P([
                                "Bidirectional ",
                                html.A('Long Short Term Memory Neural Network',
                                 href='https://en.wikipedia.org/wiki/Long_short-term_memory',
                                 target="_blank",
                                 style={'color': 'white'}),
                                """
                                    using 60 lags of multiple features to forecast the closing 
                                    price of the cryptocurrency.
                                """], ),
                            html.P(
                                """
                                  It consumes public data on five features until the last known daily close. 
                                Historical Closing Price values constitute the autoregressive nature of the model, 
                                historical traded volume is used as an additional market signal, 
                                the historical U.S. Treasury Yield at the most relevant maturity among 5 (FVX), 10 (TNX) 
                                or 30 (TYX) years is included as a way of capturing international inflation and market risk signals. 
                                The daily social interest in the cryptocurrency as measured by Google Trends (Gtrend) takes 
                                into account potential investors' and general public perception on the asset. Finally, the price of Bitcoin (BTC)
                                is also included as the most representative asset of this kind.
                                """),
                            html.P("""Bidirectional LSTM layers capture the sequential structure of the data to predict next day's 
                                    closing price. But they also read the sequences backwards, 
                                    so that they are able to capture how the future shapes the past of a series.
                                    """
                                    )
                       ]),
                       'importance_insights':html.Div([
                            html.P([
                                """
                                The forecast for ETH-Ethereum next day price is mainly explained by the 10 most recent lags of the price itself. 
                                However, when we consider more recent periods, like the last few weeks, 
                                LIME shows how other factors such as risk free rates, inflation and internet
                                searches have started to contribute to the price prediction.
                                """], ),
                            ])
                    },
                    # another time scope
            },

    },
    'NMC - Namecoin': {
        'Deep Learning LSTM' : {
                '1 day ahead': {
                    'ticker': 'NMC',
                    'coin_name': 'Namecoin',
                    'model_id': 'NMC_LSTM_60_lags_4_fts_prod',
                    'test_days': 365,
                    'lags': 60,
                    'n_features': 4,
                    'features': ['Volume', 'Close'],
                    'treasury_ticker': '^FVX',
                    'google_trend': True,
                    'ordered_features': ['Volume', 'Gtrend', 'FVX', 'BTC', 'Close'],
                    'builder_func': build_LSTM,
                    'builder_kwargs': dict(
                                    num_rnns=2,
                                    dim_rnn=128, 
                                    dense_units=100, 
                                    drop=True, 
                                    drop_rate=0.3,
                                ),
                    'about': html.Div([
                        html.P([
                            html.A('Long Short Term Memory Neural Network',
                                href='https://en.wikipedia.org/wiki/Long_short-term_memory',
                                target="_blank",
                                style={'color': 'white'}),
                            """
                                using 60 lags of multiple features to forecast the closing 
                                price of the cryptocurrency.
                            """], ),
                        html.P(
                            """
                            It consumes public data on five features until the last known daily close. 
                            Historical Closing Price values constitute the autoregressive nature of the model, 
                            historical traded volume is used as an additional market signal, 
                            the historical U.S. Treasury Yield at the most relevant maturity among 5 (FVX), 10 (TNX) 
                            or 30 (TYX) years is included as a way of capturing international inflation and market risk signals. 
                            The daily social interest in the cryptocurrency as measured by Google Trends (Gtrend) takes 
                            into account potential investors' and general public perception on the asset. Finally, the price of Bitcoin (BTC)
                            is also included as the most representative asset of this kind.
                            """),
                        html.P("LSTM layers capture the sequential structure of the data to predict next day's closing price.")
                    ]),
                    'importance_insights': ''
                },
                # another time scope
        },
        # another model type for LTC
        'Bidirectional LSTM' : {
                '1 day ahead': {
                    'ticker': 'NMC',
                    'coin_name': 'Namecoin',
                    'model_id': 'NMC_Bidirectional_LSTM_60_lags_4_fts_prod',
                    'test_days': 365,
                    'lags': 60,
                    'n_features': 4,
                    'features': ['Volume', 'Close'],
                    'treasury_ticker': '^FVX',
                    'google_trend': True,
                    'ordered_features': ['Volume', 'Gtrend', 'FVX', 'BTC', 'Close'],
                    'builder_func': build_BLSTM,
                    'builder_kwargs': dict(
                                        num_rnns=2,
                                        dim_rnn=16, 
                                        dense_units=50, 
                                        drop=True,
                                        drop_rate=0.3
                                    ),
                    'about': html.Div([
                        html.P([
                            "Bidirectional ",
                            html.A('Long Short Term Memory Neural Network',
                                href='https://en.wikipedia.org/wiki/Long_short-term_memory',
                                target="_blank",
                                style={'color': 'white'}),
                            """
                                using 60 lags of multiple features to forecast the closing 
                                price of the cryptocurrency.
                            """], ),
                        html.P(
                            """
                                It consumes public data on five features until the last known daily close. 
                                Historical Closing Price values constitute the autoregressive nature of the model, 
                                historical traded volume is used as an additional market signal, 
                                the historical U.S. Treasury Yield at the most relevant maturity among 5 (FVX), 10 (TNX) 
                                or 30 (TYX) years is included as a way of capturing international inflation and market risk signals. 
                                The daily social interest in the cryptocurrency as measured by Google Trends (Gtrend) takes 
                                into account potential investors' and general public perception on the asset. Finally, the price of Bitcoin (BTC)
                                is also included as the most representative asset of this kind.
                            """),
                        html.P("""Bidirectional LSTM layers capture the sequential structure of the data to predict next day's 
                                closing price. But they also read the sequences backwards, 
                                so that they are able to capture how the future shapes the past of a series.
                                """
                                )
                    ]),
                    'importance_insights':html.Div([
                        html.P([
                            """
                            The forecast for ETH-Ethereum next day price is mainly explained by the 10 most recent lags of the price itself. 
                            However, when we consider more recent periods, like the last few weeks, 
                            LIME shows how other factors such as risk free rates, inflation and internet
                            searches have started to contribute to the price prediction.
                            """], ),
                        ])
                },
                # another time scope
        },

    },
        
}
