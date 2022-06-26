from app import build_LSTM, build_BLSTM, build_Attentive_BLSTM
from dash import html

pred_models = {

    'BTC-USD': {
        '1D': {
            'BTC_LSTM_VGC_1D': {
                'ticker': 'BTC-USD',
                'test_days': 365,
                'label': 'LSTM NN - Market and Google Trends',
                'lags': 60,
                'weights_path': '',
                'builder': build_LSTM,
                'model_kwargs': dict(
                                num_rnns=2,
                                dim_rnn=50, 
                                dense_units=100, 
                                drop=False, 
                                drop_rate=0.5),
                'about': html.Div([
                    html.P('Long Short Term Memory Neural Network using 60 lags of multiple features to forecast the closing price of the cryptocurrency.'),
                    html.P('It consumes public data of four features until the last known daily close. '
                            'Historical Closing Price values constitute the autoregressive nature of the model, '
                            'historical traded volume is used as an additional market signal, '
                            'the historical U.S. Treasury Yield at the most relevant maturity (5, 10 or 30 years) is included as a way of capturing international inflation and market risk signals. '
                            'Finally, the daily social interest in the cryptocurrency as measured by Google Trends is used to take into account alternative sources of data.')
                    ])
                }
        }
        
    },

}