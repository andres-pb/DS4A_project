"""
Use pre-trained models metadata to reproduce and load them.
Make predictions using new data.
"""
from pyexpat import features
from app.api import yahoo_finance, GoogleTrends
from app.modules.models_meta import pred_models
from app.modules.lstm import load_scaler, load_model, prep_data
import datetime as dt
import pandas as pd

def rebuild_model(
    models_meta, 
    coin_label, 
    model_label, 
    scope_label, 
    models_path, 
    scalers_path):

    print(models_meta)
    
    # Get production model metadata
    model_meta = models_meta[coin_label][model_label][scope_label]
    builder_func = model_meta['builder_func']
    builder_kwargs = model_meta['builder_kwargs']
    ticker = model_meta['ticker']
    coin_name = model_meta['coin_name']
    lags = model_meta['lags']
    features = model_meta['features']
    # Define a model that is still not trained
    rebuilt_model = builder_func(**builder_kwargs)
    # load models weight
    load_model(rebuilt_model, model_id=model_meta['model_id'], root_path=models_path)
    # load the scaler
    scaler = load_scaler(ticker, root_path=scalers_path)
    # get sample for prediction
    yf = yahoo_finance()
    # we go back as many days as needed lags, plus an extra just in case
    history = yf.today - dt.timedelta(days=lags + 1)
    # get treasury bonds price
    tr_ticker = model_meta['treasury_ticker']
    status, yield_df = yahoo_finance.market_value(tr_ticker + '-USD', hist=history)
    if status:
        yield_df.fillna(method='ffill', inplace=True)
        yield_data = yield_df.iloc[-lags:, :].rename(columns={'Close': tr_ticker[-3:]})[tr_ticker[-3:]]
        # get the last close
        status, close_df = yf.market_value(ticker, hist=history)
        if status:
            sample_df = close_df.iloc[-lags:, :][features]
            sample_df[tr_ticker[-3:]] = yield_data
            all_features = [tr_ticker[-3:]] + features
            use_gtrend = model_meta['google_trend']
            if use_gtrend:
                # Get daily google trend interest
                gt = GoogleTrends()
                gtrend_df = gt.get_daily_trend_df([coin_name], start_dt=history)
                gtrend_df.fillna(method='ffill', inplace=True)
                gtrend_df = gtrend_df.iloc[-lags:][coin_name]
                gtrend_df.set_index(sample_df.index, drop=True, inplace=True)
                gtrend_data = gtrend_df[coin_name]
                sample_df['Gtrend'] = gtrend_data
                all_features = [tr_ticker[-3:]] + ['Gtrend'] + features
            # Dataframe con la muestra
            sample_df = sample_df[all_features]
            print(sample_df.shape)

            # Extract and scale sample
            X = prep_data(sample_df, timesteps=lags, scaler=scaler, production=True)
            print(X.shape)
            # Make model prediction
            pred_y = rebuilt_model.predict(X)

            # undo scaling
            prediction = scaler.inverse_transform(pred_y)
            
            print(prediction)

            return True
        else:
            print('Error trying to get ticker {} data from Yahoo Finance.'.format(ticker))
            return False               
    else:
        print('Error trying to get treasury yield {} data from Yahoo Finance.'.format(tr_ticker))
        return False
"""     except KeyError:
        print('Requested model is not defined in models metadata dictionary.')
        return False, None, None """


if __name__=='__main__':
    rebuild_model(
        pred_models, 
        'BTC - Bitcoin', 
        'Deep Learning LSTM', 
        '1 day ahead',
        './app/dashboard/test_models/',
        './models/'
        )






