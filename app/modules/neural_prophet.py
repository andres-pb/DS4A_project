import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet, set_log_level, set_random_seed
from enum import auto
import pickle
from app.api import yahoo_finance, GoogleTrends
import datetime as dt
import sqlite3 as sql

set_log_level("ERROR")
set_random_seed(888)


def load_npro_model(ticker, path_to_models):
    with open(path_to_models + ticker + '_NeuralProphet.pkl', 'rb') as f:
        m = pickle.load(f)
    return m


def get_npro_prediction(coin_label, path_npro_models):
    
    ticker = coin_label[:3]
    coin_name = coin_label[6:]
    print('coin name', coin_name)
    features = ['Close', 'Gtrend', 'TYX', 'Volume']
    # get sample for prediction
    yf = yahoo_finance
    # we go back as many days as needed lags, plus an extra just in case
    history = dt.datetime.today() - dt.timedelta(days=3)
    history_str = dt.datetime.strftime(history, '%Y-%m-%d')
    ticker_usd = ticker  + '-USD'
    # get treasury bonds price
    tr_ticker = '^TYX'
    status, yield_df = yf.market_value(tr_ticker, hist=history - dt.timedelta(7), interval='1d')
    if status:
        print('got treasury data')
        yield_df.fillna(method='ffill', inplace=True)
        yield_data = yield_df.rename(columns={'Close': tr_ticker[-3:]})[tr_ticker[-3:]]
        # get the last close with lags
        status, close_df = yf.market_value(ticker_usd, hist=history, interval='1d')

        if status:
            print('Successfully got coin mkt data for npro')
            last_close = close_df['Close'].values[-1]
            sample_df = close_df.iloc[:, :][['Close', 'Volume',]]
            sample_df = sample_df.merge(yield_data, how='left', left_index=True, right_index=True)
            sample_df[tr_ticker[-3:]] = sample_df[tr_ticker[-3:]].fillna(method='bfill')
            sample_df[tr_ticker[-3:]] = sample_df[tr_ticker[-3:]].fillna(method='ffill')
            

            # query our database bc google trends takes about 1 minute to load results
            print('Getting google trends data npro...')
            
            # Get daily google trend interest for the remaining days
            gt = GoogleTrends()
            gtrend_df = gt.get_last_day_df([coin_name])
            print(gtrend_df.info())
            gtrend_df.fillna(method='ffill', inplace=True)
        
            gtrend_df = gtrend_df.iloc[:, :]
            print('>> Successfully collected Google Trends data.')
            gtrend_data = gtrend_df[coin_name]
            sample_df['Gtrend'] = gtrend_data
        
            # Dataframe con la muestra
            sample_df = sample_df[features]
            sample_df.fillna(method='ffill', inplace=True)
            sample_df.fillna(method='bfill', inplace=True)
            sample_df = sample_df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
            sample_df = sample_df[['ds', 'y', 'Gtrend', 'TYX', 'Volume']]

            # Make df with future date and predict with neural prophet
            m = load_npro_model(ticker=ticker, path_to_models=path_npro_models)
            future_df = m.make_future_dataframe(sample_df)
            predicted_df = m.predict(future_df)
            print('\n>>> Prediction DF: \n', predicted_df)
            prediction = predicted_df['yhat1'].values[-1]

            ret = (prediction - last_close)/last_close

            return prediction, ret
        else:
            print('Error trying to get ticker {} data from Yahoo Finance.'.format(ticker))
            return False, False               
    else:
        print('Error trying to get treasury yield {} data from Yahoo Finance.'.format(tr_ticker))
        return False, False

