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

def load_npro_model(ticker, path_to_model):
    with open(ticker + '_' + path_to_model, 'rb') as f:
        m = pickle.load(f)
    return m

def get_npro_prediction(model, coin_label):
    
    ticker = coin_label[:3]
    coin_name = coin_label[6:]
    features = ['Close', 'Gtrend', 'TYX', 'Volume']
    # get sample for prediction
    yf = yahoo_finance
    # we go back as many days as needed lags, plus an extra just in case
    history = dt.datetime.today() - dt.timedelta(days=3)
    history_str = dt.datetime.strftime(history, '%Y-%m-%d')
    ticker_usd = ticker  + '-USD'
    # get treasury bonds price
    tr_ticker = '^TYX'
    status, yield_df = yf.market_value(tr_ticker, hist=history, interval='1d')
    if status:
        print('got treasury data')
        yield_df.fillna(method='ffill', inplace=True)
        yield_data = yield_df.iloc[:-1, :].rename(columns={'Close': tr_ticker[-3:]})[tr_ticker[-3:]]
        # get the last close with lags
        status, close_df = yf.market_value(ticker_usd, hist=history, interval='1d')

        if status:
            print('Successfully got coin mkt data')
            last_close = close_df['Close'].values[-1]
            sample_df = close_df.iloc[:-1, :][features]
            sample_df[tr_ticker[-3:]] = yield_data

            # query our database bc google trends takes about 1 minute to load results

            print('Getting google trends data...')
            conn = sql.connect('./database.db', detect_types=sql.PARSE_DECLTYPES)
            dbquery = """
                SELECT
                Date date,
                Gtrend {}
                FROM google_trend_hist
                WHERE Ticker = "{}"
                AND Date >= DATE("{}")
            """.format(coin_name, ticker_usd, history_str)
            gt_data_local = pd.read_sql(dbquery, conn)
            gt_data_local['date'] = pd.to_datetime(gt_data_local['date']).dt.date
            max_local = gt_data_local['date'].max()

            if max_local < dt.date.today():
                # Get daily google trend interest for the remaining days
                gt = GoogleTrends()
                gtrend_df_api = gt.get_daily_trend_df([coin_name], start_dt=max_local + dt.timedelta(days=1))
                gtrend_df_api.reset_index(inplace=True)

                gtrend_df = pd.concat([gt_data_local, gtrend_df_api]).reset_index(drop=True)
                print('CONCAT RESULT >>> \n')
                print(gtrend_df.info())
                gtrend_df.fillna(method='ffill', inplace=True)
            
                # Update our local database for later use
                gtrend_df_api.reset_index(inplace=True)
                gtrend_df_api.rename(columns={coin_name: 'Gtrend', 'date':'Date'}, inplace=True)
                ('>> FROM GT API >>>')
                print(gtrend_df_api.info())
                gtrend_df_api['Ticker'] = ticker_usd
                gtrend_df_api = gtrend_df_api[['Date', 'Ticker', 'Gtrend']]
                cursor = conn.cursor()
                for _, row in gtrend_df_api.iterrows():
                    insert_query = """
                        INSERT INTO google_trend_hist
                        (Date, Ticker, Gtrend)
                        VALUES
                        ({}, "{}", {})
                    """.format(row['Date'], row['Ticker'], row['Gtrend'])
                    cursor.execute(insert_query)
                    conn.commit()
                print('>> Google Trends local database updated.')
            
            else:
                gtrend_df = gt_data_local

            gtrend_df = gtrend_df.iloc[:-1, :]
            print('>> Successfully collected Google Trends data.')
            gtrend_df.set_index(sample_df.index, drop=True, inplace=True)
            gtrend_data = gtrend_df[coin_name]
            sample_df['Gtrend'] = gtrend_data
        
            # Dataframe con la muestra
            sample_df = sample_df[features]
            sample_df.fillna(method='ffill', inplace=True)
            # Make df with future date and predict with neural prophet
            future_df = m.make_future_df()
