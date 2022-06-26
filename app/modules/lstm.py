import pandas as pd
import numpy as np
from . import Statistical

import datetime as dt
import matplotlib.pyplot as plt
from os import environ
import tensorflow as tf
from tensorflow import keras
import joblib

from keras.models import Sequential, Model
from keras.layers import LSTM, Bidirectional
from keras.layers import Dense, BatchNormalization, ReLU, MaxPooling2D
from keras.layers import Activation, Masking 
from keras.layers import Dropout, PReLU
from keras.layers import Conv1D, Input, SpatialDropout2D
from keras.layers import MaxPooling1D, Reshape, ConvLSTM2D, Conv2D
from keras.layers import Flatten, RepeatVector
from keras.layers import TimeDistributed, Permute, Concatenate, Multiply, Lambda
from keras.constraints import maxnorm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from keras.callbacks import EarlyStopping
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, model_from_json

import plotly.express as px
import plotly.graph_objects as go


#Long Short Term Memory Ivan
def LSTM_model(ticker: str, number_prediction: int):
    # importing required libraries
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM

    status, value = Statistical(ticker).close()
    if not status: return None
    df=pd.DataFrame({'Date':value.index, 'Close':value.values})
    
    # creating dataframe
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
    for i in range(0, len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]

    # setting index
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)

    # creating train and test sets
    dataset = new_data.values

    train = dataset[0:len(dataset)-0, :]
    valid = dataset[len(dataset)-0:, :]
    # converting dataset into x_train and y_train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []
    for i in range(60, len(train)-number_prediction):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i+number_prediction, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,
                   input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

    # predicting 246 values, using past 60 from the train data
    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, inputs.shape[0]+1):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    return closing_price


# Functions to define NN with different architectures
def build_LSTM(in_shape, 
               num_rnns=1, 
               dim_rnn=128,
               dense_units=100,
               num_ysteps=1, 
               lr=1e-3,
               decay=6e-8,
               drop=False,
               drop_rate=0.2,
               masking=False,
               mv=-np.inf,
               model_name='LSTM',
               summarize: bool = False
               ):
  
  """Builds a classifier with LSTM layers with the given parameters

      Parameters:
      - in_shape: the constant shape of the input data X
      - num_rnns: how many lstm layers to add
      - dim_rnns: the number of units for each lstm layer
      - dense_units: number of units for the dense layer that goes after lstms
      - num_ysteps: how many time steps of the target to predict
      - drop: would you like to use dropout layers to prevent overfitting?
      - **masking: do not set this parameter to True because you cannot
                  manipulate the data to use a masking value.
      - extra Neural Network training parameters...

      Returns a keras model for classification that has not yet been trained"""

  # sequence input
  in_seq = Input(shape=in_shape, name='sequential_input')

  if masking:
    # tell rnns to ignore masked-padded timesteps
    x = Masking(mask_value=mv)(in_seq)
  else:
    x = in_seq

  lstm_out = LSTM(dim_rnn, activation='relu',
                  return_sequences=True)(x)
  if drop:
    lstm_out = Dropout(drop_rate)(lstm_out)

  rnn_layers = 1
  while rnn_layers < num_rnns:
    rnn_layers +=1
    rs = not (rnn_layers == num_rnns)
    lstm_out = LSTM(dim_rnn, activation='relu',
                    return_sequences=rs)(lstm_out)
  if drop:
    lstm_out = Dropout(drop_rate)(lstm_out)

  xx = Dense(dense_units, activation='relu')(lstm_out)
  pred_out = Dense(num_ysteps, activation='relu')(xx)

  # Define Model
  lags = in_shape[0]
  features = in_shape[1]
  predictor = Model(in_seq, pred_out, name=model_name + '_' + str(lags) + '_lags_' + str(features) + '_fts')
  # compile model
  opt = Adam(learning_rate=lr, decay=decay)
  predictor.compile(loss='mse', 
                    optimizer=opt, 
                    metrics=['mse'])
  if summarize:
    predictor.summary()

  return predictor


def build_BLSTM(in_shape, 
                num_rnns=1, 
                dim_rnn=128,
                dense_units=100, 
                num_ysteps=1,
                lr=1e-3, 
                decay=6e-8,
                drop=False,
                drop_rate=0.2,
                masking=False,
                mv=-np.inf,
                model_name='Bidirectional_LSTM',
               summarize: bool = False
                ):

  """Builds classifier with Bidirectional LSTM layers from the given parameters

    Parameters:
    - in_shape: the constant shape of the input data X
    - num_rnns: how many lstm layers to add
    - dim_rnns: the number of units for each lstm layer
    - dense_units: number of units for the dense layer that goes after lstms
    - num_ysteps: how many timesteps of the target to predict
    - drop: would you like to use dropout layers to prevent overfitting?
    - extra Neural Network training parameters...

    Returns a keras model for classification that has not yet been trained"""

  # sequence input
  in_seq = Input(shape=in_shape, name='sequential_input')

  if masking:
    # tell rnns to ignore masked-padded timesteps
    x = Masking(mask_value=mv)(in_seq)
  else:
    x = in_seq
  lstm_out = Bidirectional(LSTM(dim_rnn, activation='relu',
                                return_sequences=True))(x)
  rnn_layers = 1
  while rnn_layers < num_rnns:
    rnn_layers +=1
    rs = not (rnn_layers == num_rnns)
    lstm_out = Bidirectional(LSTM(dim_rnn, activation='relu',
                                  return_sequences=rs))(lstm_out)
  if drop:
    lstm_out = Dropout(drop_rate)(lstm_out)

  xx = Dense(dense_units, activation='relu')(lstm_out)
  pred_out = Dense(num_ysteps, activation='relu')(xx)

  # Define Model
  lags = in_shape[0]
  features = in_shape[1]
  predictor = Model(in_seq, pred_out, name=model_name + '_' + str(lags) + '_lags_' + str(features) + '_fts')
  # compile model
  opt = Adam(learning_rate=lr, decay=decay)
  predictor.compile(loss='mse', 
                     optimizer=opt, metrics=['mse'])
  if summarize:
    predictor.summary()

  return predictor


def build_AttentiveBLSTM(
    in_shape,    # (num_timesteps, num_features)
    num_rnns=1, 
    dim_rnn=64, 
    num_ysteps=1, 
    lr=1e-3, 
    decay=6e-8,
    drop=False,
    drop_rate=0.2,
    masking=False,
    mv=-np.inf,
    model_name='Attentive_BLSTM',
    suffix='',
    summarize: bool = False):
  
  """Builds classifier with Bidirectional LSTM layers and Attention 
      implementation from the given parameters

     Parameters:
     - in_shape: Tuple (num_timesteps, num_features)
                 the constant shape of each sample in the input data X
     - num_rnns: how many lstm layers to add
     - dim_rnns: the number of units for each lstm layer
     - dense_units: number of units for the dense layer that goes after lstms
     - num_ysteps: how many timesteps of the target to predict
     - drop: would you like to use dropout layers to prevent overfitting?
     - extra Neural Network training parameters...
     Returns a keras model for classification that has not yet been trained"""
  
  # sequence input
  in_seq = Input(shape=in_shape, name='sequential_input')
  
  if masking:
    # tell rnns to ignore masked-padded timesteps
    x = Masking(mask_value=mv)(in_seq)
  else:
    x = in_seq
  lstm_out = Bidirectional(LSTM(dim_rnn, return_sequences=True,input_shape=in_shape))(x)
  rnn_layers = 1
  if drop:
    lstm_out = Dropout(drop_rate)(lstm_out)
  while rnn_layers < num_rnns:
    rnn_layers +=1
    lstm_out = Bidirectional(LSTM(dim_rnn, return_sequences=True))(lstm_out)   
  if drop:
    lstm_out = Dropout(drop_rate)(lstm_out)
  # adding attention
  e = Dense(1, activation='tanh')(lstm_out)
  e = Flatten(data_format=None)(e)
  a = Activation('softmax')(e)
  temp = RepeatVector(2*dim_rnn)(a)
  temp = Permute([2, 1])(temp)
  # multiply weights with lstm output
  att_out = Multiply()([lstm_out, temp])
  # attention adjusted output state
  att_out = Lambda(lambda values: K.sum(values, axis=1))(att_out)
  pred_out = Dense(num_ysteps, activation='relu')(att_out)

  # Define Model
  lags = in_shape[0]
  features = in_shape[1]
  predictor = Model(in_seq, pred_out, name=model_name + '_' + str(lags) + '_lags_' + str(features) + '_fts' + '_' + suffix)
  # compile model
  opt = Adam(learning_rate=lr, decay=decay)
  predictor.compile(loss='mae', 
                     optimizer=opt, metrics=['mse'])
  if summarize:
    predictor.summary()

  return predictor


# Data preprocessing functions
def build_dset(n_coins: int, gtrends: bool = False):

    """
    Builds a dataframe with price and volume daily data from yahoofinance for each of a given
    number of coins and returns a dictionary with the coin name as the key and the dataframe as the value.
    The dataframe also cointains the risk free rate as the US treasury bonds yield at different maturities,
    which is thought to capture the effect of inflation and inflation expectations that might have certain effect
    on crypto prices.
    Coins are sorted by market cap and days of history available.
    """

    # Get top n_coins coins by mktcap
    load_dotenv()
    mktcap_key = environ.get('COIN_MKTCAP_KEY')
    cmc = CoinMarketCap(mktcap_key)
    coins_df = cmc.get_top_coins(n_coins)

    # Daily US yield curve to account for interest rate (inflation) effect
    yield_curve = ['^FVX', '^TNX', '^TYX']
    ir_dfs = []
    for ir in yield_curve:
        min_date = coins_df['first_historical_data'].min()
        status, ir_data = yahoo_finance.market_value(ir, hist=min_date, interval='1d')
        ir_data.rename(columns={'Close': ir[1:]}, inplace=True)
        ir_close =  ir_data[ir[1:]]
        ir_dfs.append(ir_close)
    us_treasury = pd.concat(ir_dfs, axis=1)

    # Get daily market data for top 10 coins
    data_dict = {}
    for idx, coin in coins_df.iterrows():
        coin_name = coin['name']
        ticker = coin['symbol'] + '-USD'
        start = coin['first_historical_data']

        # Get daily market data from yahoo
        status, data = yahoo_finance.market_value(ticker, hist=start, interval='1d')
        print(coin_name, coin['days_history'])
        print(data.shape)
        # Add yield curve data
        data = data.merge(us_treasury, how='left', left_index=True, right_index=True)
        xcols = ['High', 'Low', 'Volume'] + [s[1:] for s in yield_curve]

        if gtrends:
            gt = GoogleTrends()
            # Get daily google trend data
            start_str = dt.datetime.strftime(start, r'%Y-%m-%d')
            end_str = dt.datetime.strftime(dt.datetime.today(), r'%Y-%m-%d')
            gtrend = gt.get_daily_trend_df([coin_name], start_date=start_str, end_date=end_str)
            data = data.merge(gtrend, how='left', left_index=True, right_index=True)
            data = data.rename(columns={coin_name: 'Google_Trend'})
            xcols = ['High', 'Low', 'Volume'] + [s[1:] for s in yield_curve] + ['Google_Trend']

        data = data.fillna(method='ffill')[xcols + ['Close']]
        data_dict[coin_name] = data
        
    return data_dict


def select_features(data_dict, keep=['Volume', 'Gtrend']):
    """
    Simple feature selection rule for each coin.
    Keep the volume and select the US treasury bond maturity with the highest correlation to the target.
    """

    for coin, dset in data_dict.items():
        selected_features = keep
        corr = dsets[coin].corr()
        yield_curve = ['FVX', 'TNX', 'TYX']
        corr_yield = np.abs(corr[yield_curve])
        ir_index = corr_yield.loc['Close', :].argmax()
        selected_features = keep + [yield_curve[ir_index]]
        
        data_dict[coin] = dset[list(set(selected_features)) + ['Close']]


def series_to_supervised(df, n_in=1, n_out=1, target_idx=-1, 
                         dropnan=True, min_input=None):

    """
    Takes the dataframe and sets as many as n_in lags of the features as columns. (Reframes the df)
    This is required for LSTM neural networks which take 3D input (n_samples, n_lags, n_features).
    """
    
    df = df.copy(deep=True)
    n_in = int(n_in)
    min_input = int(min_input)
    target = df.columns[target_idx]
    df[target + '_t'] = df[target]
    n_vars = df.shape[1] - 1
    
    vars = list(df.columns)
    vars.remove(target)
    cols, names = list(), list()
    # input sequences (t-n, ... t-1)
    for i in range(n_in, 0, -1):
      cols.append(df.iloc[:,:target_idx].shift(i))
      names += [(var + '(t-%d)' % i) for var in vars]
    # current features (t)
    cols.append(df)
    names += [(var + '(t)') for var in vars]
    names += [target + '(t)']
     # forecast sequence (t, t+1, ... t+n)
    for i in range(1, n_out):
      cols.append(df[target].shift(-i))    
      names += [target + '(t+%d)' % i]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan and min_input:
      agg.dropna(inplace=True, thresh=n_vars*(min_input+1))
    agg.drop(columns=[(var + '(t)') for var in vars], inplace=True)
    return agg


def prep_data(df, timesteps, test_days=365, scaler=None):
    """
    Final data preparations: 
    - extracts np.arrays from df,
    - rescales the data with a minmax scaler,
    - reshapes data into the 3D sequential arrays as required by LSTMs
    - Train-Test split using the last test_days parameter as training samples.
    """

    reframed = series_to_supervised(df, n_in=timesteps, min_input=timesteps)
    seqs = reframed.values 

    if not scaler:
        scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_seqs = scaler.fit_transform(seqs)

    X = scaled_seqs[:,:-1].reshape((seqs.shape[0], 
                            timesteps, 
                            int(seqs.shape[1]/timesteps)
                            ))
    y = scaled_seqs[:, -1]

    X_train, X_test = X[:-test_days, :, :], X[-test_days:, :, :]
    y_train, y_test = y[:-test_days], y[-test_days:]

    test_dates = df.iloc[-test_days:, :].index
    
    return X_train, y_train, X_test, y_test, scaler, test_dates


# To build and train a model
def train_model(
    prep_dsets: dict, 
    coin_name: str,
    model_builder,
    n_epochs=1,
    batch_size=1,
    early_stop: bool = False,
    patience: int = None,
    model_kwargs: dict = {}):

    X_train, y_train, X_test, y_test, _, _ = prep_dsets[coin_name]
    # Build lstm model
    model = model_builder(
                    in_shape=(X_train.shape[1], X_train.shape[2]),
                    **model_kwargs
                    )
    if early_stop and patience:
        # This EarlyStopping callback stops training once it stops improving
        # so that you can set a high number of epochs and let it choose when to stop
        monitor = EarlyStopping(monitor='val_loss', 
                                min_delta=1e-3, 
                                patience=patience, 
                                verbose=0, 
                                mode='auto', 
                                restore_best_weights=True)
    # Train the model
    history = model.fit(
                    X_train, y_train, 
                    validation_data=(X_test, y_test),
                    callbacks=[monitor] if early_stop else None,
                    verbose=2, 
                    epochs=n_epochs,
                    batch_size=batch_size
                    )
    # visualize training
    plt.plot(history.history['mse'], label='train')
    plt.plot(history.history['val_mse'], label='test')
    plt.legend()
    plt.show()

    return model


# To build test predictions dataframe
def gen_test_df(model, X_test, y_test, scaler, test_dates, model_id, ticker, pred_scope, lags):
    
    predicted = model.predict(X_test)
    X_test_2d = X_test.reshape((X_test.shape[0], lags * X_test.shape[2]))
    xpred_2d = np.append(X_test_2d, predicted, axis=1)
    pred_y = scaler.inverse_transform(xpred_2d)[:, -1]
    xtrue_2d = np.append(X_test_2d, y_test.reshape(predicted.shape), axis=1)
    true_y = scaler.inverse_transform(xtrue_2d)[:, -1]

    test_df = pd.DataFrame()
    test_df['Observed'] = true_y
    test_df['Predicted'] = pred_y
    test_df['Ticker'] = ticker
    test_df['Model'] = model_id
    test_df['scope'] = pred_scope
    test_df['Date'] = test_dates

    return test_df


# To load filtered test predictions
def load_test_df(path, ticker, model_id, pred_scope):
    test_df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    test_df = test_df.query('Scope==@pred_scope & Ticker==@ticker & Model==@model_id')
    return test_df.sort_index()


# To build feature importance dataframe
def gen_importance_df(model, dset, timesteps, ticker, model_id, pred_scope):

    reframed = series_to_supervised(dset, n_in=timesteps, min_input=60)
    feature_names = reframed.iloc[:, :-1].columns
    feature_names = list(feature_names)
    _, _, X_test, y_test, _, _ = prep_data(dset, timesteps)
    results = []

    mae_err = mean_absolute_error
    mse_err = mean_squared_error 
    oof_preds = model.predict(X_test, verbose=0).squeeze() 
    baseline_mae = mae_err(y_test, oof_preds)
    baseline_mse = mse_err(y_test, oof_preds)

    np.random.seed(888)
    for k in range(timesteps):
        for f in range(3):
            # SHUFFLE LAG k OF FEATURE f
            save_col = X_test[:,k,f].copy()
            np.random.shuffle(X_test[:,k,f])
            current_ft = feature_names.pop(0)
            # COMPUTE OOF MAE WITH FEATURE K SHUFFLED
            oof_preds = model.predict(X_test, verbose=0).squeeze() 
            mae = mae_err(y_test, oof_preds) - baseline_mae
            results.append(
                {'Feature':current_ft,
                'Importance':mae,
                'Metric': 'mae'
                })
            mse = mse_err(y_test, oof_preds) - baseline_mse
            results.append(
                {'Feature':current_ft,
                'Importance':mse,
                'Metric': 'mse'
                })
            X_test[:,k,f] = save_col

    importance_df['Ticker'] = ticker
    importance_df['Model'] = model_id
    importance_df['scope'] = pred_scope
    importance_df = pd.DataFrame(results).sort_values(['Ticker', 'Scope', 'Model', 'Metric', 'Importance'])


    return importance_df


#To save and load models
def save_model(model, full_path: str= None, path="./models/", coin_ticker='BTC'):
    
    # serialize model
    model_json = model.to_json()
    if not full_path:
        full_path = path + coin_ticker + '_' + model.name + '.json'
    with open(full_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights
    weights_path = full_path.replace('.json', '.h5')
    model.save_weights(weights_path)


def save_scaler(fitted_scaler, full_path: str= None, path="./models/", coin_ticker='BTC'):
    if not full_path:
        full_path = path + coin_ticker + '_scaler.gz'
    joblib.dump(fitted_scaler, full_path)


def load_scaler(full_path: str= None, path="./models/", coin_ticker='BTC'):
    
    if not full_path:
        full_path = path + coin_ticker + '_scaler.gz'
    my_scaler = joblib.load(full_path)
    return my_scaler


def load_model(full_path: str= None, path="./models/", coin_ticker='BTC', model_name='blstm'):
    
    if not full_path:
    # load json and create model
        full_path = path + coin_ticker + '_' + model_name + '.json'
    json_file = open(full_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    weights_path = full_path.replace('.json', '.h5')
    loaded_model.load_weights(weights_path)

    return loaded_model

