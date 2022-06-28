from ..api import yahoo_finance
import pandas as pd
from pandas import DataFrame
import datetime as dt
#import talib
from typing import Tuple


class Statistical():pass
"""store_request=dict()
    time_restriction=60
    
    def __init__(self, ticker:str, today: dt = None, hist: dt = None, interval:str='1wk') -> None:
        self.ticker=ticker
        self.interval= interval
        self.today = today if today else (dt.datetime.today()+dt.timedelta(1)).strftime('%Y-%m-%d')
        self.hist = hist if hist else (dt.datetime.today()-dt.timedelta(900)).strftime('%Y-%m-%d')
        return None

    def request_external_data(self, fn, *args, **kwargs) -> any:
        if self.store_request.get(fn, None):
            if (dt.datetime.now() - self.store_request[fn]['time']).total_seconds() < self.time_restriction :
                return [True, self.store_request[fn]['data']]
        status, value =fn(*args)
        if status:
            self.store_request[fn]={'time':dt.datetime.now(), 'data':value}
            return [True, value]
        return [False, None]


    def volume(self) -> Tuple[bool, DataFrame]:
        status, value =self.request_external_data(
                                                    yahoo_finance.market_value, 
                                                    self.ticker, self.today,
                                                    self.hist, self.interval
                                                )
        if status:return [True, value['Volume']]
        return [False, None]


    def close(self) -> Tuple[bool, DataFrame]:
        status, value =self.request_external_data(
                                                    yahoo_finance.market_value, 
                                                    self.ticker, self.today,
                                                    self.hist, self.interval
                                                )
        if status:return [True, value['Close']]
        return [False, None]


    def candles(self) -> Tuple[bool, DataFrame]:
        status, value =self.request_external_data(
                                                    yahoo_finance.market_value, 
                                                    self.ticker, self.today,
                                                    self.hist, self.interval
                                                )
        if status:return [True, value['Open','High','Low','Close']]
        return [False, None]

    def rolling_mean(self) -> Tuple[bool, DataFrame]:
        status, value =self.request_external_data(
                                                    yahoo_finance.market_value, 
                                                    self.ticker, self.today,
                                                    self.hist, self.interval
                                                )
        if status:return [True, pd.Series(value['Close']).rolling(window=12).mean()]
        return [False, None]
    
    def rolling_std(self) -> Tuple[bool, DataFrame]:
        status, value =self.request_external_data(
                                                    yahoo_finance.market_value, 
                                                    self.ticker, self.today,
                                                    self.hist, self.interval
                                                )
        if status:return [True, pd.Series(value['Close']).rolling(window=12).std()]
        return [False, None]

    #Exponentially Weighted Moving Average
    def exponential_wma(self) -> Tuple[bool, DataFrame]:
        status, value =self.request_external_data(
                                                    yahoo_finance.market_value, 
                                                    self.ticker, self.today,
                                                    self.hist, self.interval
                                                )
        if status:return [True, value['Close'].ewm(halflife=200).mean()]
        return [False, None]

    #https://mrjbq7.github.io/ta-lib/funcs.html
    #Bollinger Bands
    def BBANDS(self) -> Tuple[bool, DataFrame]:
        status, value =self.request_external_data(
                                                    yahoo_finance.market_value, 
                                                    self.ticker, self.today,
                                                    self.hist, self.interval
                                                )
        if status:
            value['upperband'], value['middleband'], value['lowerband'] = talib.BBANDS(value['Close'],
                                                                                  timeperiod=20, nbdevup=2,
                                                                                  nbdevdn=2, matype=0)
            return [True, value[['upperband','middleband','lowerband']]]
        return [False, None]

    #Average Directional Movement Index
    def ADX(self) -> Tuple[bool, DataFrame]:
        status, value =self.request_external_data(
                                                    yahoo_finance.market_value, 
                                                    self.ticker, self.today,
                                                    self.hist, self.interval
                                                )
        if status:
            ADX=talib.ADX(value['High'], value['Low'], value['Close'], timeperiod=7)
            return [True, ADX]
        return [False, None]
    
    #Average True Range
    def ATR(self) -> Tuple[bool, DataFrame]:
        status, value =self.request_external_data(
                                                    yahoo_finance.market_value, 
                                                    self.ticker, self.today,
                                                    self.hist, self.interval
                                                )
        if status:
            ATR = talib.ATR(value['High'], value['Low'], value['Close'], timeperiod=14)
            return [True, ATR]
        return [False, None]
    
    #Moving Average Convergence/Divergence
    def MACD(self) -> Tuple[bool, DataFrame]:
        status, value =self.request_external_data(
                                                    yahoo_finance.market_value, 
                                                    self.ticker, self.today,
                                                    self.hist, self.interval
                                                )
        if status:
            MACD = talib.MACD(value['close'],fastperiod=12, slowperiod=26, signalperiod=9)
            return [True, MACD]
        return [False, None]"""