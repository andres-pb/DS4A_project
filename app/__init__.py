import os, sys
from .modules import Statistical, Predict, LSTM_model
from .api import Polygon, Alphavantage, yahoo_finance, Twitter, GoogleTrends, CoinMarketCap
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

