"""
Use pre-trained models metadata to reproduce and load them.
Make predictions using new data.
"""
from pyexpat import features
from app.api import yahoo_finance, GoogleTrends
from app.modules.lstm import load_scaler, prep_data, load_model
from app.modules.models_meta import pred_models
import datetime as dt
import pandas as pd

from app.modules.lstm import build_LSTM, build_BLSTM
from dash import html, dcc







