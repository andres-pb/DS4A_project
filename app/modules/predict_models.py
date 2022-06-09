import numpy as np
import pandas as pd
from typing import List
from . import Statistical
import logging

_LOG = logging.getLogger(__name__)
class Predict:
    def __init__(self) -> None:
        pass

    
    def ARIMA(self, ticker: str, number_prediction:int) -> List[float]:
        from statsmodels.tsa.arima.model import ARIMA
        status, value = Statistical(ticker).close()
        if not status: return None
        model_fit = ARIMA(value.values, order=(0, 0, 1)).fit()
        _LOG.debug("ARIMA Calculated")
        return model_fit.predict(len(value), len(value)+number_prediction)
    
    #Autoregressive
    def AR(self, ticker: str, number_prediction:int) -> List[float]:
        from statsmodels.tsa.ar_model import AutoReg
        status, value = Statistical(ticker).close()
        if not status: return None
        model_fit = AutoReg(value.values, lags=1, seasonal=True, period=12).fit()
        _LOG.debug("AR Calculated")
        return model_fit.predict(len(value), len(value)+number_prediction)
     
    #Exponential Smoothing
    def EX(self, ticker: str, number_prediction:int) -> List[float]:
        from statsmodels.tsa.api import ExponentialSmoothing
        status, value = Statistical(ticker).close()
        if not status: return None
        model_fit = ExponentialSmoothing(value.values, initialization_method='estimated').fit(smoothing_level=0.6, optimized=False)
        _LOG.debug("AR Calculated")
        return model_fit.predict(len(value), len(value)+number_prediction)
     
    #Simple Exponential Smoothing
    def SEX(self, ticker: str, number_prediction:int) -> List[float]:
        from statsmodels.tsa.api import SimpleExpSmoothing
        status, value = Statistical(ticker).close()
        if not status: return None
        model_fit = SimpleExpSmoothing(value.values, initialization_method='heuristic').fit(smoothing_level=0.6, optimized=False)
        _LOG.debug("AR Calculated")
        return model_fit.predict(len(value), len(value)+number_prediction)
    
    #Simple Exponential Smoothing
    def HOLT(self, ticker: str, number_prediction:int) -> List[float]:
        from statsmodels.tsa.api import Holt
        status, value = Statistical(ticker).close()
        if not status: return None
        model_fit = Holt(value.values, initialization_method='estimated').fit()
        _LOG.debug("AR Calculated")
        return model_fit.predict(len(value), len(value)+number_prediction)
    

    def HOLTWINTER(self, ticker: str, number_prediction:int) -> int:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        status, value = Statistical(ticker).close()
        if not status: return None
        model_fit = ExponentialSmoothing(value.values).fit()
        _LOG.debug("Holt-Winter Calculated")
        return model_fit.predict(len(value), len(value)+number_prediction)