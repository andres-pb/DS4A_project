import yfinance as yf
import datetime as dt
import time
from typing import Tuple
import pandas as pd
import logging


class YahooFinance:
    def __init__(self) -> None:
        self._LOG = logging.getLogger(__name__)
        self.today = (dt.datetime.today()+dt.timedelta(1)).strftime('%Y-%m-%d')
        self.hist = (dt.datetime.today()-dt.timedelta(900)).strftime('%Y-%m-%d')
        return None

    def market_value(self, symbol: str, today: dt = None, hist: dt = None, interval: str ='1wk') -> Tuple[bool, pd.DataFrame]:
        """
        intervals --->1d,5d,1wk,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        """
        count = 0
        today = today if today else self.today
        hist = hist if hist else self.hist
        while True:
            stock_market_value = yf.download(symbol, interval=interval, start=hist, end=today, threads=False, timeout=15).dropna()
            if stock_market_value.empty:
                count += 1
                if count > 3:
                    self._LOG.debug(f"There is not response because it was tried {count}, and  was not found any information")
                    return False, stock_market_value
                time.sleep(5)
            else: break
        return True, stock_market_value 
    