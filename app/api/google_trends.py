import pandas as pd
import datetime as dt
import logging
from pytrends.request import TrendReq

__LOG = logging.getLogger(__name__)

class GoogleTrends:

    def __init__(self) -> None:
        self.pytrend = TrendReq()
        self.today = dt.date.today()
        self.week_ago = self.today - dt.timedelta(days=7)
    
    def get_daily_trend_df(self, kw_list:list, start_dt: dt.date = None , end_dt: dt.date = None):
        
        start_dt = self.week_ago if not start_dt else start_dt
        end_dt  = self.today if not end_dt else end_dt
        
        daily_df = self.pytrend.get_historical_interest(
            kw_list, 
            year_start=start_dt.year,
            month_start=start_dt.month, 
            day_start=start_dt.day,
            hour_start=0,
            year_end=end_dt.year,
            month_end=end_dt.month,
            day_end=end_dt.day,
            hour_end=0,
            cat=0,
            geo='',
            gprop='',
            sleep=60,
            frequency='daily'
        )
        daily_df.reset_index(inplace=True)
        daily_df['date'] = daily_df['date'].dt.date
        daily_df.set_index('date', inplace=True)
        
        return daily_df


# Test
""" gt = GoogleTrends()
btc_trend = gt.get_daily_trend_df(['Bitcoin', 'Etherium'])
print(btc_trend) """