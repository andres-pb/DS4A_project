import pandas as pd
import datetime as dt           
from pytrends.request import TrendReq


class GoogleTrends:

    def __init__(self) -> None:
        self.pytrend = TrendReq()
    
    def get_daily_trend_df(self, kw_list:list, start_date:str='2022-01-01', end_date:str='2022-06-01'):

        start_dt = dt.datetime.strptime(start_date, r'%Y-%m-%d')
        end_dt = dt.datetime.strptime(end_date, r'%Y-%m-%d')
        
        hourlytrend_df = self.pytrend.get_historical_interest(
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
            sleep=0
        )

        # Google trends gives us hourly data, conver to daily
        hourlytrend_df = hourlytrend_df.iloc[:, :-1]
        daily_df = hourlytrend_df.groupby(pd.Grouper(freq='d')).mean()

        # Our feature is the trend difference with respect to the daily mean
        daily_df = (daily_df - daily_df.mean())/daily_df.mean()

        return daily_df


# Test
gt = GoogleTrends()
btc_trend = gt.get_daily_trend_df(['Bitcoin', 'Etherium'])
print(btc_trend)


