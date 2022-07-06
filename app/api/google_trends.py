import pandas as pd
import datetime as dt
import logging
import time
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

        if (end_dt - start_dt).days > 15:
        
            # Daily is faster but doesnt bring data from last days
            daily_df = self.pytrend.get_historical_interest(
                kw_list, 
                year_start=start_dt.year,
                month_start=start_dt.month, 
                day_start=start_dt.day,
                hour_start=0,
                year_end=end_dt.year,
                month_end=end_dt.month,
                day_end=end_dt.day,
                hour_end=23,
                cat=0,
                geo='',
                gprop='',
                sleep=60,
                frequency='daily'
            )

            daily_df.reset_index(inplace=True)
            # Los isPartial vienen en ceros
            daily_df = daily_df[~daily_df['isPartial']]

            daily_df['date'] = daily_df['date'].dt.date
            # De aqui en adelante usamos hourly para aproximar
            date_ok = daily_df['date'].max() + dt.timedelta(1)

            daily_df.set_index('date', inplace=True)

            if date_ok < end_dt:
                time.sleep(60)
                lastweek_df = self.pytrend.get_historical_interest(
                    kw_list, 
                    year_start=date_ok.year,
                    month_start=date_ok.month, 
                    day_start=date_ok.day,
                    hour_start=0,
                    year_end=end_dt.year,
                    month_end=end_dt.month,
                    day_end=end_dt.day,
                    hour_end=23,
                    cat=0,
                    geo='',
                    gprop='',
                    sleep=60,
                    frequency='hourly'
                )
                
                if 'date' not in list(lastweek_df.columns):
                    lastweek_df.reset_index(inplace=True)
                lastweek_df['date'] = lastweek_df['date'].dt.date
                lastweek_df = lastweek_df[['date'] + kw_list].groupby('date').mean().reset_index()
                lastweek_df.set_index('date', inplace=True)
                daily_df = pd.concat([daily_df, lastweek_df], ignore_index=False)
                daily_df = daily_df.groupby(daily_df.index).first().drop(columns=['isPartial'])
            return daily_df
        
        else:
            day_df = self.pytrend.get_historical_interest(
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
                frequency='hourly'
            )
            day_df.reset_index(inplace=True)
            # Los isPartial vienen en ceros
            day_df = day_df[~day_df['isPartial']]
            day_df['date'] = day_df['date'].dt.date
            ###
            day_df = day_df[['date'] + kw_list].groupby('date').mean().reset_index()
            day_df.set_index('date', inplace=True)
            
            return day_df

    
    def get_last_day_df(self, kw_list:list):
        
        start_dt = self.today
        end_dt  = self.today + dt.timedelta(1)
        
        # Daily is faster but doesnt bring data from last days
        day_df = self.pytrend.get_historical_interest(
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
            frequency='hourly'
        )
        day_df.reset_index(inplace=True)
        # Los isPartial vienen en ceros
        day_df = day_df[~day_df['isPartial']]
        day_df['date'] = day_df['date'].dt.date
        ###
        day_df = day_df[['date'] + kw_list].groupby('date').mean().reset_index()
        day_df.set_index('date', inplace=True)
        
        return day_df


# Test
""" gt = GoogleTrends()
btc_trend = gt.get_daily_trend_df(['Bitcoin', 'Etherium'])
print(btc_trend) """