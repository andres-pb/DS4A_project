import coinmarketcapapi
import pandas as pd
from os import environ
from dotenv import load_dotenv
import datetime as dt


class CoinMarketCap:
    
    def __init__(self, key) -> None:
        self.key = key
        self.api = coinmarketcapapi.CoinMarketCapAPI(self.key)


    def get_top_coins(self, top_n:int=50, sort_col:str='rank', sort_asc:bool=True, min_days:int=2400):

        data_id_map = self.api.cryptocurrency_map().data
        topcoins_df = pd.DataFrame.from_records(data_id_map)

        # Require a minimum of history days
        topcoins_df['first_historical_data'] = pd.to_datetime(topcoins_df['first_historical_data'])
        topcoins_df['last_historical_data'] = pd.to_datetime(topcoins_df['last_historical_data'])
        topcoins_df['days_history'] = topcoins_df['last_historical_data'] - topcoins_df['first_historical_data']
        topcoins_df.drop(columns=['platform', 'last_historical_data', 'first_historical_data'], inplace=True)
        min_hist = dt.timedelta(days=min_days)
        topcoins_df = topcoins_df[topcoins_df['days_history'] >= min_hist]

        # Sort by importance
        topcoins_df.sort_values(by=sort_col, ascending=sort_asc, inplace=True)
        topcoins_df = topcoins_df.query('is_active==1')
        topcoins_df = topcoins_df.iloc[:top_n, :].reset_index()

        # Add queries for twitter api
        topcoins_df['hash_symbol'] = '#' + topcoins_df['symbol']
        topcoins_df['hash_slug'] = '#' + topcoins_df['slug']
        keyword_cols = ['symbol', 'hash_symbol', 'name', 'slug', 'hash_slug']
        topcoins_df['twt_query'] = topcoins_df[keyword_cols].agg(' OR '.join, axis=1)

        return topcoins_df


# Test
load_dotenv()
mktcap_key = environ.get('COIN_MKTCAP_KEY')
cmc = CoinMarketCap(mktcap_key)
coins_df = cmc.get_top_coins()
# Save top 50 coins metadata
coins_df.to_csv('presel_coins.csv', index=False)
print(coins_df)