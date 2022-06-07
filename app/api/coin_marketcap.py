import coinmarketcapapi
import pandas as pd
from os import environ
from dotenv import load_dotenv


class CoinMarketCap:
    
    def __init__(self, key) -> None:
        self.key = key
        self.api = coinmarketcapapi.CoinMarketCapAPI(self.key)


    def get_top_coins(self, top_n=50):

        data_id_map = self.api.cryptocurrency_map().data
        topcoins_df = pd.DataFrame.from_records(data_id_map).drop(columns=[
                'platform', 'last_historical_data', 'first_historical_data'
                ]
            )
        topcoins_df.sort_values(by='rank', inplace=True)
        topcoins_df = topcoins_df.query('is_active==1')
        topcoins_df = topcoins_df.iloc[:top_n, :]

        # Add queries for twitter api
        topcoins_df['hash_slug'] = '#' + topcoins_df['slug']
        topcoins_df['twt_query'] = topcoins_df[['name', 'slug', 'hash_slug']].agg(' OR '.join, axis=1)
        return topcoins_df


# Test
load_dotenv()
mktcap_key = environ.get('COIN_MKTCAP_KEY')
cmc = CoinMarketCap(mktcap_key)
cmc.get_top_coins().to_csv('presel_coins.csv', index=False)
print(cmc.get_top_coins(10))