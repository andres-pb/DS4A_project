import json
import tweepy
from dotenv import load_dotenv
from os import environ
import datetime as dt
import pandas as pd


class Twitter:

    def __init__(self, key:str, secret:str, token:str, token_secret:str) -> None:
        self.key = key
        self.secret = secret
        self.token = token
        self.token_secret = token_secret
        auth = tweepy.OAuthHandler(api_key, api_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth)
    

    def get_tweets_df(self, query:str, count:int):

        results = self.api.search_tweets(
            q=query,
            count=count,
            lang='en',
            result_type='popular',
            include_entities=False
        )

        if not results:
            return pd.DataFrame()

        twitter_data = []
        for status in results:
            
            json_response = json.loads(json.dumps(status._json))

            tweet_data = {
                'id': json_response['id_str'],
                'created_at': json_response['created_at'],
                'text': json_response['text'],
                'truncated': json_response['truncated'],
                'retweets': json_response['retweet_count'],
                'user_name': json_response['user']['screen_name'],
                'user_verified': json_response['user']['verified'],
                'user_followers': json_response['user']['followers_count'],
            }

            twitter_data.append(tweet_data)

        return pd.DataFrame.from_records(twitter_data)
        


# Test
load_dotenv()
api_key = environ.get('TW_KEY')
api_secret = environ.get('TW_SECRET')
access_token = environ.get('TW_TOKEN')
access_token_secret = environ.get('TW_TOKEN_SECRET')

twt = Twitter(api_key, api_secret, access_token, access_token_secret)
#print(twt.get_tweets_df(query="#Bitcoin OR Bitcoin", count=3))

# Test for all top 50 coins
coins_df = pd.read_csv('presel_coins.csv')
tweets_dfs = []
for ticker in coins_df['symbol'].unique():
    q = coins_df[coins_df['symbol']==ticker]['twt_query'].values[0]
    ticker_tweets = twt.get_tweets_df(query=q, count=3)

    if ticker_tweets.size > 0:
        ticker_tweets['Ticker'] = ticker
        tweets_dfs.append(ticker_tweets)

print(pd.concat(tweets_dfs, ignore_index=True))



