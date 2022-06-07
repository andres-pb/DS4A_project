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
    

    def get_tweets_df(self, query:str, limit:int, popular:bool=False):
        """
        query: A UTF-8, URL-encoded search query of 500 
            characters maximum, including operators (AND OR). 
            Queries may additionally be limited by complexity.
        limit: Total number of tweets to return.
        """
        result_type = 'popular' if popular else 'mixed'
        results = tweepy.Cursor(
            self.api.search_tweets,
            q=query,
            count=100,
            lang='en',
            result_type=result_type,
            tweet_mode='extended'
        ).items(limit)

        if not results:
            return pd.DataFrame()

        twitter_data = []
        for status in results:

            json_response = json.loads(json.dumps(status._json))

            tweet_data = {
                'id': json_response['id_str'],
                'created_at': json_response['created_at'],
                'full_text': json_response['full_text'],
                'truncated': json_response['truncated'],
                'retweets': json_response['retweet_count'],
                'user_name': json_response['user']['screen_name'],
                'user_verified': json_response['user']['verified'],
                'user_followers': json_response['user']['followers_count'],
                'user_image_url': json_response['user']['profile_image_url']
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
print(twt.get_tweets_df(query="#Bitcoin OR Bitcoin", limit=20))

# Test for all top 50 coins
""" coins_df = pd.read_csv('presel_coins.csv')
tweets_dfs = []
for ticker in coins_df['symbol'].unique():
    q = coins_df[coins_df['symbol']==ticker]['twt_query'].values[0]
    ticker_tweets = twt.get_tweets_df(query=q, limit=3)

    if ticker_tweets.size > 0:
        ticker_tweets['Ticker'] = ticker
        tweets_dfs.append(ticker_tweets)

print(pd.concat(tweets_dfs, ignore_index=True)) """



