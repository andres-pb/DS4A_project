import json
import tweepy
from dotenv import load_dotenv
from os import environ
import datetime as dt
import pandas as pd
import logging

__LOG = logging.getLogger(__name__)

class Twitter:
    def __init__(self) -> None:
        #__LOG.debug(f"Initializing Twitter Object")
        self.key = environ.get('TW_KEY')
        self.secret = environ.get('TW_SECRET')
        self.token = environ.get('TW_TOKEN')
        self.token_secret = environ.get('TW_TOKEN_SECRET')
        auth = tweepy.OAuthHandler(self.key, self.secret)
        auth.set_access_token(self.token, self.token_secret)
        self.api = tweepy.API(auth)
    

    def get_tweets_df(self, ticker:str, query:str, limit:int, popular:bool=False):
        """
        query: A UTF-8, URL-encoded search query of 500 
            characters maximum, including operators (AND OR). 
            Queries may additionally be limited by complexity.
        limit: Total number of tweets to return.
        popular: boolean, filter only most popular tweets.
        
        """
        #__LOG.debug(f"Getting Tweets information")
        twitter_data = []
        result_type = 'popular' if popular else 'mixed'
        results = tweepy.Cursor(
                                self.api.search_tweets,
                                q=query,
                                count=100,
                                lang='en',
                                result_type=result_type,
                                tweet_mode='extended'
                            ).items(limit)
        if not results:return pd.DataFrame()
        for status in results:
            json_response = json.loads(json.dumps(status._json))
            twitter_data.append({
                                    'id': json_response['id_str'],
                                    'created_at': json_response['created_at'],
                                    'full_text': json_response['full_text'],
                                    'truncated': json_response['truncated'],
                                    'retweets': json_response['retweet_count'],
                                    'user_name': json_response['user']['screen_name'],
                                    'user_verified': json_response['user']['verified'],
                                    'user_followers': json_response['user']['followers_count'],
                                    'user_image_url': json_response['user']['profile_image_url'],
                                    'ticker': ticker
                                })
        return pd.DataFrame.from_records(twitter_data)
        


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



