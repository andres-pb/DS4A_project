import pandas as pd
import numpy as np
import re
from os import environ
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

#Load FinBERT model
finbert_model = BertForSequenceClassification.from_pretrained("./sentiment_models/finbert/finbert_model")
finbert_tokenizer = BertTokenizer.from_pretrained("./sentiment_models/finbert/finbert_tokenizer")
nlp_fin = pipeline("sentiment-analysis", model=finbert_model, tokenizer=finbert_tokenizer)

#Load FinancialBERT model
financialbert_model = BertForSequenceClassification.from_pretrained("./sentiment_models/financial_bert/financialbert_model")
financialbert_tokenizer = BertTokenizer.from_pretrained("./sentiment_models/financial_bert/financialbert_tokenizer")
nlp_financial = pipeline("sentiment-analysis", model=financialbert_model, tokenizer=financialbert_tokenizer)

#Load Polygon csv
polygon_prev = pd.read_csv("./sentiment_models/polygon_df_01_07_2022.csv")

class Sentiment_predict():

    def __init__(self) -> None:       
        pass
        
    def sentiment_finBERT(self, df: pd.core.frame.DataFrame, string_column: str):
        """
        This function executes sentiment prediction using FinBERT over a text column within a pandas data frame.
            Inputs:
            df: Pandas dataframe,
            string_column: name of the columng of interest ex: “column_1”
            output:
            A list with the predicted sentiment of each element of the pandas' column.
            It can be assigned to a new column as : df[new_column] =  sentiment_finBERT(df, “column_1”)
        """
        #Cleaning string: Removing URLS.
        clean_column = 'clean' + string_column
        clean_column_1 = clean_column + '_1'
        df[clean_column_1] = df[string_column].replace(r'([\w+]+\:\/\/)?([\w\d-]+\.)*[\w-]+[\.\:]\w+([\/\?\=\&\#\.]?[\w-]+)*\/?', ' ', regex = True) 
        df[clean_column] = df[clean_column_1].replace(r'[0-9]', ' ', regex = True) 
        df[clean_column] = df[clean_column].replace(re.compile('[@_!#$%^&*()<>?/\|}{~:]'), ' ', regex = True) 
        df[clean_column] = df[clean_column].replace(r"[^( A-Za-z.;,')]", ' ', regex = True) 
        df[clean_column] = df[clean_column].replace(r"\s+", ' ', regex = True) 
        df[clean_column] = df[clean_column].str.lower()

        #Iterate through pandas dataframe column
        sent_list = []
        for i in range(df.shape[0]):            
            if df[string_column].isnull()[i]:
                sent = np.nan
            else:
                try:
                    text = [df[clean_column_1].iloc[i]]
                    sent = list(nlp_fin(text)[0].values())[0]
                except:
                    try:
                        text = [df[clean_column].iloc[i]]
                        firstpart, secondpart = [text[0][:round(len(text[0])/2)]], [text[0][round(len(text[0])/2):]]
                        sent_1 = list(nlp_fin(firstpart)[0].values())[0]
                        sent_2 = list(nlp_fin(secondpart)[0].values())[0]
                        if sent_1 == sent_2:
                            sent = sent_1
                        elif sent_1 == "neutral":
                            sent = sent_2
                        elif sent_2 == "neutral":
                            sent = sent_1
                        #Deals with large text dividing it in 2 parts 
                        print(f"in row {i} the text is divided into two parts")
                    except RuntimeError: 
                        sent = np.nan
                        #Deals with large text dividing it in 2 parts 
                        print(f"in row {i} the text is too long, no sentiment predicted")
                        pass
            sent_list.append(sent)
        df.drop([clean_column_1, clean_column], axis = 1, inplace = True)
        return sent_list

    
    def sentiment_financialBERT(self, df: pd.core.frame.DataFrame, string_column: str):
        """
        This function executes sentiment prediction using FinBERT over a text column within a pandas data frame.
            Inputs:
            df: Pandas dataframe,
            string_column: name of the columng of interest ex: “column_1”
            output:
            A list with the predicted sentiment of each element of the pandas' column.
            It can be assigned to a new column as : df[new_column] =  sentiment_finBERT(df, “column_1”)
        """
        #Cleaning string: Removing URLS.
        clean_column = 'clean' + string_column
        clean_column_1 = clean_column + '_1'
        df[clean_column_1] = df[string_column].replace(r'([\w+]+\:\/\/)?([\w\d-]+\.)*[\w-]+[\.\:]\w+([\/\?\=\&\#\.]?[\w-]+)*\/?', ' ', regex = True) 
        df[clean_column] = df[clean_column_1].replace(r'[0-9]', ' ', regex = True) 
        df[clean_column] = df[clean_column].replace(re.compile('[@_!#$%^&*()<>?/\|}{~:]'), ' ', regex = True) 
        df[clean_column] = df[clean_column].replace(r"[^( A-Za-z.;,')]", ' ', regex = True) 
        df[clean_column] = df[clean_column].replace(r"\s+", ' ', regex = True) 
        df[clean_column] = df[clean_column].str.lower()

        #Iterate through pandas dataframe column
        sent_list = []
        for i in range(df.shape[0]):            
            if df[string_column].isnull()[i]:
                sent = np.nan
            else:
                try:
                    text = [df[clean_column_1].iloc[i]]
                    sent = list(nlp_financial(text)[0].values())[0]
                except:
                    try:
                        text = [df[clean_column].iloc[i]]
                        firstpart, secondpart = [text[0][:round(len(text[0])/2)]], [text[0][round(len(text[0])/2):]]
                        sent_1 = list(nlp_financial(firstpart)[0].values())[0]
                        sent_2 = list(nlp_financial(secondpart)[0].values())[0]
                        if sent_1 == sent_2:
                            sent = sent_1
                        elif sent_1 == "neutral":
                            sent = sent_2
                        elif sent_2 == "neutral":
                            sent = sent_1
                        #Deals with large text dividing it in 2 parts 
                        print(f"in row {i} the text is divided into two parts")
                    except RuntimeError: 
                        sent = np.nan
                        #Deals with large text dividing it in 2 parts 
                        print(f"in row {i} the text is too long, no sentiment predicted")
                        pass

            sent_list.append(sent)
        df.drop([clean_column_1, clean_column], axis = 1, inplace = True)
        return sent_list

    def sentiment_df(self, source: str, model: str, tick = None, quer = None):
        if source == "News":
            import logging
            from ..util.message import starting_message
            from dotenv import load_dotenv
            from ..log import logging_config
            from ..api import Polygon
            #Polygon credentials
            load_dotenv()
            logging_config.init_logging()
            __LOG = logging.getLogger(__name__)
            __LOG.info('...... Initialization Completed  ......')
            __LOG.info(starting_message())
            #Initialize Polygon
            polygon = Polygon(environ.get("POLYGON_KEY"))
            #Download News
            status, polygon_data = polygon.get_news('')
            if status:
                polygon_news_df = pd.DataFrame.from_dict(polygon_data['results'])
            polygon_news_df.drop(['publisher', 'keywords'], axis = 1, inplace = True)
            polygon_news_df['tickers'] = [' '.join(map(str, l)) for l in polygon_news_df['tickers']]
            #predict Sentiment
            if model == "FinBERT":
                polygon_news_df['title_FinBERT'] = self.sentiment_finBERT(polygon_news_df, 'title')
                polygon_news_df['description_FinBERT'] = self.sentiment_finBERT(polygon_news_df, 'description')
            elif model == "FinancialBERT":
                polygon_news_df['title_FinancialBERT'] = self.sentiment_financialBERT(polygon_news_df, 'title')
                polygon_news_df['description_FinancialBERT'] = self.sentiment_financialBERT(polygon_news_df, 'description')
            #Upload df.
            polygon_news_df = pd.concat([polygon_news_df, polygon_prev], ignore_index = True).drop_duplicates().reset_index(drop = True)
            polygon_news_df.to_csv("./sentiment_models/polygon_df_01_07_2022.csv", index = False, encoding='utf-8')
            #Polygon plots Data
            df = polygon_news_df
            return df
        elif source == "Tweets":
            from ..api import Twitter 
            twt = Twitter()
            df_twt = twt.get_tweets_df(ticker = tick, query = quer, limit = 100, popular =True)
            if model == "FinBERT":
                df_twt['tweet_FinBERT'] = self.sentiment_finBERT(df_twt, 'full_text')
            elif model == "FinancialBERT":
                df_twt['tweet_FinancialBERT'] = self.sentiment_financialBERT(df_twt, 'full_text')
            month_number = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 'May': '05','Jun':'06', 'Jul':'07', 'Aug':'08', 'Sep':'09', 'Oct':'10', 'Nov':'11', 'Dec':'12'}

            df_twt['day'] = df_twt['created_at'].str[8:10]
            df_twt['month'] = df_twt['created_at'].str[4:7].map(month_number)
            df_twt['year'] = df_twt['created_at'].str[-4:]

            df_twt['date'] = df_twt['year'] + '-' + df_twt['month'] + '-' + df_twt['day'] 
            #Twitter plots Data
            df = df_twt
            return df
        else:
            raise Exception("Valid sources are: News or Tweets") 
    
    def sent_pie(self, source: str, model: str, part = None,  ticker = None, query = None):
        import plotly.express as px
        if source == "News":
            polygon_news_df = self.sentiment_df(source, model)
            colname = part + "_" + model
            df_poly = polygon_news_df[['id','published_utc', colname]]
            df_poly['date'] = df_poly['published_utc'].str[0:10]
            df_poly["total_news_day"] = df_poly.groupby('date')['id'].transform(len)
            df_poly = df_poly.groupby(['date',colname]).size().reset_index()
            df_poly.rename({0:'count'}, axis = 1, inplace = True)
            df_poly['date'] = pd.to_datetime(df_poly['date'], format='%Y-%m-%d')
            df = df_poly
            plot_title = "News: " + part + "s by sentiment"
        elif source == "Tweets":
            df_twt = self.sentiment_df(source, model, tick = ticker, quer = query)
            colname = "tweet_" + model
            df_twt["total_news_day"] = df_twt.groupby('date')['id'].transform(len)
            twt_df = df_twt.groupby(['date', colname]).size().reset_index()
            twt_df.rename({0:'count'}, axis = 1, inplace = True)
            twt_df['date'] = pd.to_datetime(twt_df['date'], format='%Y-%m-%d')
            df = twt_df
            plot_title = "Tweets by Sentiment"

        fig = px.pie(df, values = 'count', names = colname, template='plotly_dark', title = plot_title)
        fig.update_layout(
            legend_traceorder="reversed",
            title={
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(
                #family="Courier New, monospace",
                size=25)
                },
            )   
        return fig


