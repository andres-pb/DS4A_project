import pandas
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import numpy as np
import re

#Load FinBERT model
finbert_model = BertForSequenceClassification.from_pretrained("./sentiment_models/finbert/finbert_model")
finbert_tokenizer = BertTokenizer.from_pretrained("./sentiment_models/finbert/finbert_tokenizer")
nlp_fin = pipeline("sentiment-analysis", model=finbert_model, tokenizer=finbert_tokenizer)

#Load FinancialBERT model
financialbert_model = BertForSequenceClassification.from_pretrained("./sentiment_models/financial_bert/financialbert_model")
financialbert_tokenizer = BertTokenizer.from_pretrained("./sentiment_models/financial_bert/financialbert_tokenizer")
nlp_financial = pipeline("sentiment-analysis", model=financialbert_model, tokenizer=financialbert_tokenizer)

class Sentiment_predict():

    def __init__(self) -> None:       
        pass
        
    def sentiment_finBERT(self, df: pandas.core.frame.DataFrame, string_column: str):
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

    
    def sentiment_financialBERT(self, df: pandas.core.frame.DataFrame, string_column: str):
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

#Example use
""" 
sp = Sentiment_predict()
df['title_sent'] = sp.sentiment_finBERT(df, 'title')
df['descr_sent'] = sp.sentiment_finBERT(df, 'description')

df['title_sent'] = sp.sentiment_financialBERT(df, 'title')
df['descr_sent'] = sp.sentiment_financialBERT(df, 'description')   
"""
