import pandas
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import numpy as np

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

        #Iterate through pandas dataframe column
        sent_list = []
        for i in range(df.shape[0]):
            if df[string_column].isnull()[i]:
                sent = np.nan
            else:
                try:
                    sent = list(nlp_fin(df[string_column].tolist()[i])[0].values())[0]
                except:
                    #Deals with large text dividing it in 2 parts 
                    print(f"in row {i} the text its divided in two parts")
                    firstpart, secondpart = df[string_column].tolist()[i][:round(len(df[string_column].tolist()[i])/2)], df[string_column].tolist()[i][round(len(df[string_column].tolist()[i])/2):]
                    sent_1 = [list(descr.items())[0][1] for descr in  nlp_fin(firstpart)]
                    sent_2 = [list(descr.items())[0][1] for descr in  nlp_fin(secondpart)]
                    if sent_1 == sent_2:
                        sent = sent_1
                    elif sent_1 == "neutral":
                        sent = sent_2
                    elif sent_2 == "neutral":
                        sent = sent_1
            sent_list.append(sent)
        return sent_list

    
    def sentiment_financialBERT(self, df: pandas.core.frame.DataFrame, string_column: str):
        """
        This function executes sentiment prediction using FinancialBERT over a text column within a pandas data frame.
            Inputs:
            df: Pandas dataframe,
            string_column: name of the columng of interest ex: “column_1”
            output:
            A list with the predicted sentiment of each element of the pandas' column.
            It can be assigned to a new column as : df[new_column] =  sentiment_finBERT(df, “column_1”)
        """
        #Iterate through pandas dataframe column
        sent_list = []
        for i in range(df.shape[0]):
            if df[string_column].isnull()[i]:
                sent = np.nan
            else:
                try:
                    sent = list(nlp_financial(df[string_column].tolist()[i])[0].values())[0]
                except: 
                    #Deals with large text dividing it in 2 parts 
                    print(f"in row {i} the text its divided in two parts")
                    firstpart, secondpart = df[string_column].tolist()[i][:round(len(df[string_column].tolist()[i])/2)], df[string_column].tolist()[i][round(len(df[string_column].tolist()[i])/2):]
                    sent_1 = [list(descr.items())[0][1] for descr in  nlp_financial(firstpart)]
                    sent_2 = [list(descr.items())[0][1] for descr in  nlp_financial(secondpart)]
                    if sent_1 == sent_2:
                        sent = sent_1
                    elif sent_1 == "neutral":
                        sent = sent_2
                    elif sent_2 == "neutral":
                        sent = sent_1
            sent_list.append(sent)
        return sent_list

#Example use
""" 
sp = Sentiment_predict()
df['title_sent'] = sp.sentiment_finBERT(df, 'title')
df['descr_sent'] = sp.sentiment_finBERT(df, 'description')

df['title_sent'] = sp.sentiment_financialBERT(df, 'title')
df['descr_sent'] = sp.sentiment_financialBERT(df, 'description')   
"""
