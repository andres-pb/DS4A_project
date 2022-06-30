from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

#Download and save the model and tokeneizer for local use.
    #FinancialBERT
financialbert_tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")                             #Download
financialbert_model = BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis", num_labels=3)   #Download
financialbert_model.save_pretrained("./sentiment_models/financial_bert/financialbert_model")                                        #Model is saved for local use
financialbert_tokenizer.save_pretrained("./sentiment_models/financial_bert/financialbert_tokenizer")                                #Model tokenizer is saved for local use


