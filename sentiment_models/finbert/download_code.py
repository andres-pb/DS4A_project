from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

#Download and save the model and tokeneizer for local use.
    #FinBERT
finbert_tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")                                                               #Download
finbert_model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)                                     #Download
finbert_model.save_pretrained("./sentiment_models/finbert/finbert_model")                                                     #Model is saved for local use
finbert_tokenizer.save_pretrained("./sentiment_models/finbert/finbert_tokenizer")                                              #Model tokeneizer is saved for local use