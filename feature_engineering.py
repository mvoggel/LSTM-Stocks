"""
Handles our additional features beyond pulling historical stock prices. Right now they basically are 
placeholders, but in addition to just having a history and predicting, we'll want to include additional 
data like SEC filings, insider trading forms, macroeconomic data, or even news sentiments. 

This will be a hugely important group of functions to help make our model the most predictive.

To work on first - 


"""


import requests
from textblob import TextBlob

def get_sec_insider_trades(ticker):
    # Placeholder for SEC insider trades data
    return None

def get_news_sentiment(ticker):
    # Placeholder for news sentiment data
    return None

def get_macro_data():
    # Placeholder for macroeconomic data
    return None
