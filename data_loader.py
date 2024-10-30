""" 
Handler script that fetches our stock data. 

Using our Yahoo Finance API (which is a library that python can directly query), yfinance, we can
retrieve historical data of stock prices based on the symbols we defined in main.py (which calls this function). 

Stores the associated price data in a data frame

"""

import yfinance as yf

def load_stock_data(symbols):
    # Fetch historical data for each stock symbol
    data = {}
    for symbol in symbols:
        stock_data = yf.download(symbol, start="2022-01-01", end="2024-01-01")[['Close']]
        data[symbol] = stock_data
    return data
