"""
This is our central script orchestrating the whole pipeline. It does the following in sequential order: 

1. Imports required functions, which handles data loading, model building, training, and predicting
2. Defines our stocks to analyze 
3. Loads historical stock data based on its symbol 
4. Generates placeholder features (FOR NOW) which will eventually pull from additional sources
to build out our predictions 
5. Merges and scales our data, in a normalized fashion so that its suitable to our model
6. Builds, trains and predicts: 
    6.1. Builds an LSTM model 
    6.2. Trains model using 80% of the stocks historical data, keeps 20% for predicting 
    6.3. Generates predictions on test data 
    6.4. Prints our predictions, via our output for each stock to show predicted price movements. 

    
The current output you'll see is the "print" statement in our loop, showing the model's predicted stock prices 
for the test period based on the training we asked it to do. 

If you want to get the libraries downloaded for this to run, download the requirements: 
    - In your terminal at this folder, run pip install -r requirements.txt to get all dependencies and libraries needed
"""

import numpy as np
from data_loader import load_stock_data
from feature_engineering import get_sec_insider_trades, get_news_sentiment, get_macro_data
from model import build_model, train_model, make_predictions
from utils import scale_data, merge_data

# Define a list of stock symbols to fetch data for multiple stocks
# This list includes some popular large-cap stocks as an example
stock_symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]

# Step 1: Load stock data
stock_data = load_stock_data(stock_symbols)

# Step 2: Load and create additional features (for now, placeholder functions)
sec_data = get_sec_insider_trades("AAPL")       # Placeholder
sentiment_data = get_news_sentiment("AAPL")     # Placeholder
macro_data = get_macro_data()                   # Placeholder

# Step 3: Merge and scale data
# `merged_data` will contain a DataFrame for each stock with combined features
merged_data = merge_data(stock_data, sec_data, sentiment_data, macro_data)
scaled_data, scalers = scale_data(merged_data)

for symbol, data in scaled_data.items():
    # Define the model for each stock
    model = build_model(sequence_length=60, input_shape=(60, data.shape[1]))

    # Split data into training and testing
    train_data_len = int(len(data) * 0.8)
    train_data = data[:train_data_len]  # Use 80% of data for training

    # Create x_train and y_train as sequences
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i])  # Last 60 days of data as input
        y_train.append(train_data[i, 0])      # Next day price as output

    # Convert x_train and y_train to numpy arrays and reshape for model compatibility
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], data.shape[1]))

    # Train the model with each stock’s training data
    train_model(model, x_train, y_train)

    # Make predictions with the model on test data (if needed for predictions on new data)
    predictions = make_predictions(model, x_train, scalers[symbol])

    print(f"Predictions for {symbol}: {predictions}")

