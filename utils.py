""" 
Helper functions for merging and scaling our data. This is integral in making multiple datasources in multiple
formats become one unified scale to compare. 

merge_data(): 
Merges the stock data with additional features (if available). Each stocks DataFrame is combined with 
any additional data (such as SEC or sentiment data) by aligning dates.

scale_data(): 
Scales the data for each stock using MinMax scaling, which normalizes values between 0 and 1. It returns 
the scaled data and the scaler for each stock.


"""



from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def scale_data(data_dict):
    """
    Scales each stock's data (in data_dict) using MinMax scaling.
    Returns a dictionary of scaled data for each stock and a dictionary of scalers.
    """
    scaled_data_dict = {}
    scalers = {}

    for symbol, data in data_dict.items():
        # Initialize and fit the scaler for each stock's data individually
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)  # Scale the DataFrame
        
        # Store the scaled data and scaler
        scaled_data_dict[symbol] = scaled_data
        scalers[symbol] = scaler
    
    return scaled_data_dict, scalers


def merge_data(stock_data, sec_data=None, sentiment_data=None, macro_data=None):
    """
    Merges stock data with additional data sources (e.g., SEC, sentiment, macroeconomic data).
    Returns a dictionary of merged data for each stock symbol.
    """
    # Dictionary to store merged data for each stock
    merged_data = {}
    
    for symbol, data in stock_data.items():
        # Start with the main stock data
        combined = data
        
        # Merge additional data if available
        if sec_data is not None:
            # Assuming sec_data is also a dictionary with the same stock symbols as keys
            combined = combined.join(sec_data.get(symbol, pd.DataFrame()), how='outer')
        if sentiment_data is not None:
            combined = combined.join(sentiment_data.get(symbol, pd.DataFrame()), how='outer')
        if macro_data is not None:
            combined = combined.join(macro_data.get(symbol, pd.DataFrame()), how='outer')
        
        # Store the merged DataFrame for each stock symbol
        merged_data[symbol] = combined
    
    return merged_data
