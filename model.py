"""
Funtions for building, training, and predicting with our LSTM model in a Neural Network. 

build_model(): 
Builds an LSTM model with two LSTM layers and two Dense layers. The model takes sequences of 60 days (sequence length)
 as input and outputs a single predicted value for each sequence.

train_model(): 
Trains the model on x_train and y_train data, using mean squared error (MSE) as the loss metric.

make_predictions(): 
Generates predictions on test data and returns them in the original scale (dollars) using the saved scaler for each stock.

"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def build_model(sequence_length, input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        LSTM(units=50, return_sequences=False),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, batch_size=1, epochs=1)

def make_predictions(model, x_test, scaler):
    predictions = model.predict(x_test)
    return scaler.inverse_transform(predictions)
