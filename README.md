# LSTM-Stocks
Stock prediction model using LSTM models. 

# Full roadmap of changes/implementation
~~1. Data Loading and Preparation
Description: Pull historical stock price data from Yahoo Finance for multiple stocks. Format data into sequences of 60 days to prepare it for training in an LSTM model.

Key Files/Modules: data_loader.py for data loading, main.py for orchestrating data preparation.~~

2. Feature Engineering

Description: Integrate additional data sources to enhance model predictions, including:
Sentiment Analysis: Calculate sentiment scores from news articles or social media for each stock.
SEC Insider Trades: Monitor insider buying/selling activity as an indicator of internal confidence.
Macroeconomic Indicators: Include indicators like interest rates or inflation to understand broader market influence.

Key Files/Modules: feature_engineering.py for feature extraction functions.

3. Data Merging and Scaling

Description: Combine stock data with additional features, then normalize values for model compatibility.

Key Files/Modules: utils.py for merge_data() and scale_data() functions.

4. Model Building

Description: Constructs an LSTM-based neural network that takes 60-day sequences as input and outputs a single predicted price for each sequence.

Key Files/Modules: model.py with the build_model() function.

5. Model Training

Description: Train the LSTM model on 80% of the stock’s historical data and reserve 20% for testing. The training focuses on minimizing Mean Squared Error (MSE).

Key Files/Modules: model.py using the train_model() function.

6. Generating Predictions

Description: Generate predictions for the test period. Predictions are converted back to the original scale (dollars) for comparison against actual prices.

Key Files/Modules: model.py, specifically the make_predictions() function.

7. Evaluation Metrics

Description: Calculate MSE, RMSE, and correlation between predictions and actual values to quantitatively assess prediction accuracy.

Key Files/Modules: Evaluation code within main.py.

8. LLM-Based Evaluation Integration

Description: Use an LLM (e.g., LLaMa) to qualitatively evaluate model predictions, considering contextual data. This approach enables nuanced, context-based judgment that traditional metrics alone may not capture.
Prediction Assessment: LLMs evaluate the alignment of model predictions with recent market sentiment, insider trades, and macroeconomic factors.
Scoring and Explanation: LLM provides a rating (e.g., 1-5 scale) on alignment quality and offers an explanation for flagged cases.
Composite Score: Combine LLM ratings with traditional metrics to highlight cases where quantitative and qualitative evaluations diverge.

Key Files/Modules: llm_evaluator.py (new) to handle LLM-based evaluations, including llm_evaluate() and batch_evaluate() functions.

9. Feature Expansion and Model Tuning

Description: Continue adding features to improve the model’s predictive power. Adjust hyperparameters and tune the LSTM model to capture patterns better, especially for high-volatility stocks. Analyze feature importance to refine further.

Key Files/Modules: Ongoing development in feature_engineering.py and model tuning in main.py.

