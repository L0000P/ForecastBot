import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
from statsmodels.tsa.stattools import adfuller

# Suppress warnings
warnings.filterwarnings("ignore")

# Load your dataset
df = pd.read_csv('../data/ETTh1.csv', parse_dates=['date'], index_col='date')

# Function to check stationarity
def check_stationarity(series):
    result = adfuller(series)
    return result[1]  # Return p-value

# Function to fit ARIMA model and return MSE
def fit_arima(train, test, order):
    model = sm.tsa.ARIMA(train, order=order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))
    mse = np.mean((predictions - test)**2)
    return mse

# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train_set = df.iloc[:train_size]
test_set = df.iloc[train_size:]

# Define ARIMA orders to test
orders = [(0, 1, 0), (1, 1, 2)]  # Example orders; you can modify this list

# Analyze all columns
results = {}
for column in df.columns:
    print(f'\nAnalyzing column: {column}')
    
    # Check stationarity
    p_value = check_stationarity(df[column])
    if p_value < 0.05:
        print(f'The series "{column}" is stationary (p-value: {p_value:.4f})')
    else:
        print(f'The series "{column}" is non-stationary (p-value: {p_value:.4f})')

    # Initialize dictionary to store MSE results
    mse_results = {}
    
    # Testing ARIMA models for the current column
    for order in orders:
        try:
            mse = fit_arima(train_set[column], test_set[column], order)
            mse_results[order] = mse
            print(f'Order {order} - MSE: {mse}')
        except Exception as e:
            print(f'Failed to fit ARIMA{order}: {str(e)}')

    # Print best order based on MSE for the current column
    if mse_results:
        best_order = min(mse_results, key=mse_results.get)
        print(f'Best Order for {column}: {best_order} with MSE: {mse_results[best_order]}')
