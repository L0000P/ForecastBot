import pandas as pd
import cudf
import warnings
from lib.utils import check_stationarity, fit_arima

warnings.filterwarnings("ignore") # Suppress warnings

# Ask user for input
use_gpu = input("Do you want to use GPU<|yes|,no>?") or "yes" 
timestamp_column = input("Insert timestamp column<|date|>:") or "date"
dataset_path = input("Insert dataset path <|../data/ETTh1.csv|>:") or "../data/ETTh1.csv"

# Load dataset
if use_gpu == "yes" or use_gpu=="y" or use_gpu==" " or use_gpu=="Y":
    # Use cuDF for GPU acceleration
    df = cudf.read_csv(dataset_path, parse_dates=[timestamp_column])
    df = df.set_index(timestamp_column) 
    # Convert cuDF to pandas for `statsmodels` compatibility
    df = df.to_pandas()
else:
    # Use pandas for CPU execution
    df = pd.read_csv(dataset_path, parse_dates=[timestamp_column], index_col=timestamp_column)


# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train_set = df.iloc[:train_size]
test_set = df.iloc[train_size:]

# Define ARIMA orders to test
orders = [(0, 1, 0), (1, 1, 2)]  # Example orders; modify as needed

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
            mse = fit_arima(train_set[column], test_set[column], order) # Fit ARIMA model
            mse_results[order] = mse # Store MSE
            print(f'Order {order} - MSE: {mse}') 
        except Exception as e:
            print(f'Failed to fit ARIMA{order}: {str(e)}')

    # Print best order based on MSE for the current column
    if mse_results:
        best_order = min(mse_results, key=mse_results.get) 
        print(f'Best Order for {column}: {best_order} with MSE: {mse_results[best_order]}')