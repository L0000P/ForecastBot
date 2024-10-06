import cudf
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import warnings

# Load the CSV file into a cuDF DataFrame
file_path = '../data/ETTh1.csv'  # Update this with your CSV file path
data = cudf.read_csv(file_path)

# Convert cuDF DataFrame to pandas DataFrame for statsmodels
data_pd = data.to_pandas()

# Convert the 'date' column to datetime format and sort the DataFrame
data_pd['date'] = pd.to_datetime(data_pd['date'])
data_pd = data_pd.sort_values('date')

# Set the 'date' column as the index
data_pd.set_index('date', inplace=True)

# Explicitly set the frequency to hourly ('H')
data_pd.index.freq = 'H'

# Define the target and feature columns
target_column = 'HUFL'  # Replace with the target column you want to predict
feature_columns = ['HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']  # All other columns

# Example for iterating through different SARIMAX parameters
orders = [(1, 1, 1), (2, 1, 1), (1, 1, 2)]
seasonal_orders = [(1, 1, 1, 12), (1, 0, 1, 12), (1, 1, 0, 12)]

best_aic = float('inf')
best_model = None

# Calculate total iterations for tqdm
total_iterations = len(orders) * len(seasonal_orders)

# Fitting SARIMAX models
with tqdm(total=total_iterations, desc="Fitting SARIMAX models", file=sys.stdout) as pbar:
    for order in orders:
        for seasonal_order in seasonal_orders:
            try:
                # Create and fit the SARIMAX model
                model = SARIMAX(data_pd[target_column],
                                order=order,
                                seasonal_order=seasonal_order,
                                exog=data_pd[feature_columns])
                
                # Catch all warnings
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    results = model.fit(disp=False, maxiter=1000, method='powell')
                    
                    # Check for any warnings
                    for warning in w:
                        print("Warning:", warning.message)

                # Update best model based on AIC
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_model = results
                
                # Update progress bar
                pbar.update(1)
                sys.stdout.flush()
            except Exception as e:
                print(f"Failed to fit model with order {order} and seasonal order {seasonal_order}: {e}", file=sys.stderr)
                pbar.update(1)
                sys.stdout.flush()

# If you found the best model, make predictions
if best_model:
    forecast_steps = 10  # Number of steps to forecast
    forecast = best_model.get_forecast(steps=forecast_steps, exog=data_pd[feature_columns].iloc[-forecast_steps:])
    forecast_index = pd.date_range(start=data_pd.index[-1], periods=forecast_steps + 1, freq='H')[1:]

    # Extract forecast values
    forecast_values = forecast.predicted_mean
    
    # Calculate MSE between the actual values and the forecast
    actual_values = data_pd[target_column].iloc[-forecast_steps:]  # Actual values for the forecasted period
    mse = mean_squared_error(actual_values, forecast_values)

    # Print MSE
    print(f"Mean Squared Error (MSE): {mse}")

    # Visualize results
    plt.figure(figsize=(10, 5))
    plt.plot(data_pd.index, data_pd[target_column], label='Actual')
    plt.plot(forecast_index, forecast_values, label='Forecast', color='red')
    plt.title('SARIMAX Forecast')
    plt.xlabel('Date')
    plt.ylabel(target_column)
    plt.legend()
    plt.show()
else:
    print("No model was fitted successfully.", file=sys.stderr)
