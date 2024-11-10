import os
import pandas as pd
from Sarimax import Sarimax

def main():
    target_column = 'HUFL'
    # Create Sarimax instance
    sarimax_model = Sarimax()

    # Train the SARIMAX model
    orders = [(1, 1, 1)]  # Example orders
    seasonal_orders = [(1, 0, 0, 24)]  # Example seasonal orders
    best_model = sarimax_model.train_model(orders, seasonal_orders)

    # Prepare input series for prediction
    input_series = sarimax_model.df[target_column].iloc[-10:]  # Example: last 10 values for prediction

    # Make predictions
    forecast_values, forecast_index = sarimax_model.predict_model(input_series)

    # Load actual values for comparison and slice to match forecast length
    actual_series = sarimax_model.df[target_column].iloc[-len(forecast_values):]

    # Calculate MSE
    if forecast_values is not None:
        mse = sarimax_model.calculate_mse(actual_series, forecast_values)
        print(f'Mean Squared Error: {mse}')

    # Plot forecast
    sarimax_model.plot_forecast(actual_series, forecast_values, forecast_index)

if __name__ == '__main__':
    main()
