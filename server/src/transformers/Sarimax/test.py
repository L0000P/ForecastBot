import os
import pandas as pd
from Sarimax import Sarimax

def main():
    # Paths and configurations
    dataset_path = '/server/data/ETTh1.csv'  # Path to your dataset
    results_dir = 'results'
    
    # Define target and feature columns
    target_column = 'HUFL'  # Replace with your target column
    feature_columns = ['HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']  # Replace with your feature columns

    # Create Sarimax instance
    sarimax_model = Sarimax(
        dataset_path=dataset_path,
        target_column=target_column,
        feature_columns=feature_columns,
        results_dir=results_dir
    )

    # Train the SARIMAX model
    orders = [(1, 1, 1)]  # Example orders
    seasonal_orders = [(1, 0, 0, 24)]  # Example seasonal orders
    best_model = sarimax_model.train_model(orders, seasonal_orders)

    # Prepare input series for prediction
    input_series = sarimax_model.df[target_column].iloc[-10:]  # Example: last 10 values for prediction

    # Make predictions
    forecast_values, forecast_index = sarimax_model.predict_model(input_series)

    # Load actual values for comparison
    actual_series = sarimax_model.df[target_column].iloc[-10:]  # Actual values for the same period

    # Calculate MSE
    if forecast_values is not None:
        mse = sarimax_model.calculate_mse(actual_series, forecast_values)
        print(f'Mean Squared Error: {mse}')

    # Plot forecast
    sarimax_model.plot_forecast(actual_series, forecast_values, forecast_index)

if __name__ == '__main__':
    main()