import pandas as pd
import numpy as np
from Arima import Arima  # Adjust the import path as necessary

def main():
    # Dataset path
    dataset_path = '/server/data/ETTh1.csv'
    
    # ARIMA model
    arima_model = Arima(dataset_path)

    # Specify the column to train on and the ARIMA order
    target_column = 'HUFL' # Column to train on
    order = (1, 1, 1) # ARIMA order

    # Train the ARIMA model
    trained_model = arima_model.train_model(arima_model.train_set[target_column], order)

    # If the model is trained successfully
    if trained_model:
        # Forecast on the test set (using the test set length for forecasting)
        forecasted_values = arima_model.predict_model(arima_model.test_set[target_column])

        # Calculate MSE if forecasted values are available
        if forecasted_values is not None:
            actual_values = arima_model.test_set[target_column]
            mse = arima_model.calculate_mse(actual_values, forecasted_values)
            print(f'Mean Squared Error: {mse}')

        # Optionally, plot the seasonal decomposition
        decomposition_result = arima_model.plot_decomposition(target_column)
        print(decomposition_result)

if __name__ == '__main__':
    main()