import os
from Arima import Arima

# Path to your dataset
dataset_path = "../data/ETTh1.csv"  # Make sure this path is correct
timestamp_column = "date"
column_to_test = "HUFL"  # Replace with the column you want to test
arima_order = (1, 1, 1)  # ARIMA order, adjust based on your requirements

# Initialize the Arima class
arima_instance = Arima(dataset_path=dataset_path, timestamp_column=timestamp_column, use_gpu="yes")

# Test check_series_stationarity
is_stationary = arima_instance.check_series_stationarity(column=column_to_test)
print(f"Stationarity test for {column_to_test}: {'Stationary' if is_stationary else 'Non-stationary'}")

# Test plot_decomposition
decomposition_result = arima_instance.plot_decomposition(column=column_to_test)
print(decomposition_result)

# Test train_model
model = arima_instance.train_model(column=column_to_test, order=arima_order)
if model:
    print(f"ARIMA model trained for {column_to_test} with order {arima_order}")

# Test predict_model
if model:
    forecast, mse = arima_instance.predict_model(model=model, column=column_to_test)
    if forecast is not None:
        print(f"Forecast for {column_to_test}: {forecast[:5]}")  # Display first 5 predictions
        print(f"MSE for {column_to_test}: {mse}")
    else:
        print(f"Prediction for {column_to_test} failed.")
else:
    print(f"Model for {column_to_test} could not be trained.")
