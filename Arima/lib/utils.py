from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import numpy as np

# Function to check stationarity (this runs on CPU since `adfuller` is CPU-based)
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