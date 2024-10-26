import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from lib.utils import check_stationarity, fit_arima
import cudf
import pandas as pd

class Arima:
    def __init__(self, dataset_path, timestamp_column='date', use_gpu='yes', plot_dir="/server/transformers/Arima/plots"):
        self.use_gpu = use_gpu
        self.plot_dir = plot_dir
        os.makedirs(plot_dir, exist_ok=True)

        # Carica il dataset
        if use_gpu in ["yes", "y"]:
            self.df = cudf.read_csv(dataset_path, parse_dates=[timestamp_column])
            self.df = self.df.set_index(timestamp_column).to_pandas()  # Convert to pandas
        else:
            self.df = pd.read_csv(dataset_path, parse_dates=[timestamp_column], index_col=timestamp_column)

        # Split dataset into training and testing sets
        train_size = int(len(self.df) * 0.8)
        self.train_set = self.df.iloc[:train_size]
        self.test_set = self.df.iloc[train_size:]

    def check_series_stationarity(self, column):
        """Check if the time series is stationary."""
        p_value = check_stationarity(self.df[column])
        if p_value < 0.05:
            return True  
        else:
            return False 

    def plot_decomposition(self, column):
        """Decompose the time series and plot the seasonal decomposition."""
        try:
            result = seasonal_decompose(self.df[column], model='additive')
            plot_file = f"{self.plot_dir}/{column}_seasonal_decomposition.png"
            result.plot()
            plt.savefig(plot_file)
            plt.close()
            return f"Saved decomposition plot for column '{column}' to {plot_file}"
        except Exception as e:
            return f"Failed to decompose and plot: {str(e)}"

    def train_model(self, column, order):
        """Fit the ARIMA model directly to avoid fit_arima dependency if it returns a scalar."""
        if self.check_series_stationarity(column):
            try:
                model = ARIMA(self.train_set[column], order=order).fit()
                return model
            except Exception as e:
                print(f"Failed to train ARIMA model with order {order}: {str(e)}")
                return None
        else:
            print(f"Skipping training for {column} as it is non-stationary.")
            return None

    def predict_model(self, model, column):
        """Use the trained model to make predictions and compute the MSE."""
        if model is not None:
            try:
                forecast = model.forecast(steps=len(self.test_set))
                mse = ((forecast - self.test_set[column]) ** 2).mean()  # Calculate MSE
                return forecast, mse
            except Exception as e:
                print(f"Failed to predict with the model: {str(e)}")
                return None, None
        else:
            print("No model to predict with.")
            return None, None