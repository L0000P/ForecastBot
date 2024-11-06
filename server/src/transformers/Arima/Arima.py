import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import cudf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

class Arima:
    def __init__(self, dataset_path="/server/data/ETTh1.csv", timestamp_column='date', use_gpu='yes', result_dir="/server/src/transformers/Arima/results"):
        os.makedirs(result_dir, exist_ok=True)
        self.use_gpu = use_gpu
        self.result_dir = result_dir

        if use_gpu in ["yes", "y"]:
            self.df = cudf.read_csv(dataset_path, parse_dates=[timestamp_column])
            self.df = self.df.set_index(timestamp_column).to_pandas()
        else:
            self.df = pd.read_csv(dataset_path, parse_dates=[timestamp_column], index_col=timestamp_column)

        train_size = int(len(self.df) * 0.8)
        self.train_set = self.df.iloc[:train_size]
        self.test_set = self.df.iloc[train_size:]

        self.scaler = MinMaxScaler()
        self.scaler.fit(self.train_set)

    def check_series_stationarity(self, series):
        result = adfuller(series)
        return result[1] < 0.05

    def plot_decomposition(self, column):
        try:
            result = seasonal_decompose(self.df[column], model='additive')
            plot_file = f"{self.result_dir}/{column}_seasonal_decomposition.png"
            result.plot()
            plt.savefig(plot_file)
            plt.close()
            return f"Saved decomposition plot for column '{column}' to {plot_file}"
        except Exception as e:
            return f"Failed to decompose and plot: {str(e)}"

    def train_model(self, series, order):
        if self.check_series_stationarity(series):
            try:
                model = ARIMA(series, order=order).fit()
                model_file = os.path.join(self.result_dir, 'model.pkl')
                joblib.dump(model, model_file)
                print(f"Model saved to {model_file}")
                return model
            except Exception as e:
                print(f"Failed to train ARIMA model with order {order}: {str(e)}")
                return None
        else:
            print(f"Skipping training for {series.name} as it is non-stationary.")
            return None

    def load_model(self):
        try:
            model_file = os.path.join(self.result_dir, 'model.pkl')
            model = joblib.load(model_file)
            print(f"Model loaded from {model_file}")
            return model
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            return None

    def predict_model(self, input_series):
        model = self.load_model()
        if model is not None:
            try:
                input_series = input_series.squeeze()
                forecast = model.forecast(steps=len(input_series))
                forecast_df = pd.DataFrame({"date": input_series.index, "forecast": forecast}).set_index("date")
                forecast_file = os.path.join(self.result_dir, f"forecasted.csv")
                forecast_df.to_csv(forecast_file)
                print(f"Forecast saved to {forecast_file}")
                return forecast_df["forecast"]
            except Exception as e:
                print(f"Failed to predict with the model: {str(e)}")
                return None
        else:
            print("No model to predict with.")
            return None

    def calculate_mse(self, actual_series, forecast_series):
        mse = ((actual_series - forecast_series) ** 2).mean()
        return mse