import os
import cudf
import pandas as pd
import matplotlib.pyplot as plt
import sys
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import joblib

class Sarimax:
    def __init__(self, 
                 dataset_path="/server/data/ETTh1.csv", 
                 target_column="HUFL", 
                 feature_columns=['HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
                 timestamp_column='date', 
                 use_gpu='yes', 
                 results_dir="/server/src/transformer/Sarimax/results"):
        self.use_gpu = use_gpu.lower() in ["yes", "y"]
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        if self.use_gpu:
            self.df = cudf.read_csv(dataset_path)
            self.df[timestamp_column] = cudf.to_datetime(self.df[timestamp_column])
            self.df = self.df.to_pandas()  # Convert entire DataFrame to pandas
        else:
            self.df = pd.read_csv(dataset_path, parse_dates=[timestamp_column])

        self.df = self.df.sort_values(timestamp_column).set_index(timestamp_column)
        self.df.index.freq = 'H'  # Set frequency if required
        self.target_column = target_column
        self.feature_columns = feature_columns

    def train_model(self, orders, seasonal_orders):
        best_aic = float('inf')
        best_model = None
        total_iterations = len(orders) * len(seasonal_orders)

        with tqdm(total=total_iterations, desc="Fitting SARIMAX models", file=sys.stdout) as pbar:
            for order in orders:
                for seasonal_order in seasonal_orders:
                    try:
                        target_data = self.df[self.target_column]
                        exog_data = self.df[self.feature_columns] if self.feature_columns else None

                        model = SARIMAX(
                            target_data,
                            order=order,
                            seasonal_order=seasonal_order,
                            exog=exog_data
                        )
                        
                        with warnings.catch_warnings(record=True) as w:
                            warnings.simplefilter("always")
                            results = model.fit(disp=False, maxiter=1000, method='powell')
                            
                            for warning in w:
                                print("Warning:", warning.message)

                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_model = results
                        
                        pbar.update(1)
                        sys.stdout.flush()
                    except Exception as e:
                        print(f"Failed to fit model with order {order} and seasonal order {seasonal_order}: {e}", file=sys.stderr)
                        pbar.update(1)
                        sys.stdout.flush()

        if best_model:
            model_file = os.path.join(self.results_dir, 'trained_sarimax_model.pkl')
            joblib.dump(best_model, model_file)
            print(f"Best model saved to {model_file}")

        return best_model

    def load_model(self):
        try:
            model_file = os.path.join(self.results_dir, 'trained_sarimax_model.pkl')
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
                if isinstance(input_series, cudf.Series):
                    input_series = input_series.to_pandas()
                
                exog_data = self.df[self.feature_columns].iloc[-len(input_series):] if self.feature_columns else None
                
                forecast = model.get_forecast(steps=len(input_series), exog=exog_data)
                forecast_index = pd.date_range(start=input_series.index[-1], periods=len(input_series) + 1, freq='H')[1:]
                forecast_values = forecast.predicted_mean

                forecast_df = pd.DataFrame({
                    'Date': forecast_index,
                    'Forecasted Values': forecast_values
                })
                forecast_csv = os.path.join(self.results_dir, 'forecasted.csv')
                forecast_df.to_csv(forecast_csv, index=False)
                print(f"Forecast data saved to {forecast_csv}")

                return forecast_values, forecast_index
            except Exception as e:
                print(f"Failed to predict with the model: {e}")
                return None, None
        else:
            print("No model to predict with.")
            return None, None

    def calculate_mse(self, actual_series, forecast_series):
        if isinstance(actual_series, cudf.Series):
            actual_series = actual_series.to_pandas()
        mse = mean_squared_error(actual_series, forecast_series)
        return mse

    def plot_forecast(self, actual_series, forecast_values, forecast_index):
        plt.figure(figsize=(10, 5))
        plt.plot(self.df.index[-len(actual_series):], actual_series, label='Actual', color='blue')  # Use the last actual values
        plt.plot(forecast_index, forecast_values, label='Forecast', color='red')
        plt.title('SARIMAX Forecast')
        plt.xlabel('Date')
        plt.ylabel(self.target_column)
        plt.legend()
        plot_file = os.path.join(self.results_dir, 'sarimax_forecast.png')
        plt.savefig(plot_file)
        plt.close()
        print(f"Forecast plot saved to {plot_file}")
