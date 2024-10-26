import sys
from pydantic import BaseModel

sys.path.append('/server/transformers/Arima')

from Arima import Arima

class ArimaParms(BaseModel):
    dataset_path: str = "/server/transformers/data/ETTh1.csv"
    timestamp_column: str = "date"
    use_gpu: str = "yes"
    plot_dir: str = "/server/transformers/Arima/plots"
    column: str = "HUFL"
    order: tuple = (1, 1, 0)  

def train_arima(params: ArimaParms):
    params = ArimaParms(**params)
    arima_model = Arima(
        dataset_path=params.dataset_path,
        timestamp_column=params.timestamp_column,
        use_gpu=params.use_gpu,
        plot_dir=params.plot_dir
    )
    
    # Plot decomposition
    plot_message = arima_model.plot_decomposition(params.column)
    
    # Train the model
    model = arima_model.train_model(params.column, params.order)
    return {"model": model, "plot_message": plot_message}

def predict_arima(params: ArimaParms):
    arima_model = Arima(
        dataset_path=params.dataset_path,
        timestamp_column=params.timestamp_column,
        use_gpu=params.use_gpu,
        plot_dir=params.plot_dir
    )
    
    model = arima_model.train_model(params.column, params.order)
    forecast, mse = arima_model.predict_model(model, params.column)
    return {"forecast": forecast, "mse": mse}