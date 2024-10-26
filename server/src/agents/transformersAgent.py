from models.arimaModel import train_arima, predict_arima

class TransformersAgent:
    def __init__(self, model_name):
        self.model_name = model_name

    def train(self, params):
        if self.model_name == "ARIMA":
            return train_arima(params)

    def predict(self):
        if self.model_name == "ARIMA":
            return predict_arima()