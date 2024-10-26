from pydantic import BaseModel

class SarimaxParms(BaseModel):
    p: int
    d: int
    q: int

def train_sarimax(params: SarimaxParms):
    # Codice per allenare il modello ARIMA
    pass

def predict_sarimax():
    # Codice per fare previsioni con ARIMA
    pass