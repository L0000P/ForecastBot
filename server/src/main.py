from fastapi import FastAPI
from agents import TransformersAgent

app = FastAPI(name="ForecastBot",
              descritpiton="Chatbot to Forecast Time Series Dataset using LLM and different transformers models")

@app.post("/train/{model_name}")
def train_model(model_name: str, params: dict):
    agent = TransformersAgent(model_name)
    agent.train(params)
    return {"status": "training completed"}

@app.get("/predict/{model_name}")
def predict_model(model_name: str):
    agent = TransformersAgent(model_name)
    prediction = agent.predict()
    return {"prediction": prediction}