import sys
sys.path.append('/server/src')

from fastapi import FastAPI, HTTPException
from Models import QueryRequest, QueryResponse
from Agents import Agent
app = FastAPI(name="ForecastBot",
              description="Chatbot to Forecast Time Series Dataset using LLM and different transformers models")

agent = Agent()

@app.post("/invoke", response_model=QueryResponse)
async def invoke_query(request: QueryRequest):
    try:
        response = agent.invoke(request.query)
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "API is running"}
