# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import logging
import streamlit as st

# FastAPI app instance
app = FastAPI()

# FastAPI backend API route
@app.on_event("startup")
async def startup_event():
    logging.info("FastAPI server has started.")

class InputData(BaseModel):
    name: str
    age: int

@app.post("/submit")
def submit(data: InputData):
    return {"message": f"Hello {data.name}, you are {data.age} years old!"}
