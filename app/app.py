# Import FastAPI for backend
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import logging
import streamlit as st

# FastAPI app instance
api_app = FastAPI()

# FastAPI backend API route
@api_app.on_event("startup")
async def startup_event():
    logging.info("FastAPI server has started.")

# A data model for FastAPI (you can modify this based on your needs)
class InputData(BaseModel):
    name: str
    age: int

# Streamlit frontend interface
def run_streamlit():
    st.title("FastAPI + Streamlit App")

    # Input form
    name = st.text_input("Enter your name")
    age = st.number_input("Enter your age", min_value=0, max_value=120, step=1)

    if st.button("Submit"):
        data = {"name": name, "age": age}
        # Call the FastAPI backend
        response = requests.post("http://web_app:8000/submit", json=data)  # Use service name instead of 'localhost' or 'transformers-web_app'
        if response.status_code == 200:
            st.success(f"Response: {response.json()}")
        else:
            st.error(f"Error: {response.status_code}")

# FastAPI backend API route
@api_app.post("/submit")
def submit(data: InputData):
    return {"message": f"Hello {data.name}, you are {data.age} years old!"}

# Run both FastAPI and Streamlit together
if __name__ == "__main__":
    run_streamlit()
