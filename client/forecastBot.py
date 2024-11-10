import os
import requests
import streamlit as st

# Endpoint URL
CHATBOT_URL = os.getenv("CHATBOT_URL")

with st.sidebar:
    st.header("About")
    st.markdown("[ForecastBot Repo](https://github.com/L0000P/ForecastBot)")

st.title("📈 ForecastBot")
st.caption("🤖 Chatbot designed to allow users to interact with transformer models")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "output" in message.keys():
            st.markdown(message["output"])

        if "explanation" in message.keys():
            with st.status("How was this generated", state="complete"):
                st.info(message["explanation"])

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "output": prompt})

    data = {"query": prompt}

    with st.spinner("Searching for an answer..."):
        response = requests.post(CHATBOT_URL, json=data)

        if response.status_code == 200:
            output_text = response.json()["response"]["output"]
            explanation = output_text

        else:
            output_text = """An error occurred while processing your message.
            Please try again or rephrase your message."""
            explanation = output_text

    if "{" in output_text:
        st.chat_message("assistant").json(output_text)
    else:
        st.chat_message("assistant").markdown(output_text)
    st.status("How was this generated", state="complete").info(explanation)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": output_text,
            "explanation": explanation,
        }
    )