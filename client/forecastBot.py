from openai import OpenAI
import streamlit as st
import os

default_openai_api_key = os.getenv("OPENAI_API_KEY")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", value=default_openai_api_key, key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[ForecastBot Repo](https://github.com/L0000P/ForecastBot)"

st.title("📈 ForecastBot")
st.caption("🤖 Chatbot designed to allow users to interact with transformer models")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

uploaded_file = st.file_uploader("Upload a file for analysis (optional, default dataset is [ETTh1](https://github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTh1.csv))", 
                                 type=["csv", "txt", "json", "xlsx"])
if uploaded_file:
    st.session_state["file_info"] = f"Received file: {uploaded_file.name}"
    file_preview = uploaded_file.read(1024)
    st.write("File uploaded successfully.")
    st.write(file_preview[:500])
else:
    file_preview = None

if prompt := st.chat_input("Type a message..."):
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
        
    client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    messages = st.session_state.messages
    if file_preview and {"role": "user", "content": file_preview.decode()} not in messages:
        messages.append({"role": "user", "content": file_preview.decode()})

    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)