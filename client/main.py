import streamlit as st
import openai

# Configurazione API di OpenAI
openai.api_key = "YOUR_API_KEY"

# Funzione per generare risposte
def get_forecastbot_response(user_input):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Cambia in gpt-4 se preferisci o hai accesso
            messages=[
                {"role": "system", "content": "Sei Forecastbot, un chatbot per previsioni e analisi."},
                {"role": "user", "content": user_input},
            ]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Errore: {e}"

# Configurazione di Streamlit
st.set_page_config(page_title="Forecastbot", page_icon=":robot_face:")
st.title("Forecastbot 🤖")
st.write("Benvenuto! Sono Forecastbot. Chiedimi qualsiasi cosa sulle previsioni o su altri argomenti!")

# Funzione di layout della chat
def chat_layout():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_input("Scrivi qui la tua domanda", "")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        bot_response = get_forecastbot_response(user_input)
        st.session_state.messages.append({"role": "bot", "content": bot_response})
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.write(f"**Tu:** {message['content']}")
        else:
            st.write(f"**Forecastbot:** {message['content']}")

chat_layout()
