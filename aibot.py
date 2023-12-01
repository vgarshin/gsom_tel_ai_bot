import os
import json
import requests
import streamlit as st

HEADER = {'Content-type': 'application/json'}
PORT = 40000
URL_SERVER = 'http://158.160.123.124:{}/ask'.format(
    PORT
)

st.title("GSOM TEL YaGPT bot for Teaching essentials course")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if query := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(query)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    data = {'query': query} 
    r = requests.post(
        URL_SERVER,
        data=json.dumps(data),
        headers=HEADER,
        verify=True
    )
    response = r.json()['answer']
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})