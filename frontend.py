__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from bai_app import chat

st.title("BAI (Brilliant.ai) ")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# To show chat history:-
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input:-
prompt = st.chat_input("Ask me anything about JEE/NEET or about brilliant pala", key="user_input")
if prompt:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show spinner while waiting for response
    with st.spinner("Thinking..."):
        # Get and display assistant response
        with st.chat_message("assistant"):
            message = chat(prompt)["final_response"]
            st.markdown(message)
            st.session_state.messages.append({"role": "assistant", "content": message})

