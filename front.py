import streamlit as st

st.title("Bai (Brilliant.ai) ")

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

    # Get and display assistant response
    with st.chat_message("assistant"):
        # message = "Sorry, I can only help you with academic querries related to JEE/NEET" 
        message = ''' The correct answer is option (2) as  
Conditions favourable for formation of oxyhaemoglobin in alveoli are high pO2, less H+ concentration 
low pCO2 and low temperature. 
Option (1), (3) and (4) are not correct as they do not favour the formation of oxyhaemoglobin. ''' 

        st.markdown(message)
        st.session_state.messages.append({"role": "assistant", "content": message})
        
        # Add photo in a different container with subheadings
        with st.container():
            st.subheader("NCERT Reference:-")
            st.text("Chapter: Breathing And Exchange Of Gases")
            st.text("Page Number: 189")
            st.image("bai2.png", width=300)

