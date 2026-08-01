from llama_index.llms.groq import Groq
import streamlit as st

def chat_qa(prompt):
    # Initialize the Groq model with the temperature parameter
    llm = Groq(
        model = "llama-3.3-70b-versatile",
        api_key = "",
        temperature = 1
    )
    #prompt = "How to make milkrice"
    
    # Generate a response using the complete method
    response = llm.complete(prompt)
    #print(response)
    return response


st.title(f"**:green[Alezen]** 🤖")  # Add emojis and colors to the title

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# React to user input
if prompt := st.chat_input("Ask any question here !"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display assistant response in chat message container

    response = chat_qa(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})