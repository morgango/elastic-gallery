#######################################
# Loosely based on the Build conversational Apps Tutorial: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
#######################################
import common, common.environment, common.process, common.query

# environment stuff
from dotenv import find_dotenv
from common.environment import load_env, create_new_es_index, load_session_vars, build_app_vars, env_list

# data chunking and loading stuff
from common.process import get_connections, extract_text_from_upload, load_document_text

# LLM interaction stuff
from common.query import ask_es_question, get_openai_callback, load_conv_chain 
from langchain.chains.question_answering import load_qa_chain
import openai

# UI stuff
from elasticsearch import Elasticsearch
import eland as ed
import json
import streamlit as st
from streamlit_chat import message


# load the environment file if needed.
if not 'env_file_loaded' in st.session_state:

    # upload variables from the .env file as environment variables
    env_path = find_dotenv()
    env_vars = dict(load_env(env_path=env_path))

    # load our values from the .env file into the session state.
    # only add our values, not the entire environment.
    load_session_vars(os_env_vars=env_vars, key_list=common.environment.env_list)
    build_app_vars()

    # don't load this multiple times
    st.session_state['env_file_loaded'] = True

st.title("ChatGPT-like clone")

# Initialize embeddings and language model
embeddings, llm = get_connections(provider="openai")

st.header("Ask your file 💬")

# Allow users to upload files
uploaded_files = st.file_uploader("Upload your file", 
                                    type=['txt', 'text', 'pdf', 'csv'], 
                                    accept_multiple_files=True)

# Process each uploaded file
for uploaded_file in uploaded_files:

    if uploaded_file is not None:
        
        # Extract text from the uploaded file
        document_text = extract_text_from_upload(uploaded_file)
        
        # Load the extracted text into Elasticsearch
        load_document_text(document_text, 
                            elasticsearch_url=st.session_state.elasticsearch_url, 
                            index_name=st.session_state.elasticsearch_index, 
                            embeddings=embeddings)

st.divider()

# Initialize the 'openai_model' session state if it doesn't already exist
if "openai_model" not in st.session_state:
    # Default to "gpt-3.5-turbo"
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize the 'messages' session state if it doesn't already exist, this will store chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previously stored messages in the chat
# Loop through the list of messages in the session state
for message in st.session_state.messages:
    # Create a chat message container based on the role (either "user" or "assistant")
    with st.chat_message(message["role"]):
        # Display the message content using Markdown
        st.markdown(message["content"])

# Show chat input field and wait for user input
# When user provides input, it's stored in 'prompt'
if prompt := st.chat_input("What is up?"):
    # Append the user message to the 'messages' session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display the user's message in the chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare to display the assistant's message
    with st.chat_message("assistant"):
        # Placeholder for the assistant's message while it is being generated
        message_placeholder = st.empty()
        full_response = ""
        
        # # Generate assistant's response using OpenAI's model
        # # 'stream=True' allows the model to send portions of the response as they are generated
        # for response in openai.ChatCompletion.create(
        #     model=st.session_state["openai_model"],
        #     messages=[
        #         {"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages
        #     ],
        #     stream=True,
        # ):
        #     # Append new parts of the message to 'full_response'
        #     full_response += response.choices[0].delta.get("content", "")
        #     # Display the response as it's being generated, with a cursor ('▌')
        #     message_placeholder.markdown(full_response + "▌")

        answer_docs = ask_es_question(prompt, 
                    elasticsearch_url=st.session_state.elasticsearch_url,
                    index_name=st.session_state.elasticsearch_index,
                    embeddings=embeddings)

        for doc in answer_docs:
            
        
        # Replace the placeholder with the final assistant message
        # message_placeholder.markdown(full_response)
        message_placeholder.markdown(answer_docs)
        
    # Append the assistant's message to the 'messages' session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
