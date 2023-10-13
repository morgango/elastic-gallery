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
from icecream import ic

# import logging

# logger = logging.getLogger('my_logger')
# logger.setLevel(logging.DEBUG) # or any level you need

# # create console handler and set level to debug
# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)

# # create formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# # add formatter to handler
# handler.setFormatter(formatter)

# # add handler to logger
# logger.addHandler(handler)

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

if 'new_index_created' not in st.session_state:
    create_new_es_index(index_name=st.session_state.elasticsearch_index, es_url=st.session_state.elasticsearch_url, delete_index=True) 
    st.session_state['new_index_created'] = True
    
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
    st.session_state["openai_model"] = "gpt-3.5-turbo-0301"

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
        
        answer_docs = ask_es_question(prompt, 
                    elasticsearch_url=st.session_state.elasticsearch_url,
                    index_name=st.session_state.elasticsearch_index,
                    embeddings=embeddings)

        for doc in answer_docs:
            ic(doc)
            
        if answer_docs:

            if 'qa_chain' not in st.session_state:
                # Initialize QA chain if it's not in session state
                st.session_state['qa_chain'] = load_qa_chain(llm, chain_type="stuff")
            
            # Retrieve the QA chain from session state
            qa_chain = st.session_state['qa_chain']
            
            # Run the QA chain to generate a response to the user's question
            with get_openai_callback() as cb:
                full_response = qa_chain.run(input_documents=answer_docs, question=prompt)
                    
            # Replace the placeholder with the final assistant message
            message_placeholder.markdown(full_response)
        
    # Append the assistant's message to the 'messages' session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
