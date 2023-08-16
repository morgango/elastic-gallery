
# environment stuff
from dotenv import find_dotenv
from common.environment import load_env, create_new_es_index, load_session_vars, build_app_vars, env_list

# data chunking and loading stuff
from common.process import get_connections, extract_text_from_upload, load_document_text

# LLM interaction stuff
from common.query import ask_es_question, get_openai_callback, load_conv_chain 
from langchain.chains.question_answering import load_qa_chain

# UI stuff
import streamlit as st
from streamlit_chat import message

# load the environment file if needed.
if not 'env_file_loaded' in st.session_state:

    # upload variables from the .env file as environment variables
    env_path = find_dotenv()
    env_vars = dict(load_env(env_path=env_path))

    # load our values from the .env file into the session state.
    # only add our values, not the entire environment.
    load_session_vars(os_env_vars=env_vars, key_list=env_list)
    build_app_vars()

    # don't load this multiple times
    st.session_state['env_file_loaded'] = True

if 'new_index_created' not in st.session_state:
    print(f"{st.session_state}")
    create_new_es_index(index_name=st.session_state.elasticsearch_index, elasticsearch_url=st.session_state.elasticsearch_url) 
    st.session_state['new_index_created'] = True

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

# Initialize embeddings and language model
embeddings, llm = get_connections(provider="openai")

# Create Streamlit tabs
qa, dumb_chat_bot, smarter_chat_bot = st.tabs(["File Q&A - PDF,CSV,TXT", "ChatBot", "ChatBot - Smarter"])

# Define content and functionality for the "File Q&A - PDF,CSV,TXT" tab
with qa:
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

    # Allow users to ask a question about the uploaded file(s)
    user_question = st.text_input("Ask a question about your file(s):")
    
    # Process the user's question
    if user_question:
        # Search for documents related to the question in Elasticsearch
        answer_docs = ask_es_question(user_question, 
                            elasticsearch_url=st.session_state.elasticsearch_url,
                            index_name=st.session_state.elasticsearch_index,
                            embeddings=embeddings)

        if answer_docs:

            if 'qa_chain' not in st.session_state:
                # Initialize QA chain if it's not in session state
                st.session_state['qa_chain'] = load_qa_chain(llm, chain_type="stuff")
            
            # Retrieve the QA chain from session state
            qa_chain = st.session_state['qa_chain']
            
            # Run the QA chain to generate a response to the user's question
            with get_openai_callback() as cb:
                response = qa_chain.run(input_documents=answer_docs, 
                                     question=user_question)
                
            # Display the generated response
            st.write(response)

# Define content and functionality for the "ChatBot" tab
with dumb_chat_bot:

    #############################################################################################
    # This needes to be updated to use st.cat_message and st.chat_input
    # https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
    #############################################################################################
    st.header("Unaugmented Chat")

    # Initialize conversation chain if it's not in session state
    if "dumb_conv_chain" not in st.session_state:
        conv_chain = load_conv_chain(open_api_key=st.session_state.openai_api_key)
        st.session_state['dumb_conv_chain'] = conv_chain

    # Initialize chat history if it's not in session state
    if "dumb_chat_generated" not in st.session_state:
        st.session_state["dumb_chat_generated"] = []

    # Initialize user's past chat if it's not in session state
    if "dumb_chat_past" not in st.session_state:
        st.session_state["dumb_chat_past"] = []

    # Allow users to input text
    def get_text():
        # input_text = st.text_input("You: ", "Hello, how are you?", key="input")
        input_text = st.text_input("You: ", "", key="input")
        return input_text

    # Retrieve the user's input
    user_input = get_text()

    # Process the user's input
    if user_input:
        # Run the conversation chain to generate a response to the user's input
        c =  st.session_state['dumb_conv_chain']
        output = c.run(input=user_input)

        # Add the user's input and the generated response to chat history
        st.session_state.dumb_chat_past.append(user_input)
        st.session_state.dumb_chat_generated.append(output)

    # Display chat history
    if st.session_state["dumb_chat_generated"]:
        for i in range(len(st.session_state["dumb_chat_generated"]) - 1, -1, -1):
            message(st.session_state["dumb_chat_generated"][i], key=str(i))
            message(st.session_state["dumb_chat_past"][i], is_user=True, key=str(i) + "_user")

# Define content and functionality for the "ChatBot - Smarter" tab
with smarter_chat_bot:

    pass
    # import openai

    # st.title("ChatGPT-like clone")


    # openai.api_key = os.environ["OPENAI_API_KEY"]

    # if "openai_model" not in st.session_state:
    #     st.session_state["openai_model"] = "gpt-3.5-turbo"

    # if "messages" not in st.session_state:
    #     st.session_state.messages = []

    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])

    # if prompt := st.chat_input("What is up?"):
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #     with st.chat_message("user"):
    #         st.markdown(prompt)

    #     with st.chat_message("assistant"):
    #         message_placeholder = st.empty()
    #         full_response = ""
    #         for response in openai.ChatCompletion.create(
    #             model=st.session_state["openai_model"],
    #             messages=[
    #                 {"role": m["role"], "content": m["content"]}
    #                 for m in st.session_state.messages
    #             ],
    #             stream=True,
    #         ):
    #             full_response += response.choices[0].delta.get("content", "")
    #             message_placeholder.markdown(full_response + "▌")
    #         message_placeholder.markdown(full_response)
    #     st.session_state.messages.append({"role": "assistant", "content": full_response})
