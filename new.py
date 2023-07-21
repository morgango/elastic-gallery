"""Python file to serve as the frontend"""
# streamlit stuff
import streamlit as st
from streamlit_chat import message

# langchain stuff
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.elasticsearch import ElasticsearchEmbeddings
from langchain.vectorstores import ElasticVectorSearch
# from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from elasticsearch import Elasticsearch

# environmental stuff
import os
from dotenv import load_dotenv, find_dotenv

# support stuff
import tempfile
import logging

logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG) # or any level you need

# create console handler and set level to debug
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to handler
handler.setFormatter(formatter)

# add handler to logger
logger.addHandler(handler)

env_list = ("elasticsearch_user", 
            "elasticsearch_pw", 
            "elasticsearch_host", 
            "elasticsearch_port", 
            "elasticsearch_model", 
            "elasticsearch_cloud_id", 
            "elasticsearch_index", 
            "openai_api_key")

def check_env():
    if not load_dotenv(find_dotenv()):
        print("ERROR: load_dotenv returned error")
        return False

    if all(env in os.environ for env in env_list):
        print("ENV file found!")
        return True

    return False


def write_env(session_state=st.session_state, filename=".env"):
    if session_state:
        with open(filename, 'w') as f:
            for key, value in st.session_state.items():
                if key in env_list:
                    f.write(f'{key}="{value}"\n')

def write_temp_file(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    return tmp_file_path

def extract_text_from_upload(uploaded_file, encoding="utf8", csv_loader_args={'delimiter': ','}):
    
    tmp_file_path = write_temp_file(uploaded_file)
    file_type = uploaded_file.type.lower().strip()
    if "pdf" in file_type:
        loader = PyPDFLoader(file_path=tmp_file_path)
    elif "csv" in file_type:
        loader = CSVLoader(file_path=tmp_file_path, 
                        encoding=encoding, 
                        csv_args=csv_loader_args)
    elif "text" in file_type:
        loader = TextLoader(tmp_file_path)
    else:
        loader = TextLoader(tmp_file_path)

    text = loader.load()
    return text


def load_document_text(text, 
                       elasticsearch_url=None, 
                       index_name=None, 
                       embeddings=None, 
                       separator="\n", 
                       chunk_size=1000, 
                       chunk_overlap=200, 
                       length_function=len):

    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
    )

    chunks = text_splitter.split_documents(text)
    
    # this will upload the chunks to the elasticsearch cluster
    uploaded = ElasticVectorSearch.from_documents(chunks, embeddings,
                                                elasticsearch_url=elasticsearch_url, 
                                                index_name=index_name)

def ask_es_question(question, elasticsearch_url=None, index_name=None, embeddings=None):
    
    knowldedge_base = ElasticVectorSearch(elasticsearch_url=elasticsearch_url, 
                                            index_name=index_name, 
                                            embedding=embeddings)
    docs = knowldedge_base.similarity_search(question)

    return docs


def load_conv_chain(temperature=0.5, open_api_key=""):
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=temperature, openai_api_key=open_api_key)
    chain = ConversationChain(llm=llm)
    return chain

def get_connections(provider="elasticsearch", es_connection_args={'url':'', 'cloud_id': '', 'user':'', 'pw':'', 'model_id': ''}, **kwargs):

    cleaned_provider=provider.lower().strip()
    
    if cleaned_provider == "openai":
        embeddings = OpenAIEmbeddings()
        llm = OpenAI()
    elif cleaned_provider == "elasticsearch":
        conn = Elasticsearch(es_connection_args['url'])
        # conn = Elasticsearch(cloud_id=es_connection_args['cloud_id'], 
        #                      basic_auth=(es_connection_args['user'], 
        #                                  es_connection_args['pw'])) 
        embeddings = ElasticsearchEmbeddings.from_es_connection(es_connection_args['model_id'], conn)
        llm = OpenAI()
    else:
        embeddings = OpenAIEmbeddings()
        llm = OpenAI()

    return embeddings, llm

# load the .env file and retrieve the things we need
check_env()

# NOTE: You will need to populate a file called .env in the same directory as main.py
st.session_state['openai_api_key'] = os.environ.get('openai_api_key')
st.session_state['elasticsearch_pw'] = os.environ.get('elasticsearch_pw')
st.session_state['elasticsearch_user'] = os.environ.get('elasticsearch_user')
st.session_state['elasticsearch_host'] = os.environ.get('elasticsearch_host')
st.session_state['elasticsearch_port'] = os.environ.get('elasticsearch_port')
st.session_state['elasticsearch_model_id'] = os.environ.get('elasticsearch_model_id')
st.session_state['elasticsearch_cloud_id'] = os.environ.get('elasticsearch_cloud_id')
st.session_state['elasticsearch_index'] = os.environ.get('elasticsearch_index')

# build a few useful values
st.session_state['elasticsearch_url'] = f"https://{st.session_state.elasticsearch_user}:{st.session_state.elasticsearch_pw}@{st.session_state.elasticsearch_host}:{st.session_state.elasticsearch_port}"
os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key

logger.debug(st.session_state)
chain = load_conv_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

# embeddings, llm = get_connections("openai")

# conn_args={}
# conn_args['user'] = ELASTICSEARCH_USER
# conn_args['pw'] = ELASTICSEARCH_PW
# conn_args['cloud_id'] = ELASTICSEARCH_CLOUD_ID
# conn_args['model_id'] = ELASTICSEARCH_MODEL_ID
# conn_args['url'] = ELASTICSEARCH_URL

# embeddings, llm = get_connections(es_connection_args=conn_args)
embeddings, llm = get_connections(provider="openai")

qa, dumb_chat_bot, smarter_chat_bot, preferences = st.tabs(["File Q&A - PDF,CSV,TXT", "ChatBot", "ChatBot - Smarter", "Preferences"])
def test_callback():
    if not st.session_state.get("callback_executed", False):
        logger.debug("Running test_callback()")
        logger.debug(f"cloud_id={cloud_id}")
        logger.debug(f"open_api_key={open_api_key}")
        logger.debug(f"username={username}")
        logger.debug(f"password={password}")
        logger.debug(f"index_name={index_name}")
        st.session_state.callback_executed = True

with preferences:

    with st.form("chat_form"):
        input_col1, input_col2, input_col3, input_col4 = st.columns(4)
        cloud_id = input_col1.text_input("cloud_id: ", type='password')
        open_api_key = input_col1.text_input("openai_api_key: ", type='password')
        username = input_col2.text_input("username: ")
        password = input_col3.text_input("password: ", type='password')
        index_name = input_col4.text_input("index_name: ")
        submit_button = st.form_submit_button("Save")

    if submit_button and not  st.session_state.get('callback_executed'):
        test_callback()

    # with st.form("chat_form"):
    #     input_col1, input_col2, input_col3, input_col4 = st.columns(4)
    #     cloud_id = input_col1.text_input("cloud_id: ", type='password')
    #     open_api_key = input_col1.text_input("openai_api_key: ", type='password')
    #     username = input_col2.text_input("username: ")
    #     password = input_col3.text_input("password: ", type='password')
    #     index_name = input_col4.text_input("index_name: ")
    #     submit_button = st.form_submit_button("Save")

    #     if submit_button:
    #         st.session_state["submitted"] = True
    #         test_callback()
    #     else:
    #         st.session_state["submitted"] = False
        
    #     if st.session_state.get("submitted"):
    #         logger.debug(f"submit_button={submit_button}")

with qa:
    st.header("Ask your file 💬")
    
    uploaded_files = st.file_uploader("Upload your file", 
                                      type=['txt', 'text', 'pdf', 'csv'], 
                                      accept_multiple_files=True)

    for uploaded_file in uploaded_files:

        if uploaded_file is not None:
            
            # extract the text
            document_text = extract_text_from_upload(uploaded_file)
            
            # load the data into Elasticsearch
            load_document_text(document_text, 
                                elasticsearch_url=st.session_state.elasticsearch_url, 
                                index_name=st.session_state.elasticsearch_index, 
                                embeddings=embeddings)

    # show user input
    user_question = st.text_input("Ask a question about your PDF:")
    
    if user_question:
        answer_docs = ask_es_question(user_question, 
                            elasticsearch_url=st.session_state.elasticsearch_url, 
                            index_name=st.session_state.elasticsearch_index, 
                            embeddings=embeddings)

        if answer_docs:
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                print(f"answer_docs={answer_docs}")
                response = chain.run(input_documents=answer_docs, 
                                     question=user_question)
                print(cb)
                
            st.write(response)

with dumb_chat_bot:
    st.header("Unaugmented Chat")
    # if "generated" not in st.session_state:
    #     st.session_state["generated"] = []

    # if "past" not in st.session_state:
    #     st.session_state["past"] = []

    # def get_text():
    #     input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    #     return input_text

    # user_input = get_text()

    # if user_input:
        
    #     chat_chain = load_conv_chain()

    #     output = chat_chain.run(input=user_input)

    #     st.session_state.past.append(user_input)
    #     st.session_state.generated.append(output)

    # if st.session_state["generated"]:

    #     for i in range(len(st.session_state["generated"]) - 1, -1, -1):
    #         message(st.session_state["generated"][i], key=str(i))
    #         message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
