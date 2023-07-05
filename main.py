"""Python file to serve as the frontend"""
# streamlit stuff
import streamlit as st
from streamlit_chat import message

# langchain stuff
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import ElasticVectorSearch
# from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader

# environmental stuff
import os
from dotenv import load_dotenv, find_dotenv

# support stuff
import tempfile

# load the .env file and retrieve the things we need
load_dotenv(find_dotenv())
# NOTE: You will need to populate a file called .env in the same directory as main.py
# with key value pairs for the values below.  
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ELASTICSEARCH_URL = os.environ.get("ELASTICSEARCH_URL")
ELASTICSEARCH_INDEX = os.environ.get("ELASTICSEARCH_INDEX")

def write_temp_file(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    return tmp_file_path

def extract_text(uploaded_file, encoding="utf8", csv_loader_args={'delimiter': ','}):
    
    tmp_file_path = write_temp_file(uploaded_file)
    file_type = uploaded_file.type.lower().strip():
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


def load_document_text(text, elasticsearch_url=None, index_name=None, embeddings=None):

    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(text)
    uploaded = ElasticVectorSearch.from_documents(chunks, embeddings,
                                                elasticsearch_url=elasticsearch_url, 
                                                index_name=index_name)

def ask_es_question(question, elasticsearch_url=None, index_name=None, embeddings=None):
    
    knowldedge_base = ElasticVectorSearch(elasticsearch_url=elasticsearch_url, 
                                            index_name=index_name, 
                                            embedding=embeddings)
    docs = knowldedge_base.similarity_search(question)

    return docs


def load_conv_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY)
    chain = ConversationChain(llm=llm)
    return chain

chain = load_conv_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

qa, dumb_chat_bot, smarter_chat_bot = st.tabs(["File Q&A - PDF,CSV,TXT", "ChatBot", "ChatBot - Smarter"])

with qa:
    st.header("Ask your PDF 💬")
    
    # upload file
    uploaded_files = st.file_uploader("Upload your PDF", type=['txt', 'text', 'pdf', 'csv'], accept_multiple_files=True)

    # create embeddings for each individual file
    embeddings = OpenAIEmbeddings()
    llm = OpenAI()

    for uploaded_file in uploaded_files:

        if uploaded_file is not None:
            
            # extract the text
            document_text = extract_text(uploaded_file)
            

            load_document_text(document_text, 
                                elasticsearch_url=ELASTICSEARCH_URL, 
                                index_name=ELASTICSEARCH_INDEX, 
                                embeddings=embeddings)

      
    # show user input
    user_question = st.text_input("Ask a question about your PDF:")
    if user_question:
        answer_docs = ask_es_question(user_question, 
                            elasticsearch_url=ELASTICSEARCH_URL,
                            index_name=ELASTICSEARCH_INDEX,
                            embeddings=embeddings)

        if answer_docs:
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
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
    #     output = chain.run(input=user_input)

    #     st.session_state.past.append(user_input)
    #     st.session_state.generated.append(output)

    # if st.session_state["generated"]:

    #     for i in range(len(st.session_state["generated"]) - 1, -1, -1):
    #         message(st.session_state["generated"][i], key=str(i))
    #         message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
