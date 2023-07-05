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

# environmental stuff
import os
from dotenv import load_dotenv, find_dotenv

# support stuff
from PyPDF2 import PdfReader
import tempfile

# load the .env file and retrieve the things we need
load_dotenv(find_dotenv())
# NOTE: You will need to populate a file called .env in the same directory as main.py
# with key value pairs for the values below.  
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ELASTICSEARCH_URL = os.environ.get("ELASTICSEARCH_URL")
ELASTICSEARCH_INDEX = os.environ.get("ELASTICSEARCH_INDEX")

def extract_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_csv_text(csv):

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(csv.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, 
                       encoding="utf-8", 
                       csv_args={'delimiter': ','})
    text = loader.load()

    return text


def split_and_load_text(text elasticsearch_url=None, index_name=None, embeddings=None):

    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

def ask_es_question(question, elasticsearch_url=None, index_name=None, embeddings=None):
    
    knowldedge_base = ElasticVectorSearch(elasticsearch_url=elasticsearch_url, 
                                            index_name=index_name, 
                                            embedding=embeddings)
    docs = knowldedge_base.similarity_search(question)

    return docs


def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY)
    chain = ConversationChain(llm=llm)
    return chain

chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

qa, dumb_chat_bot, smarter_chat_bot = st.tabs(["QA - PDF or CSV", "ChatBot", "ChatBot - Smarter"])

with qa:
    st.header("Ask your PDF 💬")
    
    # upload file
    uploaded_files = st.file_uploader("Upload your PDF", type=['pdf', 'csv'], accept_multiple_files=True)

    # create embeddings for each individual file
    embeddings = OpenAIEmbeddings()
    llm = OpenAI()

    for uploaded_file in uploaded_files:

        # extract the text
        if uploaded_file is not None:

            document_text = ""
            if "pdf" in uploaded_file.type.lower().strip():
                document_text = extract_pdf_text(uploaded_file)
            elif "csv" in uploaded_file.type.lower().strip():
                document_text = extract_csv_text(uploaded_file)
            
            split_and_load_text(document_text, 
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

with qa_csv:
   st.header("Question and Answer - CSV")

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
