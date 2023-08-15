"""405Python file to serve as the frontend"""
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
import logging
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError, RequestError, BadRequestError

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

# all the variables in the .env file that we care about
env_list = ("elasticsearch_user", 
            "elasticsearch_pw", 
            "elasticsearch_host", 
            "elasticsearch_port", 
            "elasticsearch_model_id", 
            "elasticsearch_cloud_id", 
            "elasticsearch_index", 
            "openai_api_key")

def write_temp_file(uploaded_file):
    """
    This function writes the contents of an uploaded file to a temporary file on the system, 
    and then returns the path to that temporary file.

    Parameters:
    uploaded_file (io.BytesIO): The uploaded file as a BytesIO object.

    Returns:
    str: The path to the temporary file on the system.
    """
    # Create a new temporary file using the NamedTemporaryFile function from the tempfile module
    # The delete=False parameter means that the file won't be deleted when it's closed
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        # Write the contents of the uploaded file to the temporary file
        tmp_file.write(uploaded_file.getvalue())
        # Save the path to the temporary file
        tmp_file_path = tmp_file.name

    # Return the path to the temporary file
    return tmp_file_path

def extract_text_from_upload(uploaded_file, encoding="utf8", 
                             csv_loader_args={'delimiter': ','}):
    """
    This function extracts text data from an uploaded file. The type of file - PDF, CSV, or text - 
    is determined by checking the file's type. Appropriate loaders are used to extract the text
    based on the file type.

    Parameters:
    uploaded_file (UploadedFile): The file uploaded by the user.
    encoding (str): Optional. The encoding to be used for text or CSV files. Defaults to "utf8".
    csv_loader_args (dict): Optional. Additional arguments to be passed to the CSV loader 
                            (such as the delimiter). Defaults to {'delimiter': ','}.

    Returns:
    str: The extracted text data from the file.
    """
    # Write the uploaded file to a temporary file on the system
    tmp_file_path = write_temp_file(uploaded_file)

    # Convert the file's type to lower case and remove leading/trailing whitespace
    file_type = uploaded_file.type.lower().strip()

    # Check the file type and create the appropriate loader
    if "pdf" in file_type:
        loader = PyPDFLoader(file_path=tmp_file_path)
    elif "csv" in file_type:
        loader = CSVLoader(file_path=tmp_file_path, 
                           encoding=encoding, 
                           csv_args=csv_loader_args)
    elif "text" in file_type:
        loader = TextLoader(tmp_file_path)
    else:
        # If the file type isn't recognized, assume it's a text file
        loader = TextLoader(tmp_file_path)

    # Use the loader to extract the text data from the file
    text = loader.load()

    # Return the extracted text
    return text

def load_document_text(text, 
                       elasticsearch_url=None, 
                       index_name=None, 
                       embeddings=None, 
                       separator="\n", 
                       chunk_size=1000, 
                       chunk_overlap=200, 
                       length_function=len):
    """
    This function splits a given text into chunks, and then uploads these chunks to an Elasticsearch 
    cluster using an instance of the ElasticVectorSearch class.

    Parameters:
    text (str): The text to be split and uploaded.
    elasticsearch_url (str): Optional. The URL of the Elasticsearch cluster. Defaults to None.
    index_name (str): Optional. The name of the Elasticsearch index where the chunks will be uploaded. Defaults to None.
    embeddings (np.ndarray or list of np.ndarray): Optional. The embeddings for the chunks, if any. Defaults to None.
    separator (str): Optional. The character used to split the text into chunks. Defaults to "\n".
    chunk_size (int): Optional. The size of each chunk. Defaults to 1000.
    chunk_overlap (int): Optional. The size of the overlap between consecutive chunks. Defaults to 200.
    length_function (function): Optional. The function used to compute the length of a chunk. Defaults to len.

    This function first creates an instance of the CharacterTextSplitter class, which splits the input text into 
    chunks. Each chunk will be of size `chunk_size`, and consecutive chunks will overlap by `chunk_overlap` characters.
    The chunks are split at the character specified by `separator`.

    The chunks are then uploaded to the Elasticsearch cluster at the specified URL, and stored in the specified index.
    If embeddings are provided, they will be associated with the chunks.
    """
    # Create an instance of CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
    )

    # Split the text into chunks
    chunks = text_splitter.split_documents(text)
    logger.debug(f"Index Name = {index_name}")
    # Upload the chunks to the Elasticsearch cluster
    uploaded = ElasticVectorSearch.from_documents(chunks, embeddings,
                                                  elasticsearch_url=elasticsearch_url, 
                                                  index_name=index_name)

def ask_es_question(question, 
                    elasticsearch_url=None, 
                    index_name=None, 
                    embeddings=None):
    """
    This function uses an instance of the ElasticVectorSearch class to perform a similarity search on a given 
    Elasticsearch index. It takes a question as input and returns the documents that are most similar to the question.

    Parameters:
    question (str): The question to be used for the similarity search.
    elasticsearch_url (str): Optional. The URL of the Elasticsearch cluster. Defaults to None.
    index_name (str): Optional. The name of the Elasticsearch index where the search will be performed. Defaults to None.
    embeddings (np.ndarray or list of np.ndarray): Optional. The embeddings for the question, if any. Defaults to None.

    Returns:
    list: A list of documents that are most similar to the input question.

    The function first creates an instance of the ElasticVectorSearch class, which connects to the specified Elasticsearch
    cluster and index, and sets up the specified embeddings.

    The function then performs a similarity search on the index using the input question and returns the most similar documents.
    """
    # Create an instance of ElasticVectorSearch
    knowldedge_base = ElasticVectorSearch(elasticsearch_url=elasticsearch_url, 
                                          index_name=index_name, 
                                          embedding=embeddings)

    # Perform a similarity search on the index using the input question
    docs = knowldedge_base.similarity_search(question)

    # Return the most similar documents
    return docs

def load_conv_chain(temperature=0.5, 
                    open_api_key=""):
    """
    This function creates and returns an instance of the ConversationChain class, which uses an instance 
    of the OpenAI class to generate responses in a conversation.

    Parameters:
    temperature (float): Optional. The temperature parameter for the OpenAI API, which controls the randomness 
                         of the generated responses. A higher value makes the responses more random, while 
                         a lower value makes them more deterministic. Defaults to 0.5.
    open_api_key (str): Optional. The API key for OpenAI. Defaults to "".

    Returns:
    ConversationChain: An instance of the ConversationChain class.

    The function first creates an instance of the OpenAI class, using the provided temperature and API key. 
    This OpenAI instance is responsible for generating responses in a conversation.

    The function then creates an instance of the ConversationChain class, using the OpenAI instance, and returns it.
    """
    # Create an instance of OpenAI
    llm = OpenAI(temperature=temperature, openai_api_key=open_api_key)
    
    # Create an instance of ConversationChain
    chain = ConversationChain(llm=llm)

    # Return the ConversationChain instance
    return chain

def get_connections(provider="openai", **kwargs):
    """
    This function initializes and returns an embeddings provider and a language model 
    based on the specified provider name.

    Parameters:
    provider (str): Optional. The name of the provider to use for embeddings and language modeling. Defaults to "openai".
    kwargs (dict): Optional. Additional arguments that can be passed to the embeddings provider and language model.

    Returns:
    tuple: A tuple containing the embeddings provider and the language model.

    The function checks the provider name, and based on this, it creates and returns an instance of the OpenAIEmbeddings 
    class (for generating embeddings) and an instance of the OpenAI class (for language modeling).

    Currently, it supports only the "openai" provider, and if any other provider name is passed, it will default to "openai".

    Note: Additional parameters for the embeddings provider and language model can be passed via **kwargs, but they are not 
    used in the current implementation.
    """
    # Clean the provider name
    cleaned_provider = provider.lower().strip()

    # Check the provider name and create the appropriate embeddings provider and language model
    if cleaned_provider == "openai":
        embeddings = OpenAIEmbeddings()
        llm = OpenAI()
    else:
        # Default to "openai" if the provider name is not recognized
        embeddings = OpenAIEmbeddings()
        llm = OpenAI()

    # Return the embeddings provider and the language model
    return embeddings, llm

def load_env(env_path=None):
    """
    This function loads environment variables from a .env file using the python-dotenv library, 
    and then returns a dictionary-like object containing the environment variables.

    Parameters:
    env_path (str): Optional. The path to the .env file. If not provided, the load_dotenv() function 
                    will look for the .env file in the current directory and if not found, then 
                    recursively up the directories. Defaults to None.

    Returns:
    os._Environ: This is a dictionary-like object which allows accessing environment variables.
    """
    # Load environment variables from a .env file
    load_dotenv(env_path)

    # Return a dictionary-like object containing the environment variables
    return os.environ

def load_session_vars(os_env_vars={}, 
                      key_list=None):
    """
    This function loads variables into Streamlit's session state.

    Parameters:
    os_env_vars (dict): Optional. A dictionary of environment variables to load into the session state. 
                        Defaults to an empty dictionary.
    key_list (list): Optional. A list of keys. If provided, only environment variables 
                     with these keys will be loaded into the session state. Defaults to None.

    The function first checks if key_list is provided. If so, it creates a new dictionary
    with key-value pairs where keys are elements in key_list and the values are corresponding
    values from os.environ.
    
    If key_list is not provided, it uses the os_env_vars dictionary.
    
    It then logs the dictionary of variables that will be loaded into the session state.

    Finally, it iterates over the dictionary of variables, logging each one, and adds each 
    variable to the session state.
    """
    # Check if key_list is provided
    if key_list is not None:
        # If so, create a new dictionary with key-value pairs for each key in key_list
        vars = {env: os.environ[env] for env in key_list if env in os.environ}
    else:
        # If not, use the os_env_vars dictionary
        vars = os_env_vars


    # Iterate over the dictionary of variables
    for key, value in vars.items():
        # Log each variable
        logger.debug(f"Setting st.{key} = {value}")

        # Add the variable to the session state
        st.session_state[key] = value

def build_app_vars():
    """
    This function builds the elasticsearch_url for use in the Streamlit app and stores it in the session state.

    It uses the Elasticsearch host, port, user, and password from the session state to construct the URL in the format required.

    Note: It's critical to ensure that the necessary variables (elasticsearch_user, elasticsearch_pw, elasticsearch_host, elasticsearch_port) are available in the session state before calling this function.

    """

    # Construct the Elasticsearch URL using the host, port, user, and password
    # The URL is formatted according to the Elasticsearch's standard URL format

    if st.session_state['elasticsearch_user'] and \
        st.session_state['elasticsearch_pw'] and \
        st.session_state['elasticsearch_host'] and \
        st.session_state['elasticsearch_port']:

        st.session_state['elasticsearch_url'] = f"https://{st.session_state['elasticsearch_user']}:{st.session_state['elasticsearch_pw']}@{st.session_state['elasticsearch_host']}:{st.session_state['elasticsearch_port']}"
        logger.debug(f"elasticsearch_url = f{st.session_state.elasticsearch_url}")

    else:
        logger.error("The elasticsearch_url could not be written because elasticsearch_user, elasticsearch_pw, elasticsearch_host, and elasticsearch_port are not defined in the session state")
    # we set this as an environment variable because it has to be present
    # in so many different places

    if 'open_api_key' not in st.session_state: 
        os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
    else:
        logger.error("The open_api_key is not defined in the session state.")

def create_new_es_index(index_name=None, elasticsearch_url=None):
    """
    Creates a new Elasticsearch index with a predefined mapping. If the index already exists,
    it will be deleted and a new index with the specified name will be created.

    Parameters:
        index_name (str, optional): The name of the index to be created. Defaults to None.
        elasticsearch_url (str, optional): The URL of the Elasticsearch cluster. Defaults to None.

    Raises:
        NotFoundError: If the index specified in index_name does not exist.
        RequestError: If there's an error in creating the index.

    Example:
        create_new_es_index(index_name="my_index", elasticsearch_url="http://localhost:9200")
    """

    # Define the mapping for the index, specifying the data types and structure
    table_mapping = {
        "mappings": {
            "properties": {
                "metadata": {
                    "properties": {
                        "page": {"type": "long"},
                        "row": {"type": "long"},
                        "source": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        }
                    }
                },
                "text": {"type": "text"},
                "vector": {"type": "dense_vector", "dims": 1536}
            }
        }
    }

    # Connect to the Elasticsearch cluster using the provided URL
    es = Elasticsearch([elasticsearch_url])

    # Check if the index exists
    try:
        if not es.indices.exists(index=index_name):
            es.indices.create(index=index_name, mappings=table_mapping["mappings"])

    except NotFoundError as e:
        # Handle case when the index is not found
        logger.error(f"An error occurred: {e}")

    except (RequestError, BadRequestError) as e:
        # Handle other request errors such as BadRequestError
        logger.error(f"An error occurred while creating the index: {e}")