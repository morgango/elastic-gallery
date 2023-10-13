# langchain stuff
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
# from langchain.vectorstores import ElasticVectorSearch
from langchain.vectorstores import ElasticsearchStore
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

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
    # Create an instance of ElasticVectorSearch (the old way of doing this)
    # knowldedge_base = ElasticVectorSearch(elasticsearch_url=elasticsearch_url, 
    #                                       index_name=index_name, 
    #                                       embedding=embeddings)

    # Create an instance of ElasticStore
    knowldedge_base = ElasticsearchStore(es_url=elasticsearch_url, 
                                          index_name=index_name, 
                                          embedding=embeddings,
                                          strategy=ElasticsearchStore.ExactRetrievalStrategy())

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
