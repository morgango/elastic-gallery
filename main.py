"""Python file to serve as the frontend"""
# streamlit stuff
import streamlit as st
from streamlit_chat import message

# langchain stuff
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

# environmental stuff
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY)
    chain = ConversationChain(llm=llm)
    return chain

chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

qa_pdf, qa_csv, chat_bot = st.tabs(["QA-PDF", "QA-CSV", "ChatBot"])

with qa_pdf:
   st.header("Question and Answer - PDF ")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

with qa_csv:
   st.header("Question and Answer - CSV")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with chat:
    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []


    def get_text():
        input_text = st.text_input("You: ", "Hello, how are you?", key="input")
        return input_text

    user_input = get_text()

    if user_input:
        output = chain.run(input=user_input)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state["generated"]:

        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
