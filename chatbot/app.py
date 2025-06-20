from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
import streamlit as st 
import os 
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "PROJECT1"

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', "You are need to answer the proper question"),
        ('user', "Question: {question}")
    ]
)

st.title("Langchain ollama chatbot")
input_text = st.text_input("Search the topic you want.")


llm = Ollama(model= "llama2")
outputparser = StrOutputParser()

chain = prompt|llm|outputparser

if input_text:
    st.write(chain.invoke({"question":input_text}))