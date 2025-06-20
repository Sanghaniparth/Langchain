import requests
import streamlit as st 
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# def get_openai_response(input_text):
#     response=requests.post("http://localhost:8000/essay/invoke",
#     json={'input':{'topic':input_text}})
#     return response.json()['output']['content']

def get_ollama_response1(input_text):
    response=requests.post(
    "http://localhost:8000/essay/invoke",
    json={'input':{'topic':input_text}})
    return response.json()['output']

def get_ollama_response(input_text):
    response=requests.post(
    "http://localhost:8000/poem/invoke",
    json={'input':{'topic':input_text}})
    return response.json()['output']

st.title("Langchain Demo with LLAMA2")
input_text = st.text_input("Write an essay on")
if input_text:
    st.write(get_ollama_response(input_text))


input_text1 = st.text_input("Write an poem on")
if input_text1:
    st.write(get_ollama_response1(input_text1))

