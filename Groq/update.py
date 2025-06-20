import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# ==== ENV ====
load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]

# ==== UI ====
st.set_page_config(page_title="🔍 ChatGroq RAG", layout="wide")
st.title("🔍 ChatGroq RAG Assistant")

# === Model Selection ===
model_choice = st.radio("Choose a model:", ["meta-llama/llama-4-scout-17b-16e-instruct", "llama3-8b-8192"], horizontal=True)

# === Web URL Input ===
web_url = st.text_input("Enter a documentation URL", placeholder="https://docs.smith.langchain.com/")

# === Load Docs Button ===
if st.button("📥 Load & Index Webpage"):
    if web_url:
        with st.spinner("Loading and embedding documents..."):
            st.session_state.embeddings = OllamaEmbeddings()
            loader = WebBaseLoader(web_url)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs[:50])  # limit for speed
            st.session_state.vectors = FAISS.from_documents(chunks, st.session_state.embeddings)
            st.success("✅ Document loaded and vector store created.")
    else:
        st.warning("Please provide a valid URL first.")

# === Prompt Input ===
if "vectors" in st.session_state:
    st.divider()
    user_prompt = st.text_input("💬 Ask your question based on the loaded content")
    if st.button("🚀 Submit"):
        # LLM setup
        llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_choice)

        # Retrieval Chain
        prompt_template = ChatPromptTemplate.from_template("""
        Answer the question based on the provided context only.
        <context>
        {context}
        <context>
        Question: {input}
        """)
        doc_chain = create_stuff_documents_chain(llm, prompt_template)
        retriever = st.session_state.vectors.as_retriever()
        chain = create_retrieval_chain(retriever, doc_chain)

        with st.spinner("Generating response..."):
            start = time.process_time()
            response = chain.invoke({"input": user_prompt})
            end = time.process_time()

            # Response
            st.subheader("🧠 Answer")
            st.write(response["answer"])
            st.caption(f"⏱️ Response time: {end - start:.2f} seconds")

            # Context chunks
            with st.expander("🔍 Retrieved Document Chunks"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)
                    st.write("---")
else:
    st.info("⚠️ Load a webpage first to begin querying.")
