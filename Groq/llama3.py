# import os
# import time 
# import streamlit as st
# from dotenv import load_dotenv

# from langchain_groq import ChatGroq
# from langchain.chains import create_retrieval_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.vectorstores import FAISS,Chroma
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain.chains.combine_documents import create_stuff_documents_chain
# load_dotenv()

# groq_api_key=os.environ['GROQ_API_KEY']

# st.title("GROQ ChatBot")

# llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-8b-8192")
# prompt=ChatPromptTemplate.from_template(
# """
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question
# <context>
# {context}
# <context>
# Questions:{input}

# """
# )
# def vector_embedding():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings=OllamaEmbeddings()
#         st.session_state.loader=PyPDFDirectoryLoader("./us_census") ## Data Ingestion
#         st.session_state.docs=st.session_state.loader.load() ## Document Loading
#         st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
#         st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
#         st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings

# prompt1 = st.text_input("Enter your question")

# if st.button("Document Embedding"):
#     vector_embedding()
#     st.write("Vector Store DB Is Ready")


# if prompt1:
#     document_chain=create_stuff_documents_chain(llm,prompt)
#     retriever=st.session_state.vectors.as_retriever()
#     retrieval_chain=create_retrieval_chain(retriever,document_chain)
#     start=time.process_time()
#     response=retrieval_chain.invoke({'input':prompt1})
#     print("Response time :",time.process_time()-start)
#     st.write(response['answer'])

#     # With a streamlit expander
#     with st.expander("Document Similarity Search"):
#         # Find the relevant chunks
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")
import os
import time 
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

st.title("GROQ ChatBot with PDF RAG")

llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

if st.button("Document Embedding"):
    with st.spinner("Loading and embedding documents..."):
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        loader = PyPDFDirectoryLoader("./us_census")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs[:120])
        vectors = FAISS.from_documents(split_docs, embeddings)
        st.session_state.vectors = vectors
        st.success("Vector store is ready.")

user_query = st.text_input("Enter your question:")

if user_query:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner("Generating answer..."):
            start = time.process_time()
            response = retrieval_chain.invoke({'input': user_query})
            st.subheader("Answer")
            st.write(response['answer'])
            st.caption(f"Response time: {time.process_time() - start:.2f} seconds")

            with st.expander("Document Chunks Used"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"Chunk {i+1}:")
                    st.write(doc.page_content)
                    st.write("---")
    else:
        st.warning("Please embed the documents first using the button above.")
