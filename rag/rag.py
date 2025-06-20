from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.embeddings import OllamaEmbeddings  # ✅ updated import
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import os

# Step 1: Load PDF
loader = PyPDFLoader('attention.pdf')
docs = loader.load()

# Step 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

# Step 3: Create embeddings and vectorstore
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
db = FAISS.from_documents(documents, embedding_model)

# Step 4: Prompt template (corrected)
prompt = ChatPromptTemplate.from_template(
    """Answer the following question based on the provided context.
Think step by step before providing the detailed answer.
I will tip you $1000 if the user finds the answer helpful.

<context>{context}</context>

Question: {input}"""
)

# Step 5: Chain setup
llm = Ollama(model="llama3")
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Step 6: Run a query
# re = retrieval_chain.invoke({"input": "Scaled Dot-Product Attention"})
# print(re)
