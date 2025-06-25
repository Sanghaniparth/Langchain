import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Streamlit UI
st.title("🧠 LLaMA3 + ObjectBox + Memory Chatbot")

# Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

# Prompt template with memory variable
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the provided context and chat history.

    <context>
    {context}
    </context>

    Chat History:
    {chat_history}

    Question: {input}
    """
)

# Vector DB + memory setup
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:20]
        )
        st.session_state.vectors = ObjectBox.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings,
            embedding_dimensions=768
        )

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            input_key="input"
        )

# UI input
input_prompt = st.text_input("Ask a question from the documents:")

# Embed documents button
if st.button("📥 Embed Documents"):
    vector_embedding()
    st.success("✅ ObjectBox Vectorstore is ready.")

# Run retrieval chain with memory
if input_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please embed documents first.")
    else:
        # Initialize chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Use memory manually
        chat_history = st.session_state.memory.chat_memory.messages

        start = time.process_time()
        response = retrieval_chain.invoke({
            "input": input_prompt,
            "chat_history": chat_history
        })
        elapsed = time.process_time() - start

        # Display answer
        st.write(f"🧠 Answer: {response['answer']}")
        st.caption(f"⏱️ Response time: {elapsed:.2f} seconds")

        # Store interaction in memory
        st.session_state.memory.chat_memory.add_user_message(input_prompt)
        st.session_state.memory.chat_memory.add_ai_message(response['answer'])

        # Show similar docs
        with st.expander("📚 Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.write("---")

        # Show chat history
        with st.expander("🗣️ Chat History"):
            for msg in chat_history:
                st.write(f"**{msg.type.capitalize()}:** {msg.content}")
