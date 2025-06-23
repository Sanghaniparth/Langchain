from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate

# === Function to load & split documents ===
def load_and_split(path):
    loader = PyPDFLoader(path)
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(raw_docs)

# === Load PDFs ===
jd_docs = load_and_split("jd.pdf")
resume1_docs = load_and_split("3.pdf")
resume2_docs = load_and_split("4.pdf")

# === Embedding Model ===
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

# === Vectorstores ===
jd_vs = FAISS.from_documents(jd_docs, embedding_model)
resume1_vs = FAISS.from_documents(resume1_docs, embedding_model)
resume2_vs = FAISS.from_documents(resume2_docs, embedding_model)

# === Prompt Template (no static JD injection) ===
prompt = ChatPromptTemplate.from_template(
    """You are an HR expert. Given context from the job description and a candidate's resume, evaluate the fit.

<context>
{context}
</context>

Question: Is this resume a good match for the JD? Rate it out of 10 and justify your answer.
"""
)

# === LLM & Chains ===
llm = Ollama(model="llama3")
stuff_chain = create_stuff_documents_chain(llm, prompt)

# === Combine JD + Resume Retrieval (Best Practice) ===
def evaluate_resume(resume_vs):
    jd_chunks = jd_vs.similarity_search("job requirements", k=3)
    resume_chunks = resume_vs.similarity_search("candidate experience", k=3)
    combined_docs = jd_chunks + resume_chunks
    return stuff_chain.invoke({
    "context": combined_docs,
    "input": "Is this resume a good match for the JD? Rate it out of 10 and justify your answer."})

# === Evaluate Each Resume ===
eval1 = evaluate_resume(resume1_vs)
eval2 = evaluate_resume(resume2_vs)

print("\n🔍 Resume 1 Evaluation:\n", eval1)
print("\n🔍 Resume 2 Evaluation:\n", eval2)


# === Final Ranking ===
rank_prompt = ChatPromptTemplate.from_template(
    """Compare the two candidate evaluations below. Decide who fits the JD better.
Say 'Resume 1', 'Resume 2', 'Both', or 'Neither', and justify.

Resume 1 Evaluation:
{eval1}

Resume 2 Evaluation:
{eval2}

Final Decision:"""
)

ranking_chain = rank_prompt | llm

eval1_text = eval1
eval2_text = eval2

# Run ranking
decision = ranking_chain.invoke({
    "eval1": eval1_text,
    "eval2": eval2_text
})

print("\n🏆 Final Ranking:\n", decision)
