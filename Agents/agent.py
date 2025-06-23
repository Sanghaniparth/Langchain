from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap= 200).split_documents(docs)
vectordb =FAISS.from_documents(documents, OllamaEmbeddings())
retriever = vectordb.as_retriever()

###########
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
# wiki.name
############
from langchain.tools.retriever import create_retriever_tool

retriever_tools  = create_retriever_tool(retriever,"Langsmith Search","Search for information about Langsmith")
##########

## Arxiv Tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
    
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)
# arxiv.name

tools =[wiki,arxiv,retriever_tools]

from langchain_community.llms import Ollama
llm = Ollama(model="llama3", temperature=0.3 )

from langchain import hub
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/react")
prompt.messages

from langchain.agents import create_react_agent
agent=create_react_agent(llm,tools,prompt)

from langchain.agents import AgentExecutor
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)
agent_executor

res = agent_executor.invoke({"input":"Tell me about Langsmith"})
print(res)

# res1 = agent_executor.invoke({"input":"What's the paper 1605.08386 about?"})
# print(res1)