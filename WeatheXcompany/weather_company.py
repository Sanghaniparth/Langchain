from langchain_community.document_loaders import WebBaseLoader,PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.tools import Tool
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENWEATHERMAP_API_KEY"] = os.getenv("OPENWEATHERMAP_API_KEY")


loader = PyPDFLoader("Document.pdf")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap= 200).split_documents(docs)
vectordb =FAISS.from_documents(documents, OllamaEmbeddings(model="mxbai-embed-large"))
retriever = vectordb.as_retriever()


# from langchain_community.tools.openweathermap.tool import OpenWeatherMapQueryRun
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
weather_wrapper = OpenWeatherMapAPIWrapper()
weather_tool = Tool.from_function(
    func=weather_wrapper.run ,name="GetWeather", description="Use this tool to get weather data for the cities"
)
# weather_tool = OpenWeatherMapQueryRun(api_wrapper=weather_wrapper)

from langchain.tools.retriever import create_retriever_tool

retriever_tools  = create_retriever_tool(retriever,"Comapny Search","Search for information about some Comapnies.")


tools =[weather_tool,retriever_tools]

from langchain_community.llms import Ollama
llm = Ollama(model="llama3", temperature=0.3 )

from langchain import hub
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/react")

from langchain.agents import create_react_agent
agent=create_react_agent(llm,tools,prompt)

from langchain.agents import AgentExecutor
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True,handle_parsing_errors=True)
agent_executor

res = agent_executor.invoke({"input": "what is weather in ahmedabad?"})
print(res["output"])

# while True:
#     query = input("Ask (or type 'exit' to quit): ")
#     if query.strip().lower() in ["exit", "quit"]:
#         print("Goodbye!")
#         break
#     try:
#         response = agent_executor.invoke({"input": query})
#         print(response['output'])
#     except Exception as e:
#         print(" Error:", str(e))
#     finally:
#         print("Reminder: Please ask about the company (from PDF) or the weather.\n")