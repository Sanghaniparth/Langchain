from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import Ollama  # or OpenAI, etc.
from langchain import hub
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

llm = Ollama(model="llama3")
# Load a ReAct-style prompt template
prompt = hub.pull("hwchase17/react")

# Step 1: Create a general Wikipedia wrapper
wiki_api = WikipediaAPIWrapper(
    top_k_results=1,                # Only most relevant result
    doc_content_chars_max=300       # Limit content length
)
# Step 2: Wrap it as a Tool (general-purpose)
wikipedia_tool = WikipediaQueryRun(
    api_wrapper=wiki_api,
    name="Wikipedia",
    description=(
        "Use this tool to search Wikipedia and retrieve short summaries "
        "about any public topic such as people, concepts, technologies, or history."
    )
)
# Register tool
tools = [wikipedia_tool]
# Create agent
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
# Create agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# Run dynamically
response = agent_executor.invoke({"input": "What is quantum computing?"})
print(response["output"])
