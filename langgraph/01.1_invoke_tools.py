from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

tool = TavilySearch(max_results=2)
tools = [tool]
response = tool.invoke("What's a 'node' in LangGraph?")
print(response["results"])
