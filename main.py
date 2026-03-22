from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from tavily import TavilyClient
from typing import List
from pydantic import BaseModel, Field
import os

load_dotenv()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

class Source(BaseModel):
    """Scheme for a Source used by agent"""
    url:str = Field(description="The url of the source")
    
class AgentResponse(BaseModel):
    """Scheme for the agent response"""
    answer:str = Field(description="The agent's answer to the query")
    sources:List[Source] = Field(default_factory= list ,
    description="The list of sources used to generate the answer")


@tool
def search(query: str) -> str:
    """
    Tool that seraches over internet
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"searching for {query}")
    data = tavily.search(query=query)
    return data


llm = ChatOpenAI(model = "gpt-5")
tools = [search]
agent = create_agent(llm, tools, response_format=AgentResponse)

def main():
    print("Hello from project-langchain!")
    response = agent.invoke({"messages": [HumanMessage("Search for 3 job posting for an ai engineer in langchain in the bay area on Linkedin and list their details")]})
    print(response)



if __name__ == "__main__":
    main()
