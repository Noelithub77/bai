from langchain_google_genai import ChatGoogleGenerativeAI   
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from datetime import datetime
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver  
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from langchain.agents import AgentExecutor, create_tool_calling_agent

# Initialize LLM
model = ChatGoogleGenerativeAI(
    # model="gemini-1.5-pro",
    # model="gemini-1.5-flash-8b",
    model="gemini-2.0-flash-exp",
    temperature=0.1,
    top_p=0.95,
    top_k=40,
    max_output_tokens=8192,
)

memory = MemorySaver()

search = TavilySearchResults(max_results=5)

@tool
def get_date_and_time_context(input: str) -> str:
    """Get current date and time information."""
    return f"Current time context: {datetime.now().strftime('%A, %B %d, %Y %I:%M %p')}"

tools = [search,get_date_and_time_context]

system_message = SystemMessage(content='''You are bai, brilliant pala's ai chatbot to help students studying for jee/neet to clear their doubts.
Only answer academic doubts, related to jee or neet and avoid any other question by simply guiding 
them to use it for academic purposes only.''')

langgraph_agent_executor = create_react_agent(
    model, tools, state_modifier=system_message, checkpointer=memory
)

agent = create_tool_calling_agent(model, tools, system_message)
agent_executor = AgentExecutor(agent=agent, tools=tools)
