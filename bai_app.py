from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_react_agent  # Updated import
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
from typing import Any, Dict
load_dotenv()

# 1. Enhanced System Instruction & Chat History
system_instruction = """You are Bai, Pala's AI assistant for JEE/NEET students. Follow these rules:
1. Only answer academic questions related to JEE/NEET
2. Use tools for calculations, web search, and knowledge retrieval
3. Always respond in markdown with clear explanations
4. Always use web search whenever you feel uncertain"""

message_history = ChatMessageHistory()

# 2. Optimized Gemini Configuration
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.3,
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./db",
    collection_name="jee_neet_knowledge",
)

# 3. Corrected Tool Definitions
@tool
def vector_retrieval(query: str) -> str:
    """Search JEE/NEET knowledge base. Returns content with sources."""
    docs = vectorstore.similarity_search(query, k=3)
    content = "\n".join([doc.page_content for doc in docs])
    sources = "\n".join([f"- {doc.metadata.get('source', '')}" for doc in docs])
    return f"{content}\n\nSources:\n{sources}"

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search web for academic JEE/NEET content. Returns structured results with sources."""
    from langchain_community.tools.tavily_search import TavilySearchResults
    tavily = TavilySearchResults(max_results=3)
    results = tavily.invoke({"query": f"{query}"})  
    
    return {
        "name": "web_search",
        "content": "\n".join([f"{i+1}. {res['content']}" for i, res in enumerate(results)]),
        "sources": [res["url"] for res in results]
    }
    
tools = [
    web_search,
    vector_retrieval,
]

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# Updated agent creation to use React agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

def format_response(response: dict) -> str:
    """Format response with proper markdown"""
    return response["output"].replace("Sources:", "\n\n**Sources:**")

def chat(prompt: str) -> dict:
    """Handle chat interaction with the agent."""
    response = agent_executor.invoke({
        "input": prompt,
        "chat_history": message_history.messages
    })
    message_history.add_user_message(prompt)
    message_history.add_ai_message(response["output"])
    return {"final_response": response["output"]}

if __name__ == "__main__":
    while True:
        try:
            user_input = input("\nStudent: ")
            if user_input.lower() in ["exit", "quit"]:
                break
                
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": message_history.messages
            })
            
            formatted = format_response(response)
            print(f"\nBai: {formatted}")
            
            message_history.add_user_message(user_input)
            message_history.add_ai_message(formatted)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            message_history.clear()
            vectorstore.persist()