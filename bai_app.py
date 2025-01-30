from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.format_scratchpad import format_log_to_str
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
from typing import Any, Dict
load_dotenv()

system_instruction = """You are Bai, Pala's AI assistant for JEE/NEET students. Follow these rules:
1. Only answer academic questions related to JEE/NEET
2. Use tools for calculations, web search, and knowledge retrieval
3. Always respond in markdown with clear explanations
4. Always use web search whenever you feel uncertain

Available Tools:
{tools}

Tool Names (use exactly these when needed): {tool_names}

Use the following format:
Question: the input question
Thought: your reasoning steps
Action: the tool name (one of [{tool_names}])
Action Input: the tool input
Observation: the tool result
... (repeat until final answer)
Thought: I now know the final answer
Final Answer: the final response

Begin!"""

message_history = ChatMessageHistory()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.3,
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./db",
    # collection_name="jee_neet_knowledge",
)

@tool
def vector_retrieval(query: str) -> str:
    """Search JEE/NEET database on d and f block elements."""
    docs = vectorstore.similarity_search(query, k=5)
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
    
tools = [web_search, vector_retrieval]
# tools = [web_search]
tool_names = ", ".join([t.name for t in tools])

react_template = system_instruction + "\n\nQuestion: {input}\nThought:{agent_scratchpad}"
prompt = PromptTemplate.from_template(react_template)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
    format_intermediate_steps=format_log_to_str
)

def format_response(response: dict) -> str:
    """Format response with proper markdown"""
    return response["output"].replace("Sources:", "\n\n**Sources:**")

def chat(prompt: str) -> dict:
    """Handle chat interaction with the agent."""
    response = agent_executor.invoke({
        "input": prompt,
        "tools": "\n".join([f"{t.name}: {t.description}" for t in tools]),
        "tool_names": tool_names,
        "agent_scratchpad": ""
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
                "tools": "\n".join([f"{t.name}: {t.description}" for t in tools]),
                "tool_names": tool_names,
                "agent_scratchpad": ""
            })
            
            formatted = format_response(response)
            print(f"\nBai: {formatted}")
            
            message_history.add_user_message(user_input)
            message_history.add_ai_message(formatted)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            message_history.clear()
            vectorstore.persist()