from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
load_dotenv()

# 1. Enhanced System Instruction & Chat History
system_instruction = """You are Bai, Pala's AI assistant for JEE/NEET students. Follow these rules:
1. Only answer academic questions related to JEE/NEET
2. Use tools for calculations, web search, and knowledge retrieval
3. Always respond in markdown with clear explanations
4. ALways use web search whenever u feel uncertain """

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

# 3. Improved Tool Definitions
@tool
def vector_retrieval(query: str) -> dict:  # Return structured data
    """Search JEE/NEET knowledge base. Returns dict with 'content' and 'sources'."""
    docs = vectorstore.similarity_search(query, k=3)
    return {
        "name": "vector_retrieval",
        "content": "\n".join([doc.page_content for doc in docs]),
        "sources": [doc.metadata.get("source", "") for doc in docs]
    }

tools = [
    TavilySearchResults(max_results=3, name="web_search"),
    vector_retrieval,
]

# 4. Enhanced Vectorstore Setup


prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad", optional=True),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

def format_response(response: dict) -> str:
    """Add source attribution and formatting"""
    output = response["output"]
    if "sources" in response:
        output += "\n\n**Sources:**\n" + "\n".join(
            [f"[^{i+1}] {src}" for i, src in enumerate(response["sources"])]
        )
    return output

def chat_loop():
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

if __name__ == "__main__":
    chat_loop()