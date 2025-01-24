from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ChatMessageHistory
from dotenv import load_dotenv
load_dotenv()

# 1. Custom System Instruction & Chat History Setup
system_instruction = """'You are bai, brilliant pala's ai chatbot to help students studying for jee/neet to clear their doubts.
Only answer academic doubts, related to jee or neet and avoid any other question by simply guiding 
them to use it for academic purposes only. you have the capabilities of
- Perform complex calculations
- Search web for latest information
- Retrieve context from jee/ neet database
Always respond in markdown format"""

message_history = ChatMessageHistory()

# 2. Initialize Google Gen AI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.3,
)

@tool
def vector_retrieval(query: str) -> str:
    """Searches knowledge base for technical documentation and internal resources."""
    results = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in results])

tools = [
    TavilySearchResults(max_results=3),
    vector_retrieval,
]

# 4. ChromaDB Vectorstore Setup with Persistence
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./db",
    collection_name="knowledge_base"
)

# Function to populate vectorstore
def embed_txt_to_chroma(file_path: str):
    with open(file_path, "r") as f:
        text = f.read()
    
    # Simple text splitter for demonstration
    texts = [text[i:i+1000] for i in range(0, len(text), 1000)]
    vectorstore.add_texts(texts)
    vectorstore.persist()

# 5. Agent Setup with Memory
prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def chat(prompt: str) -> dict:
    """Function to handle chat interaction with the agent."""
    response = agent_executor.invoke({
        "input": prompt,
        "chat_history": message_history.messages
    })
    message_history.add_user_message(prompt)
    message_history.add_ai_message(response["output"])
    return {"final_response": response["output"]}

if __name__ == "__main__":
    # embed_txt_to_chroma("dnf.txt")
    
    while True:
        user_input = input("Enter your question: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": message_history.messages
        })
        message_history.add_user_message(user_input)
        message_history.add_ai_message(response["output"])
        
        print(response["output"])
