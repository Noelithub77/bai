from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_core.prompts import ChatPromptTemplate
import os
import logging
from datetime import datetime
from vdb_management import vdb


# def set_api_key(key):
#     """Set the Google API key"""
#     os.environ["GOOGLE_API_KEY"] = key

os.environ["GOOGLE_API_KEY"] = "AIzaSyAUwMc3lBZdf73j1TSvRJHkD_nDKfzesKc"


# Initialize logging
logging.basicConfig(filename='chatbot.log', level=logging.INFO)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    # model="gemini-1.5-pro",
    # model="gemini-1.5-flash-8b",
    model="gemini-2.0-flash-exp",
    temperature=0.1,
    top_p=0.95,
    top_k=40,
    max_output_tokens=8192,
)


vdb_obj = vdb(persist_directory="db")


# Don't explain this
def load_context_file(filename):
    try:
        with open(f"./data/{filename}", "r") as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error loading {filename}: {e}")
        return ""

# Initialize memory for retrieval chain
chatHistory = ConversationBufferMemory()

# Define conversation template using ChatPromptTemplate
system_template = """You are bai, brilliant pala's ai chatbot to help students studying for jee/neet to clear their doubts.
Only answer academic doubts, related to jee or neet and avoid any other question by simply guiding 
them to use it for academic purposes only.
Context: {context}"""

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{question}")]
)

retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vdb_obj.get_retriever(),
    memory=chatHistory,
    get_chat_history=lambda h: str(h),
    combine_docs_chain_kwargs={"prompt": prompt_template, "document_variable_name": "context"},
    return_source_documents=True,
    verbose=True
)



def get_retrieval_response(query):
    """Tool function to get response from retrieval chain"""
    response = retrieval_chain({"question": query})
    return f"Knowledge Base Response: {response['answer']}\nSources: {[doc.metadata.get('source', 'Unknown') for doc in response['source_documents']]}"

# Updated tools list with retrieval tool
# Don't Explain tools in detail
tools = [
    # Tool(
    #     name="Knowledge Base Search",
    #     func=get_retrieval_response,
    #     description="Use this tool to search the knowledge base for specific information about IIIT Kottayam"
    # ),
    Tool(
        name="Load Mess Menu",
        func=lambda x: f"Mess menu context: {load_context_file('mess_menu.txt')}",
        description="Load mess menu context for food-related queries",
    ),
    Tool(
        name="Load Academic Calendar",
        func=lambda x: f"Academic calendar context: {load_context_file('inst_calender.txt')}",
        description="Load academic calendar context for academic dates and holidays",
    ),
    Tool(
        name="Get Date and Time Context",
        func=lambda x: f"Current time context: {datetime.now().strftime('%A, %B %d, %Y %I:%M %p')}",
        description="Get current date and time information",
    ),
    Tool(
        name="Load Curriculum",
        func=lambda x: f"Curriculum context: {load_context_file('caricululm.txt')}",
        description="Load curriculum context for course-related queries",
    ),
    Tool(
        name="Load Milma Menu",
        func=lambda x: f"Milma Cafe menu context: {load_context_file('milma_menu.txt')}",
        description="Load Milma Cafe menu context for food-related queries",
    ),
    Tool(
        name="Load Faculty Details",
        func=lambda x: f"Faculty details context: {load_context_file('faculty_details.txt')}",
        description="Load faculty details context for faculty-related queries",
    )
]

# agent is a _____ , that takes in tools,the gemini llm,agent(type of agent) 
# 
agent = initialize_agent(
    tools, 
    llm, 
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)


# Start here;
def chat(user_input):
    try:
        logging.info(f"User Input: {user_input}")
        
        # Get agent response with memory 
        # agent get input from here
        print(chatHistory.buffer)
        agent_response = agent({"input": user_input})
        
        return {
            "final_response": agent_response["output"],
            "full_log": {
                "thoughts": agent_response["intermediate_steps"],
                # "chat_history": chatHistory.chat_memory.messages,
                "context_response": "",  # Add empty context_response to maintain structure
                "tool_response": {"thoughts": agent_response["intermediate_steps"]}
            }
        }
    except Exception as e:
        logging.error(f"Error in chat: {str(e)}")
        return {"final_response": f"I encountered an error: {str(e)}"}

def initialize_bot():
    """Initialize the chatbot and return the chat function"""
    # Clear both memories when initializing
    return chat


if __name__ == "__main__":
    # os.environ["GOOGLE_API_KEY"] = "AIzaSyAYew4okjx4jmR7xbKhLj2mAckgtUUbR-k"
    while True:
        user_input = input("\nEnter your query (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        response = chat(user_input)
        print("Assistant:", response["final_response"])
