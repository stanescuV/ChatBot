import os
from typing import TypedDict, List, Optional # Standard library

# Third-party imports
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.chat_models import init_chat_model
from langgraph.config import get_stream_writer
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)
from langchain_openai import OpenAIEmbeddings
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.graph import StateGraph, START, END

# First-party (local) imports
from db.milvus_handler import MilvusHandler
# from messages_class import MessagesState # We define our own state below

load_dotenv()

# === Init Langfuse ===
langfuse = get_client()
langfuse_handler = CallbackHandler()

# === Init Milvus ===
milvus = MilvusHandler()


# === EMBEDING / AI ===
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("CHATBOT_MODEL_GENERATIVE")
embedding_model = os.getenv("CHATBOT_MODEL_EMBEDDING")
embeddings = OpenAIEmbeddings(model=embedding_model)

# === Init Agent ===
agent = init_chat_model(
    model=model,
)

# === Define the Graph State ===
# This replaces MessagesState. It's a "blueprint" for the data
# that flows through your graph.
class GraphState(TypedDict):
    messages: List[BaseMessage]  
    query: str
    embedding: Optional[List[float]]
    context: Optional[List[str]] 
    first: Optional[str]
    second: Optional[str]
    third: Optional[str]
    fourth: Optional[str]


# === Node Functions ===
# Each function is a "node" in our graph.
# It takes the current state and returns a dictionary to update the state.

def extract_query_node(state: GraphState):
    """
    Extracts the most recent HumanMessage as the query.
    """
    # Get the last message, which is the user's query
    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        raise ValueError("Last message is not a HumanMessage.")
    
    writer = get_stream_writer()
    writer(f"--- 1. Query extracted: {last_message.content} ---")
    return {"query": last_message.content}

def get_embedding_node(state: GraphState):
    """
    Generates an embedding for the user's query.
    """
    query = state["query"]
    print(f"--- 2. 🔎 Generating embedding for: {query} ---")
    
    with langfuse.start_as_current_observation(
        as_type="embedding",
        name="embedding-generation"
    ) as obs:
        try:
            if not isinstance(query, str) or len(query) == 0:
                raise ValueError(f"Input text for embedding is empty or invalid: {query}")
            
            vector = embeddings.embed_query(query)
            obs.update(output=vector)
            return {"embedding": vector}
        except Exception as exc:
            print(f"Error generating embedding: {exc}")
            raise

def get_context_node(state: GraphState):
    """
    Retrieves context from Milvus using the embedding.
    """
    embedding = state["embedding"]
    if embedding is None:
        raise ValueError("Embedding not found in state.")
        
    print("--- 3. 📚 Retrieving context from Milvus ---")
    context = milvus.search(embedding, top_k=2)
    print(f"--- Context found: {context} ---")
    return {"context": context}

def get_chatbot_answer_node(state: GraphState):
    """
    Generates the final answer using the query and context.
    """
    query = state["query"]
    context = state["context"]
    
    if context is None:
        raise ValueError("Context not found in state.")

    print("--- 4. 🤖 Generating final answer ---")
    
    context_str = str(context)
    # Get the system message from the state
    system_prompt = state["messages"][0]
    if not isinstance(system_prompt, SystemMessage):
         raise ValueError("First message is not a SystemMessage.")
    
    # Construct the prompt for the LLM
    final_prompt = f"""
        --IDENTITY: A romanian lawyer that answers short and concise.
        --TASK: You are going to answer the USER_QUERY by using CONTEXT.
        --USER_QUERY: {query}
        --CONTEXT: {context_str}
    """
    
    # We pass the system message and our constructed user prompt
    completion = agent.invoke(
        [
            system_prompt,
            HumanMessage(content=final_prompt)
        ],
        config={"callbacks": [langfuse_handler]},
    )
    
    # 'completion' is an AIMessage object
    answer_message = AIMessage(content=completion.content)
    
    print(f"--- Answer: {answer_message.content} ---")
    
    # Add the AI's answer to the list of messages
    return {"messages": state["messages"] + [answer_message]}

# === Build the Graph ===

print("=== 🛠️ Compiling Graph ===")

agent_builder = StateGraph(GraphState)

# 1. Add nodes
agent_builder.add_node("extract_query", extract_query_node)
agent_builder.add_node("get_embedding", get_embedding_node)
agent_builder.add_node("get_context", get_context_node)
agent_builder.add_node("get_chatbot_answer", get_chatbot_answer_node)

# 2. Add edges
agent_builder.add_edge(START, "extract_query")
agent_builder.add_edge("extract_query", "get_embedding")
agent_builder.add_edge("extract_query", "get_embedding")
agent_builder.add_edge("get_embedding", "get_context")
agent_builder.add_edge("get_context", "get_chatbot_answer")
agent_builder.add_edge("get_chatbot_answer", END)

# 3. Compile
agent_graph = agent_builder.compile()

print("=== ✅ Graph Compiled ===")

# Show the agent graph
try:
    display(Image(agent_graph.get_graph(xray=True).draw_mermaid_png()))
except Exception as e:
    print(f"Could not display graph: {e}. (This often happens if 'pygraphviz' is not installed.)")


# === Example usage ===

# Define your initial messages
system_message = SystemMessage(content="Eşti un sistem profesionist de avocatură care vorbeşte limba română şi este specializat în Codul Penal şi legislaţia română.")
user_query = HumanMessage(content="Ce se intampla daca fur ?")

initial_messages = [system_message, user_query]

# Invoke the graph
# The input is a dictionary matching the GraphState
# new_state = agent_graph.invoke({"messages": initial_messages})

graph_input = {"messages": initial_messages}

# TODO: SSE server side events 
for chunk in agent_graph.stream(graph_input, stream_mode="custom"):
    print(chunk)
    
print("\n--- 🏁 Final State ---")
# for m in new_state["messages"]:
#     m.pretty_print()