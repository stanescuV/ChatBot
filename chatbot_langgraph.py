import os
import operator
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain.messages import SystemMessage, HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from db.milvus_handler import MilvusHandler
from constants import system_promt

load_dotenv()

# === Langfuse ===
langfuse = get_client()
langfuse_handler = CallbackHandler()

# === Milvus ===
milvus = MilvusHandler()

# === Models ===
model_str = os.getenv("CHATBOT_MODEL_GENERATIVE")
embedding_model = os.getenv("CHATBOT_MODEL_EMBEDDING")

llm = init_chat_model(model=model_str, streaming=True)
embeddings = OpenAIEmbeddings(model=embedding_model)




# === State ===
class ChatbotState(TypedDict):
    question: str
    embedding: list
    context: list
    messages: Annotated[list[AnyMessage], operator.add]


# === Nodes ===
def embed_query(state: ChatbotState) -> dict:
    with langfuse.start_as_current_observation(
        as_type="embedding", name="embedding-generation"
    ) as obs:
        try:
            if not isinstance(state["question"], str) or not state["question"]:
                raise ValueError("Input text for embedding is empty.")
            vector = embeddings.embed_query(state["question"])
            obs.update(output=vector)
            return {"embedding": vector}
        except Exception as exc:
            print(f"Error generating embedding: {exc}")
            raise


def retrieve_context(state: ChatbotState) -> dict:
    context = milvus.search(state["embedding"], top_k=2)
    return {"context": context}


async def generate_answer(state: ChatbotState) -> dict:
    context = str(state["context"])
    prompt = (
        f"--IDENTITY: A romanian lawyer that answers short and concise.\n"
        f"--TASK: You are going to answer the USER_QUERY by using CONTEXT.\n"
        f"--USER_QUERY: {state['question']}\n"
        f"--CONTEXT: {context}"
    )
    messages = [
        SystemMessage(content=system_promt),
        HumanMessage(content=prompt),
    ]
    response = await llm.ainvoke(messages, config={"callbacks": [langfuse_handler]})
    return {"messages": [response]}


# === Graph ===
_builder = StateGraph(ChatbotState)
_builder.add_node("embed_query", embed_query)
_builder.add_node("retrieve_context", retrieve_context)
_builder.add_node("generate_answer", generate_answer)
_builder.add_edge(START, "embed_query")
_builder.add_edge("embed_query", "retrieve_context")
_builder.add_edge("retrieve_context", "generate_answer")
_builder.add_edge("generate_answer", END)

chatbot_graph = _builder.compile()


# === Public API ===
def get_embedding(text: str) -> list:
    """Generate an embedding vector for the given text."""
    with langfuse.start_as_current_observation(
        as_type="embedding", name="embedding-generation"
    ) as obs:
        try:
            if not isinstance(text, str) or not text:
                raise ValueError("Input text for embedding is empty. text is " + str(text))
            vector = embeddings.embed_query(text)
            obs.update(output=vector)
            return vector
        except Exception as exc:
            print(f"Error generating embedding: {exc}")
            raise


async def run_chatbot(question: str):
    """
    Run the LangGraph RAG pipeline and stream the answer token by token.

    Yields:
        str: tokens from the LLM response
    """
    initial_state: ChatbotState = {
        "question": question,
        "embedding": [],
        "context": [],
        "messages": [],
    }
    async for event in chatbot_graph.astream_events(initial_state, version="v2"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                yield chunk.content
