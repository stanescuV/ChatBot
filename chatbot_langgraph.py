import os
import asyncio
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

GUARD_RAIL_PROMPT = (
    "You are a strict classifier. Answer ONLY with 'VALID' or 'INVALID'.\n"
    "A question is VALID if and only if it can be answered using the Romanian Penal Code (Codul Penal al României).\n"
    "A question is INVALID if it is off-topic, unrelated to Romanian criminal law, or cannot be answered with the Romanian Penal Code.\n"
    "User question: {question}\n"
    "Answer:"
)

REJECTION_MESSAGE = "Această întrebare nu poate fi răspunsă cu ajutorul Codului Penal Român."

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


# === Guard Rail (runs outside the graph as a plain async task) ===
async def _check_guard_rail(question: str) -> bool:
    prompt = GUARD_RAIL_PROMPT.format(question=question)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content.strip().upper() == "VALID"


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
        f"--TASK: You are going to answer the USER_QUERY by using CONTEXT. Always include the articles numbers in the answer.\n"
        f"--USER_QUERY: {state['question']}\n"
        f"--CONTEXT: {context}"
    )
    messages = [
        SystemMessage(content=system_promt),
        HumanMessage(content=prompt),
    ]
    response = None
    async for chunk in llm.astream(messages):
        response = chunk if response is None else response + chunk
    return {"messages": [response]}


# === Graph (linear pipeline — guard rail runs as a parallel asyncio task in run_chatbot) ===
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
    Run the RAG pipeline and stream the answer token by token.
    Guard rail runs as a concurrent asyncio task — if it returns invalid
    mid-stream, the pipeline is interrupted and the rejection message is sent.

    Yields:
        str: tokens from the LLM response, or the rejection message
    """
    #This is the only LLM function that runs outside of the graph
    guard_task = asyncio.create_task(_check_guard_rail(question))

    initial_state: ChatbotState = {
        "question": question,
        "embedding": [],
        "context": [],
        "messages": [],
    }
    config = {"callbacks": [langfuse_handler]}

    async for event in chatbot_graph.astream_events(initial_state, version="v2", config=config):
        # Between every token, check if guard rail has already returned invalid
        if guard_task.done() and not guard_task.result():
            break

        node = event.get("metadata", {}).get("langgraph_node")
        if event["event"] == "on_chat_model_stream" and node == "generate_answer":
            chunk = event["data"]["chunk"]
            if chunk.content:
                yield chunk.content

    # Pipeline done (or broken out of) — ensure guard rail result is available
    is_valid = await guard_task
    if not is_valid:
        yield REJECTION_MESSAGE
