import os
from langchain.agents import create_agent
from dotenv import load_dotenv
from langfuse import Langfuse, get_client

from langfuse.langchain import CallbackHandler
from langchain_openai import OpenAIEmbeddings
from db.milvus_handler import MilvusHandler

load_dotenv()

# === Init Langfuse ===

langfuse = get_client()
# langfuse_client = 
langfuse_handler = CallbackHandler()

# === Init Milvus ===
milvus = MilvusHandler()

# EMBEDING / AI
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("CHATBOT_MODEL_GENERATIVE")
embedding_model = os.getenv("CHATBOT_MODEL_EMBEDDING")
embeddings = OpenAIEmbeddings(model=embedding_model)

# === Init Agent ===
agent = create_agent(
    model=model,
    system_prompt="Eşti un sistem profesionist de avocatură care vorbeşte limba română şi este specializat în Codul Penal şi legislaţia română.",
)

def get_chatbot_answer(query: str, articles: list):
    """
    Generates a concise, professional answer in Romanian to a legal query using provided articles.

    Args:
        query (str): The user's legal question.
        articles (list): List of relevant articles or articles to use for context.

    Returns:
        str: The chatbot's answer in Romanian.
    """
    context = str(articles)
    completion = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"""
                    --IDENTITY: A romanian lawyer that answers short and concise.
                    --TASK: You are going to answer the USER_QUERY by using CONTEXT.
                    --USER_QUERY: {query}
                    --CONTEXT: {context}
                """,
                }
            ]
        },
        config={"callbacks": [langfuse_handler]},
    )
    return completion["messages"][-1].content


def get_embedding(text: str):
    """
    Generates an embedding vector for the given text using OpenAI's embedding model.

    Args:
        text (str): The input text to embed.
        model (str, optional): The embedding model to use. Defaults to "text-embedding-3-large".

    Returns:
        list: The embedding vector for the input text.
    """
    
    # Create a new trace for embedding computation
    # Start observation with specific type
    with langfuse.start_as_current_observation(
        as_type="embedding",
        name="embedding-generation"
    ) as obs:
        try:
            vector = embeddings.embed_query(text)
            obs.update(output=vector)
            return vector
        except Exception as exc:
            print(f"Error generating embedding: {exc}")
            raise
        
    
          
      
def run_chatbot(question: str):
    """
    Generates and prints an answer to a legal question using embeddings and a chatbot.

    Args:
        question (str): The legal question to be answered.
    """
    embedding = get_embedding(question)
    context = milvus.search(embedding, top_k=2)
    print(context)
    answer_from_gpt = get_chatbot_answer(question, context)
    print(answer_from_gpt)
    return answer_from_gpt, context


run_chatbot("ce se intampla daca fur")
