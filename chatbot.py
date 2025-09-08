import os
from dotenv import load_dotenv
from openai import OpenAI
from milvus_handler import MilvusHandler
from intrebari_test import intrebari_test, write_in_csv
import numpy as np

load_dotenv()


# === Init Milvus ===
milvus = MilvusHandler()

# EMBEDING / AI
api_key = os.getenv("OPENAI_API_KEY")

# Initialize client correctly (must be keyword arg)
clientOpenAI = OpenAI(api_key=api_key)
model = "gpt-4o-mini"

def get_chatbot_answer(query, articles):
    """
    Generates a concise, professional answer in Romanian to a legal query using provided articles.

    Args:
        query (str): The user's legal question.
        articles (list): List of relevant articles or articles to use for context.

    Returns:
        str: The chatbot's answer in Romanian.
    """
    context=str(articles)
    completion = clientOpenAI.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Eşti un sistem profesionist de avocatură care vorbeşte limba română şi este specializat în Codul Penal şi legislaţia română."
            },
            {
                "role": "user", "content": f"""
                --IDENTITY: A romanian lawyer that answers short and concise.
                --TASK: You are going to answer the USER_QUERY by using CONTEXT.
                --USER_QUERY: {query}
                --CONTEXT: {context}
                """
            }
        ]
    )
    return completion.choices[0].message.content

def get_embedding(text, model="text-embedding-3-large"):
    """
    Generates an embedding vector for the given text using OpenAI's embedding model.

    Args:
        text (str): The input text to embed.
        model (str, optional): The embedding model to use. Defaults to "text-embedding-3-large".

    Returns:
        list: The embedding vector for the input text.
    """
    return clientOpenAI.embeddings.create(input=[text], model=model).data[0].embedding

def run_chatbot(question: str):
    """
    Generates and prints an answer to a legal question using embeddings and a chatbot.

    Args:
        question (str): The legal question to be answered.
    """
    embedding = get_embedding(question, model="text-embedding-3-large")
    context = milvus.search(embedding, top_k=2)
    answer_from_gpt = get_chatbot_answer(question, context)
    print(answer_from_gpt)
    return answer_from_gpt, context


full_ragas_dataset = []
# test_intrebari_test = intrebari_test[0:2]

# print(test_intrebari_test)
for _, intrebare_arr in intrebari_test:
    intrebare = intrebare_arr[0]
    answer, context = run_chatbot(intrebare)

    ragas_dataset_obj = {"question":intrebare, "context":context, "answer_gpt":answer}
    full_ragas_dataset.append(ragas_dataset_obj)

write_in_csv(str(full_ragas_dataset))




