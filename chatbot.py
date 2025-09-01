import json
import os
import re
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from milvus_handler import MilvusHandler 

load_dotenv()


# === Init Milvus ===
milvus = MilvusHandler() 

# EMBEDING / AI
api_key = os.getenv("OPENAI_API_KEY")

# Initialize client correctly (must be keyword arg)
clientOpenAI = OpenAI(api_key=api_key)


def get_chatbot_answer(query, articles):
    """
    Generates a concise, professional answer in Romanian to a legal query using provided articles.

    Args:
        query (str): The user's legal question.
        articles (list): List of relevant articles or articles to use for context.

    Returns:
        str: The chatbot's answer in Romanian.
    """
    answers_string=str(articles)
    completion = clientOpenAI.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You are a professionnal lawyer system that speaks Romanian, and specializes in Romanian Penal Code and law."},
            {"role": "user", "content": f"You are going to answer {query} in Romanian, by using {answers_string} and only this, as professionnal as you can and also short and concise. "}
        ]
    )
    print(completion.choices[0].message.content)
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


# Usage
# query = "Ce înseamnă «infracțiune» în Codul penal?"
# query_embedding = get_embedding(query)
# articles = get_articles_milvus(query_embedding, top_k=2)

# print(json.dumps(articles, ensure_ascii=False, indent=2))


# USE CASE FOR LOOP EMBEDDING QUESTIONS FOR TESTING 
intrebari_test = []
df = pd.read_csv("questions_penal_code_ro.csv")
question_and_context = df.to_json(orient="records")

for ix, row in enumerate(df.values):

    intrebari_test.append(row[0])
    
    if ix == 0:
        break
        
print(intrebari_test)


contextsFound = [] 

def get_answer_question(question: str):
    """
    Generates and prints an answer to a legal question using embeddings and a chatbot.

    Args:
        question (str): The legal question to be answered.
    """
    embedding = get_embedding(question, model="text-embedding-3-large")
    article_answers = milvus.search(embedding, top_k=2)
    answer_from_gpt = get_chatbot_answer(question, article_answers)
    print(article_answers)
    print(answer_from_gpt)


for intrebare in intrebari_test:
    get_answer_question(intrebare)




# print(get_chatbot_answer(query, articles))



