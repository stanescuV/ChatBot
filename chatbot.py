import json
import os
import re
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient, connections, DataType, Collection

load_dotenv()

MILVUS_IP = os.getenv("MILVUS_IP")
#MILVUS
#note add a .env IP variable !! 
clientMilvus = MilvusClient(
    uri="http://localhost:19530"
    # uri = f"http://{MILVUS_IP}:19530"
)

#DB
conn = connections.connect(host="localhost", port=19530)
# database = db.create_database("CodPenal")

#SCHEMA
schema = MilvusClient.create_schema(
    auto_id=True,
    enable_dynamic_field=True
)

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=3072)

# clientMilvus.drop_collection(collection_name="codPenal_collection")
# clientMilvus.drop_collection(collection_name="codPenal_collectionTest")
#

##COLLECTION
index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
    "metric_type": "COSINE"
}

collection = Collection(name="codPenal_collection", schema=schema)
collection.create_index("embedding", index_params)
collection.load()
# executed only once
# clientMilvus.create_collection(
#     collection_name="codPenal_collection",
#     dimension=3072,
#     schema=schema,
#     metric_type="COSINE"
# )
db_name = "codPenal_collection"

# clientMilvus.create_collection(
#     collection_name="codPenal_collectionTest",
#     dimension=3072,
#     schema=schema,
#     metric_type="COSINE"
# )
# db_name_test = "codPenal_collectionTest"



def insert_data(collectionName, data):
    """
    Inserts data into the specified Milvus collection.

    Args:
        collectionName (str): The name of the Milvus collection.
        data (list): A list of data entries to insert.

    Returns:
        None
    """
    res = clientMilvus.insert(
        collection_name=collectionName,
        data=data,
    )
    print(res)

# EMBEDING / AI
# Get the key from environment
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

def get_articles_milvus(query_embedding, top_k=2):
    """
    Searches the Milvus collection for articles most similar to the query embedding.

    Args:
        query_embedding (list): The embedding vector of the query.
        top_k (int, optional): Number of top results to return. Defaults to 2.

    Returns:
        list: List of dictionaries containing article id, score, and text.
    """
    res = clientMilvus.search(
        collection_name=db_name,
        data=[query_embedding],
        limit=top_k,
        search_params={"metric_type": "COSINE", "params": {}},
        output_fields=["text"]
    )

    # res is usually a list of result sets (one per query vector)
    hits = res[0]

    results = []
    for h in hits:
        # Handle both Hit API shapes
        try:
            # Classic Hit object from Collection.search
            item = {
                "id": int(getattr(h, "id", None) or getattr(h, "pk", None) or -1),
                "score": float(getattr(h, "distance", None) or getattr(h, "score", None)),
                "text": h.get("text") if hasattr(h, "get") else (h["text"] if isinstance(h, dict) else None),
            }
        except Exception:
            # MilvusClient may already return dict-like rows
            item = {
                "id": int(h.get("id", -1)) if isinstance(h, dict) else -1,
                "score": float(h.get("distance", h.get("score", 0.0))) if isinstance(h, dict) else 0.0,
                "text": (h.get("text") if isinstance(h, dict) else None),
            }
        results.append(item)

    return results

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
    
    if ix > 0:
        break
        
print(intrebari_test)


contextsFound = [] 

def get_answer_question(intrebare):
    embedding = get_embedding(intrebare, model="text-embedding-3-large")
    article_answers = get_articles_milvus(embedding, top_k=2)
    answer_from_gpt = get_chatbot_answer(intrebare, article_answers)
    print(article_answers)
    print(answer_from_gpt)


for intrebare in intrebari_test:
    get_answer_question(intrebare)




# get_answer_csv_questions(intrebari_test) 
# print(contextsFound)

# def get_answer_csv_questions(intrebari):
#     for ix, intrebare in enumerate(intrebari): 
#         embedding = get_embedding(intrebare, model="text-embedding-3-large")
#         article_answers = get_articles_milvus(embedding, top_k=2)
        
#         for answer in article_answers:
#             contextsFound.append({'indexQuestion': ix, 'context': answer['text']})

# get_answer_csv_questions(intrebari_test) 
# print(contextsFound)
# previousIx = 123
# for context in contextsFound:
#     if previousIx != context["indexQuestion"]:


    



# print(get_chatbot_answer(query, articles))






# with open("codPenal.txt", "r", encoding='utf-8') as file:
#     codPenal = file.read()


#From big String to an array of articles

# def parse_legal_articles(text):
#     # Capture the whole header (Art. 238.) AND the number (238)
#     pattern = re.compile(r'(Art\.\s*(\d+)\.)')

#     # With capturing groups, split() returns:
#     # [pre, header1, number1, content1, header2, number2, content2, ...]
#     parts = pattern.split(text)

#     articles = []
#     skip_keywords = ("CAP", "SEC", "TIT")

#     # Walk the array in chunks of 3: header, number, content
#     for i in range(1, len(parts) - 1, 3):
#         header = parts[i].strip()              # e.g., "Art. 238."
#         number_str = parts[i + 1].strip()      # e.g., "238"
#         content = parts[i + 2]                 # text after the header

#         # Stop content at the next structural keyword (if any)
#         for kw in skip_keywords:
#             pos = content.find(kw)
#             if pos != -1:
#                 content = content[:pos]
#                 break

#         # Normalize whitespace
#         content = re.sub(r'\s+', ' ', content).strip()

#         # Extract title (ArtName) = text before first period in the content
#         if '.' in content:
#             art_name, art_text = content.split('.', 1)
#             art_name = art_name.strip()
#             art_text = art_text.strip()
#         else:
#             art_name = content.strip()
#             art_text = ""

#         # Convert number to int when possible
#         try:
#             art_number = int(number_str)
#         except ValueError:
#             art_number = number_str  # fallback to string if unexpected format

#         articles.append({
#             'ArtNumber': art_number,
#             'ArtName': art_name,
#             'ArtText': art_text
#         })


#     return articles


# # INSERT MILVUS

#SET UP FOR PROD
# embeddings = []
# articles = parse_legal_articles(codPenal)

# for ix, article in enumerate(articles) :
#     embeddings.append({ "text" : str(article), "embedding" : get_embedding(str(article))})

# print(len(embeddings))

# insert_data(db_name, embeddings)


#SET UP FOR TEST
# embeddingsTest = []
# articlesTest = [articles[0],articles[1]]

# for ix, article in enumerate(articlesTest):
#     embeddingsTest.append({ "text" : str(article), "embedding" : get_embedding(str(article))})

# print(len(embeddingsTest))

# insert_data(db_name_test, embeddingsTest)




