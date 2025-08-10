import json
import os
import re
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
conn = connections.connect(host=f"localhost", port=19530)
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



def insertData(collectionName, data):
    res = clientMilvus.insert(
        collection_name=collectionName,
        data=data,
    )
    print(res)

# EMBEDING / AI
clientOpenAI = OpenAI(api_key="")


def get_chatbot_answer(query, answers):
    answersString=str(answers)
    completion = clientOpenAI.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You are a professionnal lawyer system that speaks Romanian, and specializes in Romanian Penal Code and law."},
            {"role": "user", "content": f"You are going to answer {query} in Romanian, by using {answersString} and only this, as professionnal as you can and also short and concise. "}
        ]
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

def get_embedding(text, model="text-embedding-3-large"):
    return clientOpenAI.embeddings.create(input=[text], model=model).data[0].embedding

def get_articles_milvus(query_embedding, top_k=2):
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
query = "tâlhărie"
query_embedding = get_embedding(query)
articles = get_articles_milvus(query_embedding, top_k=2)

print(json.dumps(articles, ensure_ascii=False, indent=2))

# print(get_chatbot_answer(query, articles))





# with open("codPenal.txt", "r", encoding='utf-8') as file:
#     codPenal = file.read()


#From big String to an array of articles

# def parse_legal_articles(text):
#     # Define the pattern
#     pattern = re.compile(r'(Art\.\s\d+\.)')

#     # Split the text at each article start
#     parts = pattern.split(text)[1:]  # [1:] to skip the first empty result

#     articles = []
#     skip_keywords = ["CAP", "SEC", "TIT"]

#     for i in range(0, len(parts), 2):

#         article_content = parts[i + 1]

#         for keyword in skip_keywords:
#             if keyword in article_content:
#                 keyword_position = article_content.find(keyword)
#                 article_content = article_content[:keyword_position].strip()
#                 break

#         article_content = article_content.replace('\n', ' ')

#         article_name = article_content.split('.')[0].strip()
#         article_text = article_content.split('.', 1)[1].strip() if '.' in article_content else ''

#         articles.append({
#             'ArtName': article_name,
#             'ArtText': article_text
#         })

#     return articles



# # INSERT MILVUS

#SET UP FOR PROD
# embeddings = []
# articles = parse_legal_articles(codPenal)

# for ix, article in enumerate(articles) :
#     embeddings.append({ "text" : str(article), "embedding" : get_embedding(str(article))})

# print(len(embeddings))

# insertData(db_name, embeddings)


#SET UP FOR TEST
# embeddingsTest = []
# articlesTest = [articles[0],articles[1]]

# for ix, article in enumerate(articlesTest):
#     embeddingsTest.append({ "text" : str(article), "embedding" : get_embedding(str(article))})

# print(len(embeddingsTest))

# insertData(db_name_test, embeddingsTest)




