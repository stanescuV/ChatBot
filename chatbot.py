import os
import re
from openai import OpenAI
from pymilvus import MilvusClient


#MILVUS
client = MilvusClient(
    uri="http://localhost:19530"
)

# executed only once
client.create_collection(
    collection_name="codPenal_collection",
    dimension=3072,
    metric_type="COSINE"
)
db_name = "codPenal_collection"

def insertData(collectionName, data):
    res = client.insert(
        collection_name=collectionName,
        data=data,
    )
    print(res)

# EMBEDING / AI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-3-large"):
    return client.embeddings.create(input=[text], model=model).data[0].embedding


with open("codPenal.txt", "r", encoding='utf-8') as file:
    codPenal = file.read()


#From big String to an array of articles

def parse_legal_articles(text):
    # Define the pattern
    pattern = re.compile(r'(Art\.\s\d+\.)')

    # Split the text at each article start
    parts = pattern.split(text)[1:]  # [1:] to skip the first empty result

    articles = []
    skip_keywords = ["CAP", "SEC", "TIT"]

    for i in range(0, len(parts), 2):

        article_content = parts[i + 1]

        for keyword in skip_keywords:
            if keyword in article_content:
                keyword_position = article_content.find(keyword)
                article_content = article_content[:keyword_position].strip()
                break

        article_content = article_content.replace('\n', ' ')

        article_name = article_content.split('.')[0].strip()
        article_text = article_content.split('.', 1)[1].strip() if '.' in article_content else ''

        articles.append({
            'ArtName': article_name,
            'ArtText': article_text
        })

    return articles

articles = parse_legal_articles(codPenal)

embeddings = []
articlesTest = [articles[0],articles[1],articles[2]]

# OPEN AI EMBEDDING

for ix, article in enumerate(articles) :
    embeddings.append({"id" : ix, "text" : article, "embedding" : get_embedding(str(article))})

print(len(embeddings))

insertData(db_name, embeddings)




