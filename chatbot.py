import os
import re
from openai import OpenAI
from pymilvus import MilvusClient,connections, db


#MILVUS
clientMilvus = MilvusClient(
    # uri="http://localhost:19530"
    uri = "http://49.12.46.230:19530"
)

#DB
# conn = connections.connect(host="49.12.46.230", port=19530)
# database = db.create_database("CodPenal")

# executed only once
clientMilvus.create_collection(
    collection_name="codPenal_collection",
    dimension=3072,
    metric_type="COSINE"
)
db_name = "codPenal_collection"

clientMilvus.create_collection(
    collection_name="codPenal_collectionTest",
    dimension=3072,
    metric_type="COSINE"
)
db_name_test = "codPenal_collectionTest"

def insertData(collectionName, data):
    res = clientMilvus.insert(
        collection_name=collectionName,
        data=data,
    )
    print(res)

# EMBEDING / AI
clientOpenAI = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-3-large"):
    return clientOpenAI.embeddings.create(input=[text], model=model).data[0].embedding


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



# OPEN AI EMBEDDING

#SET UP FOR PROD
# embeddings = []
articles = parse_legal_articles(codPenal)

# for ix, article in enumerate(articles) :
#     embeddings.append({"id" : ix, "text" : article, "embedding" : get_embedding(str(article))})

# print(len(embeddings))

# insertData(db_name, embeddings)


#SET UP FOR TEST
embeddingsTest = []
articlesTest = [articles[0],articles[1]]

for ix, article in enumerate(articlesTest):
    embeddingsTest.append({"id" : ix, "text" : article, "embedding" : get_embedding(str(article))})

print(len(embeddingsTest))

insertData(db_name_test, embeddingsTest)





