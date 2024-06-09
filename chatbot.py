import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# EMBEDING / AI

client = OpenAI()


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


with open("codPenal.txt", "r", encoding='utf-8') as file:
    content = file.read()


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


print(parse_legal_articles(content))
