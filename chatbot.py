import os
import json
import re
import numpy as np
from pymilvus import (connections,FieldSchema,CollectionSchema,DataType,Collection)
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
load_dotenv()



#Connection to DB Milvus
connections.connect(
  alias="default", 
  host='49.12.46.230', 
  port='19530'
)


# EMBEDING / AI 

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding



#PDF READER 

#From PDF to STRING

reader = PdfReader('CodPenalRo.pdf')
# print(len(reader.pages)) 
codPenal=""
for i in range(0, len(reader.pages)) :
  page = reader.pages[i]
  pageText = page.extract_text()
  codPenal+=pageText
 
#From big String to an array of articles 

def parse_legal_articles(text):
    # Define the pattern to split articles; this pattern looks for "Art. " followed by any number of digits and a period
    pattern = re.compile(r'(Art\.\s\d+\.)')
    
    # Split the text at each article start, keeping the delimiters (article numbers and titles)
    parts = pattern.split(text)[1:]  # [1:] to skip the first empty result from split
    
    # Reconstruct article splits into pairs of (title, content)
    articles = [{'ArtNumber': parts[i][4:].strip().replace(".", ""), 'ArtName': parts[i+1].split('.')[0].strip(), 'ArtText': parts[i+1].split('.', 1)[1].strip()} 
                for i in range(0, len(parts), 2)]
    
    return articles



# Parse the text
articles = parse_legal_articles(codPenal)
# print(articles)
# Print the result to see how it looks
# for i in range(0, len(articles)):
#     stringArticle = json.dumps(articles[i])
#     print(get_embedding(stringArticle))





# print(page.extract_text())









# with open('test_chatbot.txt') as file :
#     arrFraze = (file.read().split("\n"))
#     arrFrazeFiltrate = [fraza.strip() for fraza in arrFraze if fraza != ""]
#     print(get_embedding(arrFrazeFiltrate[0]))


#     # non pythonic code 
#     # for fraza in arrFraze :
#     #     if fraza != "":
#     #         arrFrazeFiltrate.append(fraza)

#     print(arrFrazeFiltrate)


#COMPLETION CHAT 

# completion = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   response_format={ "type": "json_object" },
#   messages=[
#     {"role": "system", "content": "You return JSON with the following keys : ArticleName, ArticleNumber, ArticleText from a string."},
#     {"role": "user", "content": f"Please convert the following legal text into a structured JSON format. The text comprises multiple articles under different sections and chapters of a legal document. Each article is indicated by 'Art.' followed by a number and its title. The articles contain multiple paragraphs that detail legal provisions. For each article, create a separate JSON object and include the following attributes: 'ArtNumber', 'ArtName', and 'ArtText'. 'ArtText' should encompass all the subparagraphs within that article. Ensure the JSON objects are correctly formatted and capture all the details as specified. The text: {codPenal}"}
#   ]
# )

# result=completion.choices[0].message.content
# print(result)

