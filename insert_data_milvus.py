# import re
# import json
# from milvus_handler import MilvusHandler
# from chatbot import get_embedding

##THIS FILE ONLY PURPOSE IS TO INSERT THE DATA FROM Penal CODE pdf into milvus,
## That s all, use it only once. 


# with open("codPenal.txt", "r", encoding='utf-8') as file:
#     codPenal = file.read()
    
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


#     # print(articles)
#     return articles

# newArticles = parse_legal_articles(codPenal)

# # print(newArticles)

# embeddings = []
# for art in newArticles:
#     text = f"Articol {art['ArtNumber']} - {art['ArtName']}\n{art['ArtText']}"
#     vector = get_embedding(text)  
#     embeddings.append({"text": text, "embedding": vector})

# print(embeddings)

# with open("embeddings.json", "w", encoding="utf-8") as f:
#     json.dump(embeddings, f, ensure_ascii=False)

# print("✅ Saved", len(embeddings), "embeddings to embeddings.json")


# # 2. Insert into Milvus
# milvus = MilvusHandler()
# milvus.insert(embeddings)
# milvus.collection.flush()