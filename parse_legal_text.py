
import re


with open("codPenal.txt", "r", encoding='utf-8') as file:
    codPenal = file.read()


# From big String to an array of articles

def parse_legal_articles(text):
    # Capture the whole header (Art. 238.) AND the number (238)
    pattern = re.compile(r'(Art\.\s*(\d+)\.)')

    # With capturing groups, split() returns:
    # [pre, header1, number1, content1, header2, number2, content2, ...]
    parts = pattern.split(text)

    articles = []
    skip_keywords = ("CAP", "SEC", "TIT")

    # Walk the array in chunks of 3: header, number, content
    for i in range(1, len(parts) - 1, 3):
        header = parts[i].strip()              # e.g., "Art. 238."
        number_str = parts[i + 1].strip()      # e.g., "238"
        content = parts[i + 2]                 # text after the header

        # Stop content at the next structural keyword (if any)
        for kw in skip_keywords:
            pos = content.find(kw)
            if pos != -1:
                content = content[:pos]
                break

        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content).strip()

        # Extract title (ArtName) = text before first period in the content
        if '.' in content:
            art_name, art_text = content.split('.', 1)
            art_name = art_name.strip()
            art_text = art_text.strip()
        else:
            art_name = content.strip()
            art_text = ""

        # Convert number to int when possible
        try:
            art_number = int(number_str)
        except ValueError:
            art_number = number_str  # fallback to string if unexpected format

        articles.append({
            'ArtNumber': art_number,
            'ArtName': art_name,
            'ArtText': art_text
        })


    return articles


