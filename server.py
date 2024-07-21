from flask import Flask, request, jsonify
from chatbot import get_embedding, get_articles_milvus,get_chatbot_answer

app = Flask(__name__) 

# Variable to store the received string
stored_string = ""


@app.route('/query', methods=['POST'])
def store_string():
    global stored_string
    data = request.json

    stored_string = data.get('text', '')
    print(f"Received text: {stored_string}")

    try:
        embeddingArray = get_embedding(stored_string, model="text-embedding-3-large")

        answers = get_articles_milvus(embeddingArray)

        answer = get_chatbot_answer(stored_string, answers)
        print(f"Chatbot Answer: {answer}")

        return jsonify({"message": answer}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"message": "An error occurred"}), 500

@app.route('/get_string', methods=['GET'])
def get_string():
    return jsonify({"stored_string": stored_string}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # de schimbat in productie 
