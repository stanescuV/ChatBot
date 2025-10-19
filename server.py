from fastapi import FastAPI
from chatbot import run_chatbot
from pydantic import BaseModel

#uvicorn server:app --reload

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def read_root()-> dict:
    return {"Hello": "World"}

@app.post("/chatbot")
async def read_chatbot(chat_request: ChatRequest)-> dict:
    """
    Receives a question and returns a simulated chatbot response based on the Romanian penal code.
    """
    # FastAPI automatically validates the request matches ChatRequest
    # and makes the data available in the chat_request object.
    question = chat_request.question
    
    # Call your chatbot logic with the question
    answer = run_chatbot(question)
    
    # Return the response as JSON
    return {"answer": answer[0]}

