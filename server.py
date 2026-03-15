from fastapi import FastAPI
from chatbot import run_chatbot
from pydantic import BaseModel

#uvicorn server:app --reload

app = FastAPI()

class ChatRequest(BaseModel):
    question: str


@app.post("/chatbot")
async def read_chatbot(chat_request: ChatRequest)-> dict:
    """
    Receives a question and returns a simulated chatbot response based on the Romanian penal code.
    """
    question = chat_request.question
    
    answer = run_chatbot(question)
    
    return {"answer": answer[0]}

