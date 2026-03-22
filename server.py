from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from chatbot import run_chatbot
from pydantic import BaseModel

#uvicorn server:app --reload

app = FastAPI()

class ChatRequest(BaseModel):
    question: str


@app.post("/chatbot")
def read_chatbot(chat_request: ChatRequest)-> dict:
    """
    Receives a question and returns a simulated chatbot response based on the Romanian penal code.
    """
    answer = run_chatbot(chat_request.question)

    return {"answer": answer[0]}

