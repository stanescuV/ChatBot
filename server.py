from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from chatbot import run_chatbot
from pydantic import BaseModel

#uv run uvicorn server:app --reload
#uv run uvicorn server:app --host 0.0.0.0 --port 8000 --reload

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str


@app.post("/chatbot")
def read_chatbot(chat_request: ChatRequest)-> dict:
    """
    Receives a question and returns a simulated chatbot response based on the Romanian penal code.
    """
    answer = run_chatbot(chat_request.question)

    return {"answer": answer[0]}

