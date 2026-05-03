from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
async def read_chatbot(chat_request: ChatRequest):
    """
    Receives a question and streams the chatbot response token by token.
    """
    return StreamingResponse(
        run_chatbot(chat_request.question),
        media_type="text/plain",
    )
