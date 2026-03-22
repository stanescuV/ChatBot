from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from chatbot_langgraph import run_chatbot
from pydantic import BaseModel

#uvicorn server:app --reload

app = FastAPI()

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
