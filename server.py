from fastapi import FastAPI
from pydantic import BaseModel

from agent import graph

app = FastAPI()

class ChatRequest(BaseModel):
    query: str

@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    result = graph.invoke({
        "query": request.query
    })

    return result