
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from adv_agent import graph


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
def chat(request: ChatRequest):

    initial_state = {
        "query": request.query,
        "messages": [
            HumanMessage(content=request.query)
        ],
        "proteins": [],
        "go_terms": [],
        "entities": []
    }

    result = graph.invoke(
        initial_state,
        config={"recursion_limit": 40}
    )

    final_message = result["messages"][-1]

    answer = final_message.content

    if not isinstance(answer, str):
        answer = str(answer)

    return {
        "answer": answer,
        "query_protein": result.get("query_protein"),
        "proteins": result.get("proteins", []),
        "go_terms": result.get("go_terms", []),
        "entities": result.get("entities", []),
        "blast_result": result.get("blast_result", [])
    }
