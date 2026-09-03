from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path 
from pydantic import BaseModel, Field 
import json
import subprocess

from agent import graph

DATABASE_DIR = Path("blast_database")
DATABASE_DIR.mkdir(exist_ok=True)

app = FastAPI()

class ChatRequest(BaseModel):
    query: str

@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(
    organism_database: str = Form(...),
    file: UploadFile = File(...)
):
    organism_database_dir = Path(f"{DATABASE_DIR}/{organism_database}")
    organism_database_dir.mkdir(exist_ok=True)
    file_path = organism_database_dir/file.filename
    with open(file_path, "wb") as f:
        while chunk := await file.read(1024*1024):
            f.write(chunk)

    return{
        "status": "success",
        "filename": file.filename
    }


@app.post("/chat")
def chat(request: ChatRequest):
    result = graph.invoke({
        "query": request.query
    })

    return result