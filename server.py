from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path 
from pydantic import BaseModel, Field 
import json
import subprocess
from sqlalchemy import create_engine, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from agent import graph

DATABASE_DIR = Path("blast_database")
DATABASE_DIR.mkdir(exist_ok=True)
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import os

load_dotenv()
database_url = os.getenv("DATABASE_URL")

engine = create_engine(database_url)

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "user"

    name: Mapped[str] = mapped_column(String)
    email: Mapped[str] = mapped_column(String,primary_key=True)
    password: Mapped[str] = mapped_column(String)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/login")
def login(email: str, password: str):
    with Session(engine) as session:
        user = session.query(User).filter(User.email==email,User.password==password).first()
        if user is None:
            return {
                "message": "No user found"
            }
        return{
            "message": "Success"
        }

@app.get("/register")
def login(name: str, email: str, password: str):
    with Session(engine) as session:
        user = session.query(User).filter(User.email==email).first()
        if user is not None:
            return {
                "message": "email already in use"
            }
        user = User(name=name,email=email,password=password)
        session.add(user)
        session.commit()
        return{
            "message": "Success"
        }

@app.post("/chat")
def chat(request: ChatRequest):
    result = graph.invoke({
        "query": request.query
    })

    return result