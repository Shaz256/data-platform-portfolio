from fastapi import FastAPI
from app.rag import get_response

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Medical AI Chatbot Running"}

@app.get("/chat")
def chat(query: str):
    answer = get_response(query)
    return {"response": answer}