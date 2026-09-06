from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag import ask_question

app = FastAPI(title="Portfolio AI Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://santhoshreddy.netlify.app",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    resume_id: str
    question: str

@app.get("/")
def root():
    return {"message": "Portfolio AI Assistant API is running!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ask")
def ask(request: AskRequest):
    result = ask_question(
        question=request.question,
        resume_id=request.resume_id
    )
    
    print("Asked question:", request.question)
    print("Result:", result)
    return result