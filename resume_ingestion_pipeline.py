import os
import uuid
import requests
from pathlib import Path
from pypdf import PdfReader
from docx import Document
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME","portfolio-chatbot")
NAMESPACE = os.getenv("PINECONE_NAMESPACE","resume-content")
DIMENSION = 384

def parse_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)

def parse_docx(file_path: str) -> str:
    document = Document(file_path)
    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)

def parse_resume(file_path: str) -> str:
    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return parse_pdf(file_path)
    elif extension == ".docx":
        return parse_docx(file_path)
    else:
        raise ValueError(
            "Only PDF and DOCX files are supported!"
        )

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = text.replace("\r", "\n")

    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    text = "\n".join(lines)

    chunks = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

def create_embeddings(chunks):
    API_URL = (
        "https://router.huggingface.co/"
        "hf-inference/models/"
        "sentence-transformers/all-MiniLM-L6-v2/"
        "pipeline/feature-extraction"
    )

    headers = {
        "Authorization": f"Bearer {os.getenv('HF_API_KEY')}",
        "Content-Type": "application/json"
    }

    embeddings = []

    for chunk in chunks:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": chunk},
            timeout=60
        )

        response.raise_for_status()

        embedding = response.json()

        if isinstance(embedding[0], list):
            embedding = embedding[0]

        embeddings.append(embedding)

    return embeddings

def get_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    indexes = [
        index["name"]
        for index in pc.list_indexes()
    ]

    if INDEX_NAME not in indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print("Pinecone index created successfully!")

    return pc.Index(INDEX_NAME)


def insert_resume(resume_id,chunks,embeddings):
    index = get_index()

    vectors = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"{resume_id}-{i}",
            "values": embedding,
            "metadata": {
                "resume_id": resume_id,
                "chunk_id": i,
                "text": chunk
            }
        })

    index.upsert(vectors=vectors,namespace=NAMESPACE)
    print(f"Inserted {len(vectors)} chunks successfully!")

def ingest(file_path, resume_id=None):

    if resume_id is None:
        resume_id = str(uuid.uuid4())

    print(f"Resume ID: {resume_id}")

    print("Parsing resume...")
    text = parse_resume(file_path)

    if not text.strip():
        raise ValueError(
            "No text found in the uploaded resume!"
        )

    print("Extracted text from the resume successfully!")
    
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks!")

    print("Creating embeddings...")
    embeddings = create_embeddings(chunks)

    print("Uploading to pinecone...")
    insert_resume(resume_id, chunks, embeddings)

    print("Resume ingestion completed successfully!")

print("Resume ingestion pipeline started!")
file_path = input("Enter the path to the resume file (PDF or DOCX):").strip()
resume_id = "resume1"

ingest(file_path, resume_id)