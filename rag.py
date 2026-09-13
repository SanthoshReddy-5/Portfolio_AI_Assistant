import os
import requests
from groq import Groq
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

def embed_query(query: str):
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

    response = requests.post(
        API_URL,
        headers=headers,
        json={
            "inputs": query
        },
        timeout=60
    )

    response.raise_for_status()

    embedding = response.json()

    if isinstance(embedding[0], list):
        embedding = embedding[0]

    return embedding

INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "portfolio-chatbot"
)

NAMESPACE = os.getenv(
    "PINECONE_NAMESPACE",
    "resume-content"
)

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index = pc.Index(
    INDEX_NAME
)

def search(query_embedding, resume_id, top_k=5):
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE,
        filter={
            "resume_id": {
                "$eq": resume_id
            }
        }
    )

    return results

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

groq = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def ask_question(question: str, resume_id: str):
    query_embedding = embed_query(question)

    results = search(
        query_embedding,
        resume_id,
        top_k=5
    )

    matches = results.get(
        "matches",
        []
    )

    if not matches:
        return {
            "answer": (
                "I could not find relevant "
                "information in the portfolio or resume."
            ),
            "sources": []
        }

    context_parts = []
    sources = []

    for match in matches:
        metadata = match.get(
            "metadata",
            {}
        )

        text = metadata.get(
            "text",
            ""
        )

        score = match.get(
            "score",
            0
        )

        context_parts.append(
            text
        )

        sources.append({
            "chunk_id": metadata.get(
                "chunk_id"
            ),
            "score": score
        })

    context = "\n\n".join(
        context_parts
    )

    system_prompt = """
    You are Santhosh Reddy's Portfolio AI Assistant.

    Answer questions using ONLY information available in the portfolio or resume.

    RESPONSE RULES:
    - Never use Markdown tables.
    - Never use | characters to create tables.
    - Use Markdown headings, paragraphs, bold text, and bullet points.
    - Keep responses concise, professional, and easy to read.
    - Use short paragraphs instead of large blocks of text.
    - Use bullet points when listing technologies, features, skills, responsibilities, or achievements.
    - Highlight important technologies using **bold**.
    - Do not use HTML tags such as <br>, <table>, <tr>, or <td>.
    - Do not repeat information unnecessarily.
    - Never invent or assume information.
    - If information is not available, clearly say that it is not available in the portfolio or resume.

    PROJECTS:
    When asked about projects, use this format:

    ## Projects

    ### Project Name
    **Duration:** Start Date – End Date

    Brief description of the project.

    **Technologies:**
    - Technology 1
    - Technology 2
    - Technology 3

    **Key Features:**
    - Feature 1
    - Feature 2
    - Feature 3

    ---

    ### Another Project
    **Duration:** Start Date – End Date

    Brief description of the project.

    **Technologies:**
    - Technology 1
    - Technology 2

    **Key Features:**
    - Feature 1
    - Feature 2

    SKILLS:
    When asked about skills, organize them into categories:

    ## Skills

    ### Programming Languages
    - Python
    - JavaScript
    - TypeScript

    ### Frameworks & Libraries
    - React
    - Next.js
    - FastAPI

    ### Databases
    - MySQL
    - PostgreSQL

    EXPERIENCE:
    When asked about experience:

    ## Experience

    ### Job Title — Company
    **Duration:** Start Date – End Date

    Brief description of the role.

    **Responsibilities:**
    - Responsibility 1
    - Responsibility 2
    - Responsibility 3

    EDUCATION:
    When asked about education:

    ## Education

    ### Degree — Institution
    **Duration:** Start Date – End Date

    Relevant details.

    GENERAL STYLE:
    - Be professional and conversational.
    - Be concise but informative.
    - Prioritize readability.
    - Never use tables.
    """

    user_prompt = f"""
    Portfolio/Resume Context:
    {context}

    Question:
    {question}
    """

    response = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.1,
        max_tokens=1000
    )

    answer = (
        response.choices[0].message.content
    )

    return {
        "answer": answer,
        "sources": sources
    }