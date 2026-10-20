# Portfolio AI Assistant

A **FastAPI-based AI chatbot** that uses **Retrieval-Augmented Generation (RAG)** to answer questions about my portfolio and resume. Resume data is processed and converted into embeddings locally using **FastEmbed** and the `all-MiniLM-L6-v2` embedding model. These embeddings are stored in **Pinecone**, where relevant information is retrieved to generate accurate, context-aware responses using the **Groq API**.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge\&logo=fastapi\&logoColor=white)](https://fastapi.tiangolo.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-000000?style=for-the-badge\&logo=pinecone\&logoColor=white)](https://www.pinecone.io/)
![FastEmbed](https://img.shields.io/badge/FastEmbed-Local%20Embeddings-4CAF50?style=for-the-badge)
[![Groq](https://img.shields.io/badge/Groq-API-F55036?style=for-the-badge\&logo=groq\&logoColor=white)](https://groq.com/)
[![RAG](https://img.shields.io/badge/RAG-Retrieval--Augmented%20Generation-8A2BE2?style=for-the-badge)](#)

## Features

- **Resume-Based Chatbot:** Ask questions about my skills, experience, education, projects, and portfolio.
- **RAG Pipeline:** Retrieves relevant resume information before generating responses.
- **Local Embeddings:** Generates text embeddings using FastEmbed without requiring a paid embedding inference API.
- **Lightweight Model:** Uses `sentence-transformers/all-MiniLM-L6-v2` for 384-dimensional embeddings.
- **Vector Search:** Stores and retrieves resume embeddings using Pinecone.
- **Groq LLM:** Generates fast, relevant responses using the Groq API.
- **FastAPI Backend:** Provides a REST API for chatbot functionality.
- **Memory-Conscious Inference:** Uses ONNX-based embedding inference and a cached model instance to reduce repeated initialization overhead.
- **Context-Aware Responses:** Grounds answers in retrieved portfolio and resume information.

## Tech Stack

- **Framework:** FastAPI
- **Language:** Python
- **Embedding Library:** FastEmbed
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Inference Runtime:** ONNX Runtime
- **Vector Database:** Pinecone
- **LLM Provider:** Groq API
- **Architecture:** Retrieval-Augmented Generation (RAG)
- **Deployment:** Render

## RAG Pipeline

```text
Resume / Portfolio Data
        ↓
   Text Extraction
        ↓
    Text Chunking
        ↓
 FastEmbed Embedding Model
        ↓
  384-Dimensional Vectors
        ↓
      Pinecone
        ↓
   Vector Similarity Search
        ↓
  Relevant Resume Chunks
        ↓
      Groq API
        ↓
   AI-Generated Answer
```

## Project Structure

```text
Portfolio_AI_Assistant/
│
├── main.py
│   └── FastAPI application and API endpoints
│
├── rag.py
│   └── RAG pipeline, local query embeddings,
│       Pinecone retrieval, and Groq responses
│
├── resume_ingestion_pipeline.py
│   └── Resume processing, chunking, local embeddings,
│       and Pinecone vector ingestion
│
├── requirements.txt
│   └── Project dependencies
│
├── .env
│   └── Environment variables (not committed)
│
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.11+
- Pinecone account and API key
- Groq API key
- Internet connection for the initial model download

**Note:** A Hugging Face API token is not required for local embedding generation with a publicly accessible model.

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/SanthoshReddy-5/Portfolio_AI_Assistant.git
cd Portfolio_AI_Assistant
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux or macOS:

```bash
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index
PINECONE_NAMESPACE=resume-content
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Replace the example values with your actual configuration. Never commit your `.env` file or expose API keys publicly.

**5. Run the application**

```bash
uvicorn main:app --reload
```

Open the API documentation: 
```
http://127.0.0.1:8000/docs
```

The embedding model runs locally, but **Pinecone and Groq remain external services**. Their own plans, limits and costs still apply.
