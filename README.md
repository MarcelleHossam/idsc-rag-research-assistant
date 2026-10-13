# IDSC Research Assistant — RAG System

An Arabic-English Retrieval-Augmented Generation (RAG) research assistant that answers questions using IDSC publications and video transcripts.

## Features
- Semantic search using vector embeddings
- Retrieval from research publications and video transcripts
- AI-generated answers grounded in retrieved context
- Source references with page numbers or video timestamps
- Arabic and English query support
- Web-based chat interface

## Tech Stack
- Python
- Flask
- ChromaDB
- OpenAI-compatible API client
- Fireworks AI
- PyMuPDF

## Setup
1. Install Python.
2. Clone this repository.
3. Install dependencies with `pip install -r requirements.txt`.
4. Create a `.env` file containing your Fireworks API key.
5. Run `python server.py`.
6. Open `http://127.0.0.1:5000`.

The vector database may need to be rebuilt on first startup, which requires a valid API key.


