# Pro-RAG: Persistent, Hybrid, and Agentic RAG System

This project is an advanced **Retrieval-Augmented Generation (RAG)** system designed for high precision and production-level persistence. Unlike basic RAG tutorials, this system implements professional techniques to solve common accuracy and context issues.

##  Advanced Features

* **Parent-Document Retrieval:** Uses small chunks for high-accuracy vector search but returns larger parent documents to the LLM for better context.
* **Hybrid Search (Ensemble):** Combines **Semantic Search** (Vector) with **Keyword Search** (BM25) to catch both meaning and exact terminology.
* **JSON-Based Multi-Query Expansion:** Rewrites user queries into multiple variations using JSON parsing to prevent "bullet-point noise" from ruining vector embeddings.
* **Cross-Encoder Reranking:** Uses a secondary model (`ms-marco-MiniLM`) to re-score retrieved documents, significantly reducing hallucinations.
* **Local Persistence:** Uses `ChromaDB` and `LocalFileStore` so that your indexed data survives script restarts (no redundant re-indexing).

## Tech Stack
- **Framework:** LangChain
- **LLM:** OpenAI GPT-3.5-Turbo (or GPT-4)
- **Vector DB:** ChromaDB
- **Embeddings:** BGE-Small-EN (HuggingFace)
- **Reranker:** Cross-Encoder (Sentence-Transformers)

##  Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/pro-rag-system.git](https://github.com/YOUR_USERNAME/pro-rag-system.git)
   cd pro-rag-system
