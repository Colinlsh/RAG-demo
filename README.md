# 🚀 **COLIN Rag Demo 1.0 – GraphRAG & Chat History Integration!**

This chatbot enables **fast, accurate, and explainable retrieval of information** from PDFs, DOCX, and TXT files using **OpenAI embeddings**, **BM25**, **FAISS**, **Neural Reranking (Cross-Encoder)**, **GraphRAG**, and **Chat History Integration**.

---

## Different pipelines are available for various RAG (Retrieval-Augmented Generation) methods.

Example:
Basic RAG Pipeline: Uses FAISS for retrieval and a simple prompt format.
Advanced RAG Pipeline: Uses ChromaDB with metadata filtering for more precise document retrieval.

## \*Installation & Setup\*\*

You can install and run the **COLIN Rag Demo 1.0** in one of two ways:

1. **Traditional (Python/venv) Installation**
2. **Docker Installation** (ideal for containerized deployments)

---

## **1️⃣ Traditional (Python/venv) Installation**

### **Step A: Clone the Repository & Install Dependencies**

```
gh repo clone Colinlsh/RAG-demo
cd RAG-Demo

# Create a virtual environment
python -m venv venv

# Activate your environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip (optional, but recommended)
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

### **Step B: Run the Chatbot**

1. Launch the Streamlit app:
   ```
   streamlit run app.py
   ```
2. Open your browser at **[http://localhost:8501](http://localhost:8501)** to access the chatbot UI.

---

## **2️⃣ Docker Installation**

### **A) Single-Container Approach (Ollama on Your Host)**

If **Ollama** is already **installed on your host machine** and listening at `localhost:11434`, do the following:

1. **Build & Run**:
   ```
   docker-compose build
   docker-compose up
   ```
2. The app is now served at **[http://localhost:8501](http://localhost:8501)**. Ollama runs on your host, and the container accesses it via the specified URL.

### **B) Two-Container Approach (Ollama in Docker)**

If you prefer **everything** in Docker:

```
version: "3.8"

services:
  ollama:
    image: ghcr.io/jmorganca/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"

  deepgraph-rag-service:
    container_name: deepgraph-rag-service
    build: .
    ports:
      - "8501:8501"
    environment:
      - OLLAMA_API_URL=http://ollama:11434
      - MODEL=deepseek-r1:7b
      - EMBEDDINGS_MODEL=nomic-embed-text:latest
      - CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
    depends_on:
      - ollama
```

Then:

```
docker-compose build
docker-compose up
```

Both **Ollama** and the chatbot run in Docker. Access the chatbot at **[http://localhost:8501](http://localhost:8501)**.

### **But consider step A) for comfort..**

---

# **How the Chatbot Works**

1. **Upload Documents**: Add PDFs, DOCX, or TXT files via the sidebar.
2. **Hybrid Retrieval**: Combines **BM25** and **FAISS** to fetch the most relevant text chunks.
3. **GraphRAG Processing**: Builds a **Knowledge Graph** from your documents to understand relationships and context.
4. **Neural Reranking**: Uses a **Cross-Encoder** model for reordering the retrieved chunks by relevance.
5. **Query Expansion (HyDE)**: Generates hypothetical answers to **expand** your query for better recall.
6. **Chat Memory History Integration**: Maintains context by referencing previous user messages.

---
