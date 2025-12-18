# 📘 CSV RAG Chatbot for Pharma

A Retrieval-Augmented Generation (RAG) chatbot designed to assist with **Computer System Validation (CSV)** and regulatory guidance in pharmaceutical and biotech environments.

This project demonstrates how AI can be built and documented in a **GxP-compliant, validation-style manner**, including requirements, risk assessment, and traceability.

---

## 🚀 Features

- **Phi-3 Mini** language model for grounded responses
- **BAAI/bge-small-en** embeddings for semantic search
- **ChromaDB** vector database for retrieval
- **LangChain** orchestration
- **Streamlit** user interface
- **Validation-style documentation** (URS, FRS, Risk Assessment, Trace Matrix, Test Plan)

---

## 🏗️ Architecture

```mermaid
flowchart TD
    A[User Query] --> B[Embed Query (BGE-small-en)]
    B --> C[Retrieve Top-k Chunks (ChromaDB)]
    C --> D[Construct Prompt (LangChain)]
    D --> E[Phi-3 Mini LLM]
    E --> F[Answer with Sources]
