# rag_pipeline.py

import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import streamlit as st
import os
import requests

# ---------------------------------------------------------
# 0. Load API Key
# ---------------------------------------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ---------------------------------------------------------
# 1. Load Embedding Model 
# ---------------------------------------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------------------------
# 2. Load Llama LLM
# ---------------------------------------------------------
def llm(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 500
    }

    response = requests.post(url, headers=headers, json=payload)

    # Parse JSON safely
    try:
        data = response.json()
    except Exception:
        return "Groq returned a non‑JSON response."

    # If Groq returned an error, surface it
    if "error" in data:
        return f"Groq API error: {data['error']}"

    # If choices are missing, show the whole response for debugging
    if "choices" not in data:
        return f"Unexpected Groq response format: {data}"

    # Normal success path
    return data["choices"][0]["message"]["content"]

# ---------------------------------------------------------
# 3. MongoDB Initialization 
# ---------------------------------------------------------
def init_mongo(uri):
    client = MongoClient(uri)
    db = client["gxp_guide"]            
    return db["regulatory_chunks"]      

# ---------------------------------------------------------
# 4. Semantic Search 
# ---------------------------------------------------------
def semantic_search(query, collection, k=5):

    initial_k = 20  # retrieve more for reranking

    # Embed + normalize query (MATCHES NOTEBOOK)
    query_emb = embedder.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    results = []

    for doc in collection.find():
        emb = np.array(doc["embedding"], dtype=float)

        # Normalize stored embedding (MATCHES NOTEBOOK)
        emb = emb / np.linalg.norm(emb)

        # Cosine similarity (MATCHES NOTEBOOK)
        score = np.dot(query_emb, emb)

        results.append((score, doc))

    # Sort by similarity
    results.sort(key=lambda x: x[0], reverse=True)

    # Return top initial_k
    return [doc for _, doc in results[:initial_k]]

# ---------------------------------------------------------
# 5. Build Context
# ---------------------------------------------------------
def build_context(chunks):
    context = ""
    for i, c in enumerate(chunks):
        context += f"=== CHUNK {i+1} ===\n{c['text']}\n\n"
    return context

# ---------------------------------------------------------
# 6. Generate Answer
# ---------------------------------------------------------
def generate_answer(question, context):

    prompt = f"""
Use ONLY the context to answer the question. 
If the answer is not in the context, say so.

Context:
{context}

Question:
{question}

Provide a clear, natural explanation in your own words based on the context.
Do not copy the context verbatim unless necessary for accuracy.
Identify the section number that contains the answer and cite it once at the end of your response in parentheses.
Do NOT mention or evaluate any other section numbers.
Do NOT repeat phrases.
"""

    response = llm(prompt)
    return response

# ---------------------------------------------------------
# 7. Public API for Streamlit 
# ---------------------------------------------------------
def ask(question, collection, k=5):

    # Step 1: retrieve initial candidates
    initial_chunks = semantic_search(question, collection, k=k)

    if not initial_chunks:
        return "No relevant context found in the knowledge base."

    # Step 2: build context (no reranker in notebook)
    context = build_context(initial_chunks[:3])  

    # Step 3: generate answer
    answer = generate_answer(question, context)

    return answer
