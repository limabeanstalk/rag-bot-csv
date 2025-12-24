# rag_pipeline.py

import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# ---------------------------------------------------------
# 1. Load Embedding Model  (MATCHES NOTEBOOK)
# ---------------------------------------------------------
# Your notebook uses: sentence-transformers/all-MiniLM-L6-v2
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------------
# 2. Load Flan‑T5 LLM (MATCHES NOTEBOOK)
# ---------------------------------------------------------
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

llm = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    max_length=768,
    temperature=0.2,
)

# ---------------------------------------------------------
# 3. MongoDB Initialization (MATCHES NOTEBOOK)
# ---------------------------------------------------------
def init_mongo(uri):
    client = MongoClient(uri)
    db = client["gxp_guide"]              # <-- NOTEBOOK DB
    return db["regulatory_chunks"]        # <-- NOTEBOOK COLLECTION


# ---------------------------------------------------------
# 4. Semantic Search (MATCHES NOTEBOOK)
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
# 5. Build Context (MATCHES NOTEBOOK)
# ---------------------------------------------------------
def build_context(chunks):
    context = ""
    for i, c in enumerate(chunks):
        context += f"=== CHUNK {i+1} ===\n{c['text']}\n\n"
    return context


# ---------------------------------------------------------
# 6. Generate Answer (MATCHES NOTEBOOK)
# ---------------------------------------------------------
def generate_answer(question, context):

    prompt = f"""
You are a regulatory assistant. Use ONLY the text in the context below to answer the question. 
If the answer is not contained in the context, say you cannot find it.

Context:
{context}

Question:
{query}

Answer clearly and cite section numbers when possible.
"""

    response = llm(prompt)[0]["generated_text"]
    return response


# ---------------------------------------------------------
# 7. Public API for Streamlit (MATCHES NOTEBOOK LOGIC)
# ---------------------------------------------------------
def ask(question, collection, k=5):

    # Step 1: retrieve initial candidates
    initial_chunks = semantic_search(question, collection, k=k)

    if not initial_chunks:
        return "No relevant context found in the knowledge base."

    # Step 2: build context (no reranker in notebook)
    context = build_context(initial_chunks[:3])   # top 3 like notebook

    # Step 3: generate answer
    answer = generate_answer(question, context)

    return answer
