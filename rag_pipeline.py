# rag_pipeline.py

import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -----------------------------
# 1. Load Embedding Model
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 2. Connect to MongoDB
# -----------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["ragdb"]
collection = db["chunks"]

# -----------------------------
# 3. Load LLM (TinyLlama)
# -----------------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.2,
)

# -----------------------------
# 4. Semantic Search
# -----------------------------
def semantic_search(query, k=5):
    query_emb = embedder.encode(query)
    results = []

    for doc in collection.find():
        score = np.dot(query_emb, doc["embedding"]) / (
            np.linalg.norm(query_emb) * np.linalg.norm(doc["embedding"])
        )
        results.append((score, doc))

    results.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in results[:k]]

# -----------------------------
# 5. Build Context
# -----------------------------
def build_context(chunks):
    context = ""
    for c in chunks:
        context += c["text"] + "\n---\n"
    return context

# -----------------------------
# 6. Generate Answer
# -----------------------------
def generate_answer(question, context):
    prompt = f"""
You are a helpful assistant answering questions about 21 CFR Part 11.

Use ONLY the information in the CONTEXT to answer.
If the answer is not in the context, say you do not know.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    output = llm(prompt)[0]["generated_text"]
    return output

# -----------------------------
# 7. Public API for Streamlit
# -----------------------------
def ask(question, k=5):
    chunks = semantic_search(question, k=k)
    context = build_context(chunks)
    answer = generate_answer(question, context)
    return answer
