# rag_pipeline.py
print("Loaded rag_pipeline.py successfully")

import numpy as np
import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import CrossEncoder

# -----------------------------
# 1. Load Embedding Model
# -----------------------------
embedder = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

# ----------------------------- 
# # 1B. Load Reranker 
# ----------------------------- 
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"

# -----------------------------
# 2. Connect to MongoDB
# -----------------------------
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

db = client["gxp-guide"]
collection = db["chunks"] 

# -----------------------------
# 3. Load LLM (Flan)
# -----------------------------
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

llm = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=400,
    temperature=0.2,
)

# -----------------------------
# 4. Semantic Search
# -----------------------------
def semantic_search(query, k=5):
   initial_k = 20
   query_emb = embedder.encode(query)
    results = []

    for doc in collection.find():
        score = np.dot(query_emb, doc["embedding"]) / (
            np.linalg.norm(query_emb) * np.linalg.norm(doc["embedding"])
        )
        results.append((score, doc))

    results.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in results[:initial_k]]

# ----------------------------- 
# # 4B. Rerank Results
# ----------------------------- 
def rerank_results(query, docs):
    pairs for reranker pairs = [(query, d["text"]) for d in docs] 
    scores = reranker.predict(pairs) 
    
    ranked = sorted(
        zip(docs, scores), 
        key=lambda x: x[1], 
        reverse=True
     ) 
     
     top_docs = [doc for doc, score in ranked[:5]]
     
     return top_docs

# -----------------------------
# 5. Build Context
# -----------------------------
def build_context(chunks):
    context = ""
    for i,c in enumerate(chunks):
        context +=f"+++ CHUNK {i+1} ===\n{c["text"]}\n\n"
    return context

# -----------------------------
# 6. Generate Answer
# -----------------------------
def generate_answer(question, context):
    prompt = f"""
You are an expert assistant specializing in FDA 21 CFR Part 11 and GxP compliance.

Using ONLY the information provided in the CONTEXT, write a detailed and accurate answer to the QUESTION.
If the answer is not contained in the context, say: "The context does not contain that information."

Your answer must be:
- specific
- complete
- written in full sentences
- based strictly on the context
- not generic or vague

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
    initial_chunks = semantic_search(question, k=k)
    reranked_chunks = rerank_results(question, initial_chunks)
    context = build_context(reranked_chunks)
    answer = generate_answer(question, context)
    return answer