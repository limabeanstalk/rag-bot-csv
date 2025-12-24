# rag_pipeline.py

import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# ---------------------------------------------------------
# 1. Load Embedding Model
# ---------------------------------------------------------
embedder = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")


# ---------------------------------------------------------
# 2. Load Reranker
# ---------------------------------------------------------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------------------------------------------------------
# 3. Load Flan-T5 LLM
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# 4. MongoDB Initialization (called from Streamlit)
# ---------------------------------------------------------
def init_mongo(uri):
    client = MongoClient(uri)
    db = client["gxp-guide"]
    return db["chunks"]


# ---------------------------------------------------------
# 5. Semantic Search (retrieve top 20 candidates)
# ---------------------------------------------------------
def semantic_search(query, collection, k=5):

    initial_k = 20

    # Embed and normalize query
    query_emb = embedder.encode(query, normalize_embeddings=True)

    results = []

    for doc in collection.find():
        emb = np.array(doc["embedding"], dtype=float)

        # Normalize stored embedding
        emb = emb / np.linalg.norm(emb)

        # Cosine similarity
        score = np.dot(query_emb, emb)

        results.append((score, doc))

    # Sort by similarity
    results.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in results[:initial_k]]

# ---------------------------------------------------------
# 6. Rerank Results
# ---------------------------------------------------------
def rerank_results(query, docs):
    if not docs:
        return []

    pairs = [(query, d["text"]) for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, score in ranked[:5]]


# ---------------------------------------------------------
# 7. Build Context
# ---------------------------------------------------------
def build_context(chunks):
    context = ""
    for i, c in enumerate(chunks):
        context += f"=== CHUNK {i+1} ===\n{c['text']}\n\n"
    return context


# ---------------------------------------------------------
# 8. Generate Answer
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# 9. Public API for Streamlit
# ---------------------------------------------------------
def ask(question, collection, k=5):

    # Step 1: retrieve initial candidates
    initial_chunks = semantic_search(question, collection, k=k)

    if not initial_chunks:
        return "No relevant context found in the knowledge base."

    # Step 2: rerank
    reranked_chunks = rerank_results(question, initial_chunks)

    # Step 3: build context
    context = build_context(reranked_chunks)

    # Step 4: generate answer
    answer = generate_answer(question, context)

    return answer
