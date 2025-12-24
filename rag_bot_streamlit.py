import streamlit as st
from rag_pipeline import ask, init_mongo

st.title("GxP Guide")

# Load MongoDB URI from Streamlit Secrets
MONGO_URI = st.secrets["MONGO_URI"]

# Initialize DB connection
collection = init_mongo(MONGO_URI)

question = st.text_input(
    "Ask a question about GxP compliance (so far, I only know about 21 CFR Part 11 - I'm still learning!):"
)

if st.button("Ask"):
    answer = ask(question, collection=collection)
    st.subheader("Answer")
    st.write(answer)
