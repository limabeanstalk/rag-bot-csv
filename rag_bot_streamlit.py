import streamlit as st
from rag_pipeline import ask

st.title("21 CFR Part 11 RAG Assistant")

question = st.text_input("Ask a question about Part 11:")

if st.button("Submit"):
    answer = ask(question)
    st.write(answer)
