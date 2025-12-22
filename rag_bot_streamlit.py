import streamlit as st
from rag_pipeline import ask

st.title("GxP Guide")

question = st.text_input("Ask a question about GxP compliance (so far, I only know about 21 CFR Part 11 - I'm still learning!):")

if st.button("Ask"):
    answer = ask(question)
    st.write(answer)
