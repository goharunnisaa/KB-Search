import streamlit as st
import os
from ingest import build_index
from rag import generate_answer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="📚 KB Search Engine", layout="wide")
st.title("📚 Knowledge-base Search Engine (RAG)")

uploaded_files = st.file_uploader(
    "Upload PDFs or TXT files", type=['pdf', 'txt'], accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("data", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join("data", file.name), "wb") as f:
            f.write(file.getbuffer())
    st.success(f"{len(uploaded_files)} file(s) uploaded!")

    with st.spinner("Building/loading vector store..."):
        try:
            if not os.path.exists("vector_store.index/index.pkl"):
                vector_store = build_index()
            else:
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_store = FAISS.load_local(
                    "vector_store.index", embeddings, allow_dangerous_deserialization=True
                )
            st.success("✅ Index ready!")

            query = st.text_input("Ask a question about your documents:")
            if query:
                with st.spinner("Retrieving answer..."):
                    answer = generate_answer(vector_store, query)
                st.markdown("### Answer:")
                st.write(answer)

        except Exception as e:
            st.error(f"Error building index: {e}")

else:
    st.info("Upload one or more PDFs/TXT files to get started.")
