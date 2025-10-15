import streamlit as st
import os
from ingest import build_index
from rag import generate_answer

st.set_page_config(page_title="📚 KB Search Engine", layout="wide")
st.title("📚 Knowledge-base Search Engine (Offline RAG)")

# Upload documents
uploaded_files = st.file_uploader(
    "Upload PDFs or TXT files", type=['pdf','txt'], accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("data", exist_ok=True)
    for file in uploaded_files:
        file_path = os.path.join("data", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.success(f"{len(uploaded_files)} file(s) uploaded!")

    # Build/load FAISS index
    with st.spinner("Building/loading index..."):
        try:
            vector_store = build_index()
            st.success("Index ready!")

            # Query input
            query = st.text_input("Ask a question about your documents:")
            if query:
                with st.spinner("Generating answer..."):
                    answer = generate_answer(vector_store, query)
                st.markdown("### Answer:")
                st.write(answer)

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Upload one or more PDFs/TXT files to get started.")
