import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader

INDEX_DIR = "vector_store.index"

def load_documents(folder_path="data"):
    docs = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for file_name in os.listdir(folder_path):
        path = os.path.join(folder_path, file_name)
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif file_name.endswith(".txt"):
            loader = TextLoader(path)
            docs.extend(loader.load())
    return docs

def build_index():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create the folder if not present
    os.makedirs(INDEX_DIR, exist_ok=True)

    docs = load_documents()
    if not docs:
        raise ValueError("No documents found in 'data/' folder.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(INDEX_DIR)
    print("✅ Index built and saved.")
    return vector_store
