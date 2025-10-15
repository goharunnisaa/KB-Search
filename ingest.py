import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader

INDEX_FILE = "vector_store.index"

# Load documents
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

# Build FAISS index
def build_index():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load existing index if available
    if os.path.exists(INDEX_FILE):
        return FAISS.load_local(INDEX_FILE, embeddings, allow_dangerous_deserialization=True)

    docs = load_documents()
    if not docs:
        raise ValueError("No documents found in 'data/' folder.")

    # Split docs into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    # Create vector store
    vector_store = FAISS.from_documents(split_docs, embeddings)

    # Save locally
    vector_store.save_local(INDEX_FILE)
    return vector_store
