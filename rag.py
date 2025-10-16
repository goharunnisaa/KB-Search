import os
import pickle
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# ------------------------------
# 1. Build or load the vector store
# ------------------------------

INDEX_PATH = "vector_store.index/index.pkl"
DATA_PATH = "data/knowledge_base.txt"  # adjust if your data file has a different name

if os.path.exists(INDEX_PATH):
    with open(INDEX_PATH, "rb") as f:
        vector_store = pickle.load(f)
else:
    print("⚙️ Building new vector store...")
    os.makedirs("vector_store.index", exist_ok=True)

    # Load documents
    loader = TextLoader(DATA_PATH)
    docs = loader.load()

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Build vector store
    vector_store = FAISS.from_documents(docs, embeddings)

    # Save it for future use
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(vector_store, f)
    print("✅ Vector store created and saved!")

# ------------------------------
# 2. Initialize small local LLM
# ------------------------------
pipe = pipeline("text2text-generation", model="google/flan-t5-small")
llm = HuggingFacePipeline(pipeline=pipe)

# ------------------------------
# 3. Generate answer function
# ------------------------------
def generate_answer(query):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)

    if not docs:
        return "No relevant content found."

    # Combine top 3 chunks
    context = " ".join([doc.page_content for doc in docs[:3]])
    prompt = f"Using the following text, answer the question in simple words:\n\nText: {context}\n\nQuestion: {query}\nAnswer:"

    # Generate answer using local LLM
    answer = llm(prompt)
    return answer
