from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Initialize small local LLM
pipe = pipeline("text2text-generation", model="google/flan-t5-small")
llm = HuggingFacePipeline(pipeline=pipe)

def generate_answer(vector_store, query):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)

    if not docs:
        return "No relevant content found."

    # Combine top 3 chunks
    context = " ".join([doc.page_content for doc in docs[:3]])
    prompt = f"Using the following text, answer the question in simple words:\n\nText: {context}\n\nQuestion: {query}\nAnswer:"

    # LLM generates answer
    answer = llm(prompt)  # returns string
    return answer
