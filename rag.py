from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

def generate_answer(vector_store, query):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 256})
    
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa.run(query)
    return result
