from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def generate_answer(vector_store, query):
    # Define a compact question-answer prompt
    template = """
    You are a knowledgeable tutor. Use the given context to answer clearly and briefly.
    Context: {context}
    Question: {question}
    Answer:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    # Load a small open model from Hugging Face (no key needed)
    from transformers import pipeline
    qa_model = pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )

    # Perform similarity search to get relevant chunks
    docs = vector_store.similarity_search(query, k=3)
    if not docs:
        return "Sorry, I couldn't find relevant information in your uploaded files."

    # Combine the top chunks
    context = "\n".join([d.page_content for d in docs])

    # Generate summarized answer
    input_text = template.format(context=context, question=query)
    response = qa_model(input_text, max_length=256, do_sample=False)

    return response[0]['generated_text']
