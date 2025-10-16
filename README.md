# 📚 Knowledge-base Search Engine (RAG)

# can access web application here

 Streamlit App : https://goharunnisaa-kb-search-app-o3umwu.streamlit.app/

## Description
This project is a **document-based search engine** that allows users to upload PDF or TXT files and ask questions about their content. It uses **vector embeddings** and **FAISS** for fast retrieval of relevant text chunks, and a **local language model (HuggingFace FLAN-T5 small)** to generate **clean, synthesized answers** in natural language.

The system ensures that:  
- Multiple documents can be uploaded at once  
- Relevant information is retrieved quickly from large documents  
- Answers are readable and concise  
- Fully offline operation — **no OpenAI API key needed**

**Use Cases:**  
- Students querying lecture notes or study materials  
- Professionals searching through reports or manuals  
- Any scenario where quick, accurate answers from documents are needed

---

## Features
- Upload PDFs or TXT files  
- Build and save a FAISS vector index for fast retrieval  
- Retrieve relevant content based on your query  
- Generate clean and readable answers using a local LLM  
- Fully offline — no API key required  

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/goharunnisaa/KB-Search.git
cd KB-search

Author

Mohammad Goharunnisaa

GitHub: https://github.com/goharunnisaa
