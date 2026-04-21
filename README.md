# Smart RAG Chatbot

An AI-powered chatbot that intelligently answers questions from uploaded PDFs or general knowledge using Retrieval-Augmented Generation (RAG).

---

##  Features

-  Upload a PDF and ask questions
-  Smart switching between:
- Document-based answers (RAG)
- General AI responses
-  Fast responses using Groq LLM (LLaMA 3)
-  Chat memory for conversational flow
- Semantic search using FAISS vector database
-  Clean and interactive UI with Streamlit

---

## Tech Stack

- **Python**
- **Streamlit** (UI)
- **LangChain**
- **Groq API (LLaMA 3)**
- **FAISS** (Vector database)
- **HuggingFace Embeddings**

---

## How It Works

1. PDF is loaded and split into smaller chunks  
2. Text is converted into embeddings  
3. Stored in FAISS vector database  
4. User query is analyzed:
   - If relevant → uses RAG (PDF)
   - Else → normal LLM response  
5. Returns accurate and contextual answer  

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
