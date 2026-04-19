# Phase 1 libraries
import os
import warnings
import logging

import streamlit as st

# Phase 2
from langchain_groq import ChatGroq

# Phase 3
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# Disable warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title('Ask Chatbot!')

# -------- CHAT MEMORY --------
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])


def get_chat_history():
    history = ""
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    return history


# -------- VECTOR STORE (CACHED) --------
@st.cache_resource
def get_vectorstore():
    loader = PyPDFLoader("./reflexion.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    texts = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L12-v2"
    )

    from langchain_community.vectorstores import FAISS
    return FAISS.from_documents(texts, embeddings)


# -------- INPUT --------
prompt = st.chat_input("Pass your prompt here")

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # -------- MODEL --------
    groq_chat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

    history = get_chat_history()

    # -------- SMART RAG DECISION --------
    use_rag = any(word in prompt.lower() for word in [
        "ai", "machine learning", "neural", "model", "data", "algorithm"
    ])

    try:
        if use_rag:
            # Show loading instead of "Running..."
            with st.spinner("Reading PDF and thinking..."):
                vectorstore = get_vectorstore()

                chain = RetrievalQA.from_chain_type(
                    llm=groq_chat,
                    chain_type='stuff',
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True
                )

                result = chain({"query": prompt})
                response_from_pdf = result["result"]
                response_from_pdf = response_from_pdf.replace("Ɵ", "ti").replace("ﬁ", "fi")

            #  Filter weak answers
            bad_phrases = ["don't know", "not provided", "not mentioned"]

            if any(p in response_from_pdf.lower() for p in bad_phrases):
                response = groq_chat.invoke(f"""
You are a helpful chatbot.

Conversation so far:
{history}

Rules:
- Answer naturally
- Do NOT assume everything is about AI

User: {prompt}
""").content
            else:
                response = response_from_pdf

        else:
            #  FAST RESPONSE (NO PDF)
            response = groq_chat.invoke(f"""
You are a helpful chatbot.

Conversation so far:
{history}

Rules:
- Continue conversation naturally
- Do NOT assume everything is about AI

User: {prompt}
""").content

    except Exception:
        # FULL FALLBACK
        response = groq_chat.invoke(f"""
You are a helpful chatbot.

Conversation so far:
{history}

User: {prompt}
""").content

    # -------- DISPLAY --------
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})