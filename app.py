# -------- IMPORTS --------
import os
import warnings
import logging
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# -------- SETTINGS --------
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# -------- UI --------
st.title(" Smart RAG Chatbot")
st.caption("Chat with your PDF or ask anything")

# -------- SIDEBAR --------
st.sidebar.header("Controls")

if st.sidebar.button(" Clear Chat"):
    st.session_state.messages = []
    st.rerun()

uploaded_file = st.sidebar.file_uploader(" Upload PDF", type="pdf")

# -------- CHAT MEMORY --------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])


# -------- VECTOR STORE (SAFE VERSION) --------
@st.cache_resource(show_spinner=False)
def create_vectorstore(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    texts = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    safe_texts = []

    for t in texts:
        try:
            if t.page_content and isinstance(t.page_content, str):
                text = t.page_content.strip()

                # Clean problematic characters
                text = text.replace("\n", " ").replace("\x00", "")

                if len(text) > 30:
                    t.page_content = text

                    # Test embedding (prevents crash)
                    embeddings.embed_query(text[:500])

                    safe_texts.append(t)

        except:
            continue

    return FAISS.from_documents(safe_texts, embeddings)


# -------- INPUT --------
prompt = st.chat_input("Ask something...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    groq_chat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

    try:
        with st.spinner("Thinking..."):

            # -------- IF PDF EXISTS --------
            if uploaded_file:
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.read())

                vectorstore = create_vectorstore("temp.pdf")

                chain = RetrievalQA.from_chain_type(
                    llm=groq_chat,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )

                result = chain({"query": prompt})
                answer = result["result"]

                # Ensure safe string
                if not isinstance(answer, str):
                    answer = str(answer)

                answer = answer.replace("Ɵ", "ti").replace("ﬁ", "fi")

                sources = result.get("source_documents", [])

                # -------- SMART FILTER --------
                bad_phrases = ["don't know", "not mentioned", "not provided"]

                use_pdf_answer = (
                    sources
                    and not any(p in answer.lower() for p in bad_phrases)
                )

                if use_pdf_answer:
                    response = answer

                    with st.expander("Source from PDF"):
                        for doc in sources[:2]:
                            st.write(doc.page_content[:300] + "...")
                else:
                    response = groq_chat.invoke(prompt).content

            else:
                # -------- NORMAL CHAT --------
                response = groq_chat.invoke(prompt).content

    except Exception as e:
        response = f"Error: {str(e)}"

    # -------- DISPLAY --------
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
