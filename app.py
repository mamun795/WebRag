import os
import asyncio

import streamlit as st
from dotenv import load_dotenv

# .env load + USER_AGENT must be set BEFORE importing langchain_community loaders
load_dotenv()
os.environ.setdefault("USER_AGENT", "m795-chatbot")

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# -----------------------------
# Settings (change here if needed)
# -----------------------------
CHAT_MODEL = "gemini-2.5-flash"

# Local embedding model (runs on your PC, no API key, no rate limit).
# English websites : "BAAI/bge-small-en-v1.5"        (~130 MB, fast)
# Other languages  : "intfloat/multilingual-e5-large" (~2 GB, slower on CPU)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_CACHE_DIR = "./fastembed_cache"   # model is downloaded once and kept here

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_HISTORY_MESSAGES = 16  # last 8 turns (8 user + 8 AI messages)


def is_rate_limit_error(error):
    text = str(error)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower()


@st.cache_resource(show_spinner="Loading embedding model (first time downloads it)...")
def get_embeddings():
    return FastEmbedEmbeddings(model_name=EMBED_MODEL, cache_dir=EMBED_CACHE_DIR)


# -----------------------------
# Vector Store Creation
# -----------------------------
def get_vector_store_from_url(url):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    loader = WebBaseLoader(url)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No readable text found on this page.")

    return FAISS.from_documents(chunks, get_embeddings())


# -----------------------------
# Conversational RAG Chain
# -----------------------------
def get_conversational_rag_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are **M795 AI chatbot**, developed by **Mamun Ahmed**. "
         "He is a 4th-year, 2nd-semester student of the ICT Department at MBSTU. "
         "When asked 'who are you' or 'who developed you', answer only with your name and developer identity in **bold**. "
         "Otherwise, just answer the query normally. "
         "Answer based on the context below:\n\n{context}\n\n"
         "Answer in the same language the user used. "
         "Avoid grammar mistakes, sometimes bold important words."
         ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    retriever = vector_store.as_retriever()
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, stuff_documents_chain)


# -----------------------------
# Response Function (Streaming)
# -----------------------------
def get_response(user_input, placeholder):
    history = st.session_state.chat_History[-MAX_HISTORY_MESSAGES:]

    response_text = ""
    for chunk in st.session_state.conversation_rag_chain.stream({
        "chat_history": history,
        "input": user_input,
    }):
        if "answer" in chunk:
            response_text += chunk["answer"]
            placeholder.write(response_text)  # live update

    return response_text


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Chat with website", page_icon="🤖")
st.title("Chat with website")

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL").strip()

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found. Add it to your .env file: GOOGLE_API_KEY=your_key")
    st.stop()

if not website_url:
    st.info("Please enter a website URL")
    st.stop()

# Build everything again only when the URL changes
if st.session_state.get("current_url") != website_url:
    try:
        with st.spinner("Reading website and building search index..."):
            vector_store = get_vector_store_from_url(website_url)
        st.session_state.vector_store = vector_store
        st.session_state.conversation_rag_chain = get_conversational_rag_chain(vector_store)
        st.session_state.chat_History = [
            AIMessage(content="Hello, I am **M795 AI chatbot**. How can I help you?")
        ]
        st.session_state.current_url = website_url
    except Exception as e:
        st.error(f"Could not load this website: {e}")
        st.stop()

# ----------- show conversation -----------
for message in st.session_state.chat_History:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# ----------- user input -----------
user_query = st.chat_input("Type your message here...")

if user_query:
    with st.chat_message("Human"):
        st.write(user_query)

    with st.chat_message("AI"):
        live_placeholder = st.empty()

    try:
        response = get_response(user_query, live_placeholder)
    except Exception as e:
        if is_rate_limit_error(e):
            st.error("Google API rate limit reached. Please wait a minute and ask again.")
        else:
            st.error(f"Error while generating the answer: {e}")
        st.stop()

    # Save history
    st.session_state.chat_History.append(HumanMessage(content=user_query))
    st.session_state.chat_History.append(AIMessage(content=response))