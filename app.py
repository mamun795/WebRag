import asyncio
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# -----------------------------
# Vector Store Creation
# -----------------------------
def get_vector_store_from_url(url):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    document_chunk = text_splitter.split_documents(document)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_documents(document_chunk, embeddings)
    return vector_store

# -----------------------------
# Conversational RAG Chain
# -----------------------------
def get_conversational_rag_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", streaming=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are **M795 AI chatbot**, developed by **Mamun Ahmed** "
         "(4th year, 1st Semester, ICT, MBSTU). "
         "When asked 'who are you' or 'who developed you', answer only with your name and developer identity in **bold**. "
         "Otherwise, just answer the query normally. "
         "Answer based on the context below:\n\n{context}\n\n"
         "If out of context → reply briefly and mention it's out of context. "
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
def get_response(user_input):
    # Truncate history (last 8 turns only for speed)
    history = st.session_state.chat_History[-8:]

    # Stream response
    response_text = ""
    for chunk in st.session_state.conversation_rag_chain.stream({
        "chat_history": history,
        "input": user_input
    }):
        if "answer" in chunk:
            response_text += chunk["answer"]
            st.session_state.live_placeholder.write(response_text)  # live update

    return response_text

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Chat with website", page_icon="🤖")
st.title("Chat with website")

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if not website_url:
    st.info("Please enter a website URL")
else:
    if "chat_History" not in st.session_state:
        st.session_state.chat_History = [
            AIMessage(content="Hello, I am **M795 AI chatbot**. How can I help you?")
        ]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_store_from_url(website_url)

    if "conversation_rag_chain" not in st.session_state:
        st.session_state.conversation_rag_chain = get_conversational_rag_chain(st.session_state.vector_store)
    

#-----------show conversation-----------

    for message in st.session_state.chat_History:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)


    # User Input
    user_query = st.chat_input("Type your message here...")

    if user_query:
        with st.chat_message("Human"):
            st.write(user_query)

        with st.chat_message("AI"):
            st.session_state.live_placeholder = st.empty()
            response = get_response(user_query)

        # Save history
        st.session_state.chat_History.append(HumanMessage(content=user_query))
        st.session_state.chat_History.append(AIMessage(content=response))

    # Show Conversation
   # for message in st.session_state.chat_History:
      #  if isinstance(message, AIMessage):
        #    with st.chat_message("AI"):
            #    st.write(message.content)
        #elif isinstance(message, HumanMessage):
         #   with st.chat_message("Human"):
            #    st.write(message.content)
