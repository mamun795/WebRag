import asyncio
import streamlit as st
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain




load_dotenv()
def get_vector_store_from_url(url):   

   try:
        asyncio.get_running_loop()
   except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

   loader=WebBaseLoader(url)
   document=loader.load()
   text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
   document_chunk=text_splitter.split_documents(document)
   embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
   vector_store=FAISS.from_documents(document_chunk,embeddings)
   return vector_store


def get_context_retrival_chain(vector_store):
   llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

   retriever = vector_store.as_retriever()

   prompts=ChatPromptTemplate.from_messages([
      ("system", ""),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user","{input}"),
      ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. ")
   ])

   retriever_chain = create_history_aware_retriever(llm, retriever, prompts)

    
   return retriever_chain


def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    prompt = ChatPromptTemplate.from_messages([
       ("system", "You are M795 AI chatbot. You are developed by Mamun Ahmed who reads in 4th year 1st Semester at ICT department in MBSTU. "
               " Mention that you are developed by Mamun Ahmed when asked about who developed you then mention you are M795 AI chatbot with your developer identity when asked who  you are and give your name and your developer name as bold and when a user asked who are you or who is developed you or something like that then you will only say your name and your developer name otherwise no need to say who you are and who is your developer "
               "Answer the user's questions based on the below context:\n\n{context}"
               " when you  give the answer of following question try to give example for better understanding to user if user query is out of the context then give the answer very shortly with mentioning it is out of the context and try to answer in the same language that the user asks the question in and try to avoid grammatically mistake and sometimes bold the answer"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)





def get_response(user_input):
   retriever_chain = get_context_retrival_chain(st.session_state.vector_store)
   conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
   response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_History,
        "input": user_input
    })
    
   return response['answer']

# app config
st.set_page_config(page_title="Chat with website",page_icon="robot")
st.title("Chat with website")

# sidebar
with st.sidebar:
    st.header("settings")
    websit_url=st.text_input("website url")


if websit_url is None or websit_url=="":
   st.info("Please enter a website URL")

else:

   if "chat_History" not in st.session_state:
        st.session_state.chat_History = [
            AIMessage(content="Hello, I am  M795 AI chatbot. How can I help you?"),
        ]

   if "vector_store" not in st.session_state:
        st.session_state.vector_store =get_vector_store_from_url(websit_url)

   #vector_store=get_vector_store_from_url(websit_url)
   #retriever_chain=get_context_retrival_chain(vector_store)

   #User Input
   user_query=st.chat_input("Type youre message here...")

   if user_query is not None and user_query !="":
      
      response=get_response(user_query)
      st.session_state.chat_History.append(HumanMessage(content=user_query))
      st.session_state.chat_History.append(AIMessage(content=response))



   # conversitaion

   for message in st.session_state.chat_History:
      if isinstance(message,AIMessage):
         with st.chat_message("AI"):
            st.write(message.content)
      elif isinstance(message,HumanMessage):
         with st.chat_message("Human"):
          st.write(message.content)





