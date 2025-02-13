import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt

# ✅ Set page config with custom colors and icons
st.set_page_config(
    page_title="Medical Chatbot",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "This is a Medical Chatbot designed to provide health-related information."
    }
)

# Custom CSS for styling
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stChatMessage.user {
        background-color: #e3f2fd;
        margin-left: auto;
        margin-right: 0;
        max-width: 70%;
    }
    .stChatMessage.assistant {
        background-color: #f5f5f5;
        margin-left: 0;
        margin-right: auto;
        max-width: 70%;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stTitle {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .stSubheader {
        font-size: 1.2rem;
        color: #34495e;
        margin-bottom: 2rem;
    }
    .stSpinner {
        color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = "pcsk_3tvNku_A9UeFeqpn3DqSuCKf1RJ8rpnDdJdm69bXowNF3pEwsUmbvfZZTLnVDdEMbsqWJE"
GROQ_API_KEY = "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"

# ✅ Cache embeddings to prevent reloading on every interaction (silent loading)
@st.cache_resource
def load_embeddings():
    return download_hugging_face_embeddings()

# ✅ Cache Pinecone retriever (silent loading)
@st.cache_resource
def load_retriever():
    embeddings = load_embeddings()
    index_name = "medicalbot"
    docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    return docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ✅ Cache LLM model (silent loading)
@st.cache_resource
def load_llm():
    return ChatGroq(
        temperature=0.4,
        max_tokens=500,
        model_name="llama-3.3-70b-versatile"
    )

# ✅ Track initialization state
if "initialized" not in st.session_state:
    st.session_state["retriever"] = load_retriever()
    st.session_state["llm"] = load_llm()
    st.session_state["initialized"] = True

# Create RAG chain
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(st.session_state["llm"], prompt)
rag_chain = create_retrieval_chain(st.session_state["retriever"], question_answer_chain)

# Title and Subheader
st.markdown('<div class="stTitle">🩺 AI Powered Health Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="stSubheader">Ask me anything about health!</div>', unsafe_allow_html=True)

# Initialize chat messages in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        response = rag_chain.invoke({"input": user_input})
        bot_response = response["answer"]

    st.session_state["messages"].append({"role": "assistant", "content": bot_response})
    
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(bot_response)