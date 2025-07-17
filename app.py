import streamlit as st
import os
import time
import asyncio
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Set API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Streamlit App UI
st.title("📄🔍 Gemma (LLaMA3) Document Q&A with FAISS + Google Embeddings")

# Initialize the LLM (Groq)
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

# Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.

<context>
{context}
</context>

Question: {input}
""")

# Vector Embedding Setup Function
def vector_embedding():
    if "vectors" not in st.session_state:
        # Fix: Set an event loop manually for the current thread
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        # Load Embeddings and Documents
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./data")
        st.session_state.docs = st.session_state.loader.load()

        # Text Splitting
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])

        # Vector Store (FAISS)
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )

# UI for user question
prompt1 = st.text_input("❓ Enter your question about the documents")

# Button to trigger document ingestion and vector embedding
if st.button("📥 Build Vector Store from PDFs"):
    vector_embedding()
    st.success("✅ Vector Store is ready!")

# Handle Q&A if a question is provided
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please embed the documents first by clicking the button above.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Measure time
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        elapsed = time.process_time() - start

        # Show response
        st.subheader("🧠 Answer")
        st.write(response['answer'])
        st.caption(f"⏱️ Response generated in {elapsed:.2f} seconds")

        # Show matching document chunks
        with st.expander("📄 Document Chunks Used"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.write("---")
