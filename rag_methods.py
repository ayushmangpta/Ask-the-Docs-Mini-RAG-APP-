import streamlit as st
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

def stream_llm_response(llm_stream, messages):
    """
    Stream the response from the LLM.
    """
    # Initialize the response variable
    full_response = ""
    
    # Stream the response
    for chunk in llm_stream.stream(messages):
        full_response += chunk.content
        yield full_response
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- RAG Backend with FAISS ---

def load_documents(file_paths, links):
    """
    Load documents from PDF, TXT files and web links.
    """
    docs = []
    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext == ".txt":
            loader = TextLoader(path)
        else:
            continue
        docs.extend(loader.load())
    for link in links:
        loader = WebBaseLoader(link)
        docs.extend(loader.load())
    return docs

def create_faiss_index(documents):
    """Create a FAISS index from a list of documents"""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    # Use Google's embeddings to avoid compatibility issues with HuggingFace
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Convert documents to embeddings and create FAISS index
    faiss_index = FAISS.from_documents(documents, embeddings)
    return faiss_index

def get_retriever(faiss_index):
    """
    Get retriever from FAISS index.
    """
    return faiss_index.as_retriever()

def answer_with_rag(query, retriever, llm=None):
    """
    Answer a query using RAG (retrieval-augmented generation).
    """
    if llm is None:
        llm = OpenAI(temperature=0)
    
    # Get relevant documents
    relevant_docs = retriever.get_relevant_documents(query)
    
    # Format the retrieved context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # For Gemini model, it's often better to work with the context directly
    if isinstance(llm, ChatGoogleGenerativeAI):
        return context
    else:
        # For other models, use the standard RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
        return qa_chain.run(query)

def save_faiss_index(faiss_index, index_dir="faiss_index"):
    """
    Save FAISS index to disk.
    
    Args:
        faiss_index: FAISS index object
        index_dir: Directory to save the index
    """
    # Create directory if it doesn't exist
    os.makedirs(index_dir, exist_ok=True)
    
    # Save the index
    faiss_index.save_local(index_dir)
    return index_dir

def load_faiss_index(directory="faiss_index"):
    """
    Load FAISS index from disk.
    
    Args:
        directory: Directory where the index is stored
    
    Returns:
        FAISS index object
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    if not os.path.exists(directory):
        return None
    
    # Use the same embeddings model as in create_faiss_index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_index = FAISS.load_local(directory, embeddings)
    return faiss_index
