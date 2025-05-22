import streamlit as st
import os
import dotenv
import uuid
import getpass
import datetime
import shutil
import atexit
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from rag_methods import stream_llm_response, load_documents, create_faiss_index, get_retriever, answer_with_rag, save_faiss_index, load_faiss_index

dotenv.load_dotenv()

# Create a base temp directory for all user sessions
BASE_TEMP_DIR = os.path.join(os.getcwd(), "temp_user_data")
os.makedirs(BASE_TEMP_DIR, exist_ok=True)

# Function to clean up old session directories
def cleanup_old_sessions(max_age_hours=24):
    """Clean up session directories older than the specified hours"""
    if os.path.exists(BASE_TEMP_DIR):
        current_time = datetime.datetime.now()
        for session_dir in os.listdir(BASE_TEMP_DIR):
            session_path = os.path.join(BASE_TEMP_DIR, session_dir)
            if os.path.isdir(session_path):
                # Get the directory's last modification time
                mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(session_path))
                # If older than max_age_hours, delete it
                if (current_time - mod_time).total_seconds() > max_age_hours * 3600:
                    try:
                        shutil.rmtree(session_path)
                        print(f"Cleaned up old session: {session_dir}")
                    except Exception as e:
                        print(f"Error cleaning up session {session_dir}: {str(e)}")

# Function to clean up specific session
def cleanup_session(session_id):
    """Clean up a specific session directory"""
    session_dir = os.path.join(BASE_TEMP_DIR, session_id)
    if os.path.exists(session_dir):
        try:
            shutil.rmtree(session_dir)
            print(f"Cleaned up session: {session_id}")
        except Exception as e:
            print(f"Error cleaning up session {session_id}: {str(e)}")

# Register cleanup function to run on exit
atexit.register(lambda: cleanup_old_sessions())

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    top_p=0.95,
    max_retries=2,
)

st.set_page_config(
    page_title="Ask the Docs",
    page_icon=":robot:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.html("""<h2 style="text-align: center;">Ask the Docs</h2>""")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    # Create session-specific directory
    st.session_state.user_dir = os.path.join(BASE_TEMP_DIR, st.session_state.session_id)
    os.makedirs(st.session_state.user_dir, exist_ok=True)
    
    # Session-specific FAISS index directory
    st.session_state.faiss_index_dir = os.path.join(st.session_state.user_dir, "faiss_index")

# Clean up old sessions periodically (every time a new session starts)
cleanup_old_sessions()

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! I am your AI assistant. How can I help you today?",
    }]

if "use_rag" not in st.session_state:
    st.session_state.use_rag = False

if "question_history" not in st.session_state:
    st.session_state.question_history = []

# Try to load existing session-specific FAISS index if available
if os.path.exists(st.session_state.faiss_index_dir):
    try:
        faiss_index = load_faiss_index(st.session_state.faiss_index_dir)
        if faiss_index:
            st.session_state.retriever = get_retriever(faiss_index)
    except Exception as e:
        print(f"Error loading FAISS index: {str(e)}")

with st.sidebar:
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["Settings", "RAG", "History"])
    
    with tab1:
        st.markdown("### API Settings")
        
        # Initialize key status in session state if not present
        if "api_key_set" not in st.session_state:
            st.session_state.api_key_set = os.getenv("GOOGLE_API_KEY") is not None
        
        # Show API key status instead of the actual key
        st.write("API Key Status: " + ("✅ Set" if st.session_state.api_key_set else "❌ Not Set"))
        
        # Input for new API key (never pre-filled)
        with st.popover("gemini-2.0-flash"):
            new_api_key = st.text_input(
                "Enter your Google API key",
                value="",
                type="password",
                placeholder="Enter your Google API key"
            )
            
            # Only update the API key if a new one is provided
            if new_api_key:
                os.environ["GOOGLE_API_KEY"] = new_api_key
                st.session_state.api_key_set = True
                st.success("API key updated successfully!")
    
    with tab2:
        # Add RAG toggle switch
        st.markdown("### RAG Settings")
        st.session_state.use_rag = st.toggle("Enable RAG", value=st.session_state.use_rag, help="Toggle to enable Retrieval Augmented Generation")
        
        # File upload section (only shown when RAG is enabled)
        if st.session_state.use_rag:
            st.markdown("### Document Upload")
            uploaded_files = st.file_uploader(
                "Upload PDF or TXT files", 
                accept_multiple_files=True,
                type=["pdf", "txt"]
            )
            
            # Display uploaded file count
            if uploaded_files:
                st.info(f"{len(uploaded_files)} files selected for upload")
                
            st.markdown("### Web Links")
            web_links = st.text_area(
                "Enter web links (one per line)",
                height=100,
                help="Enter URLs to web pages you want to include in your knowledge base"
            )
            
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    # Save uploaded files to session-specific directory
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(st.session_state.user_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(file_path)
                    
                    # Process web links
                    links = [link.strip() for link in web_links.split("\n") if link.strip()]
                    
                    # Load documents and create FAISS index
                    if file_paths or links:
                        docs = load_documents(file_paths, links)
                        faiss_index = create_faiss_index(docs)
                        st.session_state.retriever = get_retriever(faiss_index)
                        
                        # Save the FAISS index to session-specific directory
                        save_dir = save_faiss_index(faiss_index, index_dir=st.session_state.faiss_index_dir)
                        st.success(f"Successfully processed {len(file_paths)} files and {len(links)} web links.")
                    else:
                        st.warning("No documents or links provided")
    
    with tab3:
        st.markdown("### Question History")
        if st.session_state.question_history:
            for timestamp, question in st.session_state.question_history:
                st.text(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                st.info(question)
        else:
            st.text("No questions asked yet.")
    
    cols0=st.columns(2)
    with cols0[0]:
        # Add logout button that will clear user data
        if st.button("Delete Session", help="Log out and delete all your uploaded documents", type="secondary"):
            cleanup_session(st.session_state.session_id)
            st.session_state.clear()
            st.rerun()
            
    with cols0[1]:
        st.button(
            "Clear Chat",
            key="clear",
            help="Clear the chat history",
            on_click=lambda: [st.session_state.messages.clear(), st.session_state.question_history.clear()],
            type="primary"
        )

modelprovider = st.session_state.get("modelprovider", "Google")
if modelprovider == "Google":
    llm_stream = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1,
        streaming=True
    )
else:
    llm_stream = None

for messages in st.session_state.messages:
    with st.chat_message(messages["role"]):
        st.markdown(messages["content"])

# Replace the streaming implementation to fix the duplicated words issue
if prompt := st.chat_input("Ask a question about your documents"):
    # Log the question with timestamp
    st.session_state.question_history.append((datetime.datetime.now(), prompt))
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Use RAG if enabled and retriever exists
        if st.session_state.use_rag and 'retriever' in st.session_state:
            try:
                with st.spinner("Searching documents for relevant information..."):
                    rag_response = answer_with_rag(prompt, st.session_state.retriever, llm)
                    
                    # Format the message with the RAG response
                    rag_prompt = f"""
                    Based on the retrieved information: 
                    
                    {rag_response}
                    
                    Answer the following question: {prompt}
                    """
                    messages = [HumanMessage(content=rag_prompt)]
                    
                    # Improved streaming to prevent duplicated words
                    for chunk in llm_stream.stream(messages):
                        if hasattr(chunk, 'content'):
                            content = chunk.content
                            full_response += content
                            message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Error using RAG: {str(e)}")
                # Fallback to regular response
                recent_messages = st.session_state.messages[-2:]
                messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in recent_messages]
                
                # Improved streaming implementation
                for chunk in llm_stream.stream(messages):
                    if hasattr(chunk, 'content'):
                        content = chunk.content
                        full_response += content
                        message_placeholder.markdown(full_response)
        else:
            recent_messages = st.session_state.messages[-2:]
            messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in recent_messages]

            # Improved streaming implementation
            for chunk in llm_stream.stream(messages):
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    full_response += content
                    message_placeholder.markdown(full_response)
        
        # After streaming completes, save the response to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})
