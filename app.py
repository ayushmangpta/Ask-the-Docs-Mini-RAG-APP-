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

# Maximum number of messages before auto-clearing chat
MAX_MESSAGES = 100

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

def check_message_limit():
    """Check if message limit is reached and clear chat if necessary"""
    if len(st.session_state.messages) >= MAX_MESSAGES:
        # Keep only the initial assistant message
        initial_message = st.session_state.messages[0] if st.session_state.messages else {
            "role": "assistant",
            "content": "Hello! I am your AI assistant. How can I help you today?",
        }
        st.session_state.messages = [initial_message]
        st.warning(f"Chat automatically cleared after reaching {MAX_MESSAGES} messages limit.")
        return True
    return False

# Register cleanup function to run on exit
atexit.register(lambda: cleanup_old_sessions())

# Function to validate Google API key
def validate_api_key(api_key):
    """Validate a Google API key by making a test request"""
    try:
        # Create a temporary LLM instance with the key to test
        test_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            max_tokens=10,  # Minimal tokens for quick validation
            google_api_key=api_key,
        )
        
        # Try a simple test message to validate the API key
        test_message = [HumanMessage(content="Test")]
        _ = test_llm.invoke(test_message)  # We only care if this succeeds
        return True, "API key is valid"
    except Exception as e:
        error_msg = str(e)
        # Check for common auth-related errors in the exception message
        if "auth" in error_msg.lower() or "api key" in error_msg.lower() or "apikey" in error_msg.lower() or "credential" in error_msg.lower() or "permission" in error_msg.lower():
            return False, "Invalid API key - Authentication failed"
        else:
            return False, f"API connection error: {error_msg}"

# Function to initialize LLM models with session-specific API key
def initialize_llm(api_key=None, streaming=False):
    # Use provided API key or fall back to environment variable
    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1,
        max_tokens=None if not streaming else None,
        timeout=None if not streaming else None,
        top_p=0.95 if not streaming else None,
        max_retries=2 if not streaming else 1,
        streaming=streaming,
        google_api_key=api_key,
    )

# Get default API key from environment
default_api_key = os.environ.get("GOOGLE_API_KEY")

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
    
    # Initialize session API key
    st.session_state.api_key = default_api_key
    st.session_state.api_key_set = default_api_key is not None

# Clean up old sessions periodically (every time a new session starts)
cleanup_old_sessions()

# Initialize LLM models with session API key
if "llm" not in st.session_state or "llm_stream" not in st.session_state:
    st.session_state.llm = initialize_llm(st.session_state.api_key)
    st.session_state.llm_stream = initialize_llm(st.session_state.api_key, streaming=True)

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
                with st.spinner("Validating API key..."):
                    is_valid, message = validate_api_key(new_api_key)
                    
                    if is_valid:
                        # Store API key in session state instead of environment variable
                        st.session_state.api_key = new_api_key
                        st.session_state.api_key_set = True
                        
                        # Re-initialize LLM instances with the new API key
                        st.session_state.llm = initialize_llm(st.session_state.api_key)
                        st.session_state.llm_stream = initialize_llm(st.session_state.api_key, streaming=True)
                        
                        st.success("API key updated successfully!")
                    else:
                        st.error(message)
                        st.session_state.api_key_set = False
    
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

# Display message count and limit
st.sidebar.markdown(f"**Messages:** {len(st.session_state.messages)}/{MAX_MESSAGES}")

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
                    rag_response = answer_with_rag(prompt, st.session_state.retriever, st.session_state.llm)
                    
                    # Format the message with the RAG response
                    rag_prompt = f"""
                    Based on the retrieved information: 
                    
                    {rag_response}
                    
                    Answer the following question: {prompt}
                    """
                    messages = [HumanMessage(content=rag_prompt)]
                    
                    # Use session-specific streaming LLM
                    for chunk in st.session_state.llm_stream.stream(messages):
                        if hasattr(chunk, 'content'):
                            content = chunk.content
                            full_response += content
                            message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Error using RAG: {str(e)}")
                # Fallback to regular response
                recent_messages = st.session_state.messages[-2:]
                messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in recent_messages]
                
                # Use session-specific streaming LLM
                for chunk in st.session_state.llm_stream.stream(messages):
                    if hasattr(chunk, 'content'):
                        content = chunk.content
                        full_response += content
                        message_placeholder.markdown(full_response)
        else:
            recent_messages = st.session_state.messages[-2:]
            messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in recent_messages]

            # Use session-specific streaming LLM
            for chunk in st.session_state.llm_stream.stream(messages):
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    full_response += content
                    message_placeholder.markdown(full_response)
        
        # After streaming completes, save the response to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Check if message limit is reached after adding the response
        check_message_limit()
