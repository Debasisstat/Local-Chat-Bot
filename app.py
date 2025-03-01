import sys
sys.path.insert(0,"Chatbot")
from dotenv import load_dotenv
import streamlit as st
import os
import time
import nltk
from loaddoc import load_documents
from vectorstore import setup_vectorstore
from chain import create_chain


# Download NLP resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

working_dir = os.getcwd()

# --- 🎨 Streamlit Page Configuration ---
st.set_page_config(page_title="📑 AI Chatbot", page_icon="🤖", layout="wide")

# --- 🌟 Sidebar Customization ---
st.sidebar.markdown("## ⚙️ Settings")
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)
model_options = ["gemma", "llama2", "phi3"]
selected_model = st.sidebar.selectbox("🧠 Select Model", model_options, index=0)
temperature = st.sidebar.slider("🔥 Creativity Level", 0.0, 1.0, 0.5, 0.1)
custom_prompt = st.sidebar.text_area("💬 Custom System Prompt", "You are a helpful AI. Answer based on the uploaded document.")

# --- 🎨 Dark/Light Mode CSS ---
def set_theme(dark):
    theme_css = """
        <style>
        body { background-color: #1E1E2E; color: white; }
        .stChatMessage { border-radius: 10px; padding:10px; }
        .user { background: linear-gradient(135deg, #00B4DB, #0083B0); color: white; }
        .assistant { background: linear-gradient(135deg, #ff758c, #ff7eb3); color: white; }
        .sidebar .block-container { background-color: #333; color: white; }
        </style>
    """
    light_css = """
        <style>
        body { background-color: white; color: black; }
        .stChatMessage { border-radius: 10px; padding:10px; }
        .user { background: linear-gradient(135deg, #2196F3, #1E88E5); color: white; }
        .assistant { background: linear-gradient(135deg, #FF9800, #F57C00); color: white; }
        .sidebar .block-container { background-color: #f0f0f0; color: black; }
        </style>
    """
    st.markdown(theme_css if dark else light_css, unsafe_allow_html=True)

set_theme(dark_mode)

# --- 🏷️ App Header ---
st.markdown("<h1 style='text-align: center;'>🤖 Multi-Document Chatbot 📄</h1>", unsafe_allow_html=True)

# --- 📥 File Uploader ---
st.markdown("### 📂 Upload a Document (PDF, TXT, DOCX, CSV)")
uploaded_file = st.file_uploader("Drop a file here", type=["pdf", "txt", "docx", "csv"])



# --- 📊 Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

file_path = None

if uploaded_file:
    file_path = f"{working_dir}/{uploaded_file.name}"
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    # Show processing status
    with st.spinner("🔄 Processing Document... Please wait."):
        time.sleep(2)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display Progress Bar
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)

        # Set up vector store
        if "vectorstores" not in st.session_state:
            documents = load_documents(file_path, file_extension)
            st.session_state.vectorstores = setup_vectorstore(documents)

        # Update conversation chain
        if (
            "conversation_chain" not in st.session_state or
            st.session_state.model != selected_model or
            st.session_state.temperature != temperature or
            st.session_state.prompt != custom_prompt
        ):
            with st.spinner("🚀 Initializing AI Model..."):
                st.session_state.conversation_chain = create_chain(
                    st.session_state.vectorstores, selected_model, temperature, custom_prompt
                )
            st.session_state.model = selected_model
            st.session_state.temperature = temperature
            st.session_state.prompt = custom_prompt

# --- 🚀 Chat Interface ---
st.markdown("### 💬 Chat with AI")

# Display chat history
for message in st.session_state.chat_history:
    role = message["role"]
    color = "#00B4DB" if role == "user" else "#FF758C"
    st.markdown(f"""
        <div style="background-color:{color}; padding:10px; border-radius:10px; margin-bottom:10px;">
            <b>{role.capitalize()}:</b> {message["content"]}
        </div>
        """, unsafe_allow_html=True)

# Chat input box
user_input = st.chat_input("💡 Type your question here...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"💡 {user_input}")

    with st.spinner("🤖 Thinking..."):
        if "conversation_chain" in st.session_state:
            response = st.session_state.conversation_chain({"question": user_input})
            assistant_response = response["answer"]
        else:
            assistant_response = "⚠️ Please upload a document before asking questions."

    with st.chat_message("assistant"):
        st.markdown(f"🤖 {assistant_response}")

    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

# --- 📌 Footer ---
st.markdown("<hr><p style='text-align: center;'>✨ Created with ❤️ by Debasis 🚀</p>", unsafe_allow_html=True)
