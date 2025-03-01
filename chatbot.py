from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Streamlit UI Configuration
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Choose Model", ["gemma", "phi3", "llama2"], index=0)
    temperature = st.sidebar.slider("🔥 Creativity Level", 0.0, 1.0, 0.5, 0.1)
    st.markdown("---")
    st.write("🔹 **Powered by LangChain & Ollama**")
    st.write("🔹 **Runs Locally for Privacy**")

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Page Title
st.title("🤖 AI Chatbot with Local Models")
st.markdown("**Ask any question, and the AI will respond!**")

# User Input
input_text = st.text_input("💬 Type your question below:", placeholder="e.g., Explain Quantum Computing")

# Define the Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Provide clear and informative answers."),
    ("user", "Question: {question}")
])

# Initialize LLM (Ollama)
llm = Ollama(model=model_choice, temperature=temperature)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Display Chat History
st.markdown("### 📝 Chat History")
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["response"])

# Handle Response
if input_text:
    with st.spinner("Thinking... 🤔"):
        response = chain.invoke({"question": input_text})

        # Store in session state
        st.session_state.chat_history.append({"question": input_text, "response": response})

        # Display response
        with st.chat_message("user"):
            st.write(input_text)
        with st.chat_message("assistant"):
            st.write(response)

# Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    # st.experimental_rerun()
    st.rerun()

# Footer
st.markdown("---")
st.write("📌 *This chatbot runs locally using open-source models!*")
