
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
import os


def load_documents(file_path, file_type):
    """Loads documents from different file formats."""
    if file_type == "pdf":
        loader = UnstructuredPDFLoader(file_path, poppler_path="/usr/bin", tesseract_path="/usr/bin/tesseract")
    elif file_type == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_type == "docx":
        loader = Docx2txtLoader(file_path)
    elif file_type == "csv":
        loader = CSVLoader(file_path)
    else:
        st.error("⚠️ Unsupported file format!")
        return None
    return loader.load()
