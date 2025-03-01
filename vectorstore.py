from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters.character import CharacterTextSplitter


def setup_vectorstore(documents):
    """Creates a FAISS vector store."""
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(documents)
    vectorstores = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstores