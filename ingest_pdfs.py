# ingest_pdfs.py
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

# Folder with PDFs
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

def ingest():
    documents = []
    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(DATA_DIR, filename))
            docs = loader.load()
            documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    db.persist()
    print(f"✅ Ingested {len(chunks)} chunks from {len(documents)} pages into ChromaDB")

if __name__ == "__main__":
    ingest()
