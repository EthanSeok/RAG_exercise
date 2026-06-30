from pathlib import Path
import shutil

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


PDF_DIR = "docs"
DB_DIR = "chroma_db"

EMBED_MODEL = "embeddinggemma"
# 안 되면 아래로 변경
# EMBED_MODEL = "nomic-embed-text"


def main():
    if not Path(PDF_DIR).exists():
        raise FileNotFoundError(f"{PDF_DIR} 폴더가 없습니다.")

    if Path(DB_DIR).exists():
        shutil.rmtree(DB_DIR)

    loader = PyPDFDirectoryLoader(PDF_DIR)
    docs = loader.load()
    print(f"Loaded pages: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    chunks = [doc for doc in chunks if doc.page_content.strip()]

    print(f"Created chunks: {len(chunks)}")

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url="http://localhost:11434",
    )

    vectorstore = Chroma(
        collection_name="pdf_references",
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )

    batch_size = 8

    for idx in range(0, len(chunks), batch_size):
        batch = chunks[idx:idx + batch_size]
        print(f"Adding batch {idx // batch_size + 1}, size={len(batch)}")
        vectorstore.add_documents(batch)

    print(f"Saved vector DB to: {DB_DIR}")


if __name__ == "__main__":
    main()