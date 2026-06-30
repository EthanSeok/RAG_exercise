import glob
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

REFERENCE_DIR = Path(__file__).parent / "reference"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "pest_reference"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is not None:
        return _collection

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    ef = embedding_functions.DefaultEmbeddingFunction()
    col = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

    if col.count() == 0:
        _index_pdfs(col)

    _collection = col
    return _collection


def _extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


def _index_pdfs(collection: chromadb.Collection) -> None:
    # use set to deduplicate (Windows FS is case-insensitive, so *.pdf and *.PDF can match the same file)
    seen = set()
    pdf_files = []
    for pattern in ("*.pdf", "*.PDF"):
        for p in glob.glob(str(REFERENCE_DIR / "**" / pattern), recursive=True):
            key = str(Path(p).resolve()).lower()
            if key not in seen:
                seen.add(key)
                pdf_files.append(p)

    documents, metadatas, ids = [], [], []
    for pdf_path in pdf_files:
        text = _extract_pdf_text(pdf_path)
        filename = Path(pdf_path).name
        for i, chunk in enumerate(_chunk_text(text)):
            documents.append(chunk)
            metadatas.append({"source": filename, "chunk_index": i})
            ids.append(f"{filename}__{i}")

    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)


def search_reference(query: str, top_k: int = 3) -> str:
    """Search indexed PDF documents and return the most relevant passages."""
    col = _get_collection()
    if col.count() == 0:
        return "참고 문서가 색인되지 않았습니다. reference 폴더에 PDF 파일을 추가하세요."

    n = min(top_k, col.count())
    results = col.query(query_texts=[query], n_results=n)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    if not docs:
        return "관련 정보를 찾을 수 없습니다."

    sections = [f"[출처: {m['source']}]\n{d}" for d, m in zip(docs, metas)]
    return "\n\n---\n\n".join(sections)


def reindex(force: bool = False) -> str:
    """Re-index all PDFs in the reference folder. Use force=True to rebuild from scratch."""
    global _collection
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    ef = embedding_functions.DefaultEmbeddingFunction()

    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    col = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)
    _index_pdfs(col)
    _collection = col
    return f"색인 완료: {col.count()}개 청크"
