import re
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate


BASE_DIR = Path(__file__).resolve().parent
DB_DIR = str(BASE_DIR / "chroma_db")

LLM_MODEL = "gemma4:31b"
EMBED_MODEL = "embeddinggemma"
COLLECTION_NAME = "pdf_references"

API_HOST = "0.0.0.0"
API_PORT = 8000
API_KEY = "Smartfarm208!"

app = FastAPI(title="Gemma4 RAG API")


class PromptRequest(BaseModel):
    prompt: str
    temperature: float = 0.1
    max_tokens: int = 2048
    top_k: int = 8
    fetch_k: int = 30
    debug: bool = False


SYSTEM_PROMPT = """
너는 PDF 레퍼런스 기반 연구 보조 AI다.

절대 규칙:
1. 반드시 제공된 context 안의 내용만 근거로 답변한다.
2. context에 없는 배경지식, 일반상식, 모델의 사전지식을 사용하지 않는다.
3. context에서 직접 확인되지 않는 내용은 반드시 "문서에서 확인되지 않습니다"라고 답한다.
4. 답변에는 반드시 [근거 번호]를 인용한다.
5. 답변 마지막에는 사용한 출처를 source와 page로 정리한다.
6. 여러 근거가 충돌하면 충돌한다고 명시한다.
7. 한국어로 답변한다.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """
질문:
{question}

<context>
{context}
</context>

위 context만 근거로 답변하라.
각 핵심 주장 뒤에는 반드시 [근거 번호]를 붙여라.
""",
        ),
    ]
)


if not Path(DB_DIR).exists():
    raise RuntimeError(f"chroma_db not found: {DB_DIR}")

embeddings = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url="http://localhost:11434",
)

vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)


def verify_api_key(x_api_key: str | None):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[근거 {i}]\n"
        f"source: {doc.metadata.get('source', 'unknown')}\n"
        f"page: {doc.metadata.get('page', 'unknown')}\n"
        f"content:\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )


def build_sources(docs):
    return [
        {
            "rank": i,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown"),
            "preview": doc.page_content[:300],
        }
        for i, doc in enumerate(docs, start=1)
    ]


def rag_check(text, docs, context):
    refs = re.findall(r"\[근거\s*\d+\]", text)
    return {
        "retrieved_docs": len(docs),
        "context_chars": len(context),
        "has_context": bool(context.strip()),
        "has_evidence_refs": bool(refs),
        "evidence_refs": refs,
        "likely_rag_answer": bool(docs) and bool(context.strip()) and bool(refs),
    }


def get_retriever(top_k, fetch_k):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": fetch_k,
        },
    )


@app.post("/generate")
def generate(req: PromptRequest, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)

    docs = get_retriever(req.top_k, req.fetch_k).invoke(req.prompt)
    context = format_docs(docs)

    if not docs or not context.strip():
        result = {"response": "문서에서 확인되지 않습니다."}

        if req.debug:
            result.update(
                {
                    "retrieved_count": 0,
                    "sources": [],
                    "rag_check": rag_check("", [], ""),
                }
            )

        return result

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url="http://localhost:11434",
        temperature=req.temperature,
        num_ctx=32768,
        num_predict=req.max_tokens,
    )

    response = (PROMPT | llm).invoke(
        {
            "question": req.prompt,
            "context": context,
        }
    )

    result = {"response": response.content}

    if req.debug:
        result.update(
            {
                "model": LLM_MODEL,
                "embedding_model": EMBED_MODEL,
                "question": req.prompt,
                "retrieved_count": len(docs),
                "rag_check": rag_check(response.content, docs, context),
                "sources": build_sources(docs),
                "context": context,
            }
        )

    return result


@app.get("/debug/search")
def debug_search(
    q: str,
    top_k: int = 8,
    fetch_k: int = 30,
    x_api_key: str | None = Header(default=None),
):
    verify_api_key(x_api_key)

    docs = get_retriever(top_k, fetch_k).invoke(q)

    return {
        "query": q,
        "retrieved_count": len(docs),
        "sources": build_sources(docs),
    }


@app.get("/health")
def health(x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)

    return {
        "status": "ok",
        "mode": "rag",
        "model": LLM_MODEL,
        "embedding_model": EMBED_MODEL,
        "db_exists": Path(DB_DIR).exists(),
        "collection": COLLECTION_NAME,
        "file": str(Path(__file__).resolve()),
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        reload=False,
    )