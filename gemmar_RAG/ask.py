import sys

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate


DB_DIR = "chroma_db"

LLM_MODEL = "gemma4:31b"
EMBED_MODEL = "embeddinggemma"

SYSTEM_PROMPT = """
너는 PDF 레퍼런스 기반 연구 보조 AI다.

규칙:
1. 반드시 제공된 context에 근거해서 답변한다.
2. context에 없는 내용은 추측하지 말고 "문서에서 확인되지 않습니다"라고 말한다.
3. 여러 문서의 근거가 충돌하면 충돌한다고 명시한다.
4. 답변 마지막에 사용한 출처를 파일명과 페이지 번호로 정리한다.
5. 질문이 추론을 요구하면, 근거 문장들을 연결해서 논리적으로 결론을 도출한다.
6. 한국어로 답변한다.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """
질문:
{question}

검색된 PDF context:
{context}

위 context만 근거로 답변하라.
""",
        ),
    ]
)


def format_docs(docs):
    formatted = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")

        formatted.append(
            f"[근거 {i}]\n"
            f"source: {source}\n"
            f"page: {page}\n"
            f"content:\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(formatted)


def main():
    if len(sys.argv) < 2:
        print("Usage: python ask.py '질문을 입력하세요'")
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url="http://localhost:11434",
    )

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="pdf_references",
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 30,
        },
    )

    docs = retriever.invoke(question)
    context = format_docs(docs)

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url="http://localhost:11434",
        temperature=0.1,
        num_ctx=32768,
    )

    chain = PROMPT | llm

    response = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    print(response.content)


if __name__ == "__main__":
    main()