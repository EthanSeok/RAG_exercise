# Gemma4 RAG API 서버

로컬에서 실행되는 **Gemma4:31b** 모델(Ollama)에 PDF 문서 검색(RAG)을 결합한 FastAPI 서버입니다.  
스마트팜 관련 PDF 문서를 기반으로 질문에 근거 인용과 함께 답변합니다.

## 시스템 구조

```
클라이언트 (HTTP POST)
        │
        ▼
FastAPI 서버 (api_RAG.py)
        │
   ┌────┴────┐
   ▼         ▼
ChromaDB      Ollama
(벡터 검색)   (Gemma4:31b 추론)
   ▲
embeddinggemma
(로컬 임베딩)
```

## 파일 구성

| 파일 | 역할 |
|------|------|
| `ingest.py` | PDF → 청킹 → 임베딩 → ChromaDB 저장 (최초 1회 실행) |
| `api_RAG.py` | RAG 기반 FastAPI 서버 (메인) |
| `api.py` | RAG 없이 Gemma4에 직접 질문하는 API (비교용) |
| `ask.py` | CLI에서 RAG 질문을 테스트하는 스크립트 |

## 요구사항

- Python 3.10+
- [Ollama](https://ollama.com/) 설치 및 실행
- 아래 모델 사전 설치:

```bash
ollama pull gemma4:31b         # LLM
ollama pull embeddinggemma     # 임베딩 모델 (또는 nomic-embed-text)
```

## 설치

```bash
pip install fastapi uvicorn langchain langchain-chroma langchain-ollama langchain-community pypdf
```

## 사용 방법

### 1단계: PDF 색인 (최초 1회)

`docs/` 폴더에 PDF 파일을 넣고 실행합니다.

```bash
python ingest.py
```

- PDF를 페이지 단위로 읽어 청크(chunk_size=900, overlap=150)로 분할
- `embeddinggemma` 모델로 임베딩 생성
- `chroma_db/` 폴더에 벡터 DB 저장

### 2단계: RAG API 서버 실행

```bash
python api_RAG.py
```

서버가 `http://0.0.0.0:8000` 에서 시작됩니다.

### 3단계: 질문

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{"prompt": "온도 관리가 작물 생육에 미치는 영향은?"}'
```

### CLI로 빠른 테스트

```bash
python ask.py "담배가루이 방제 방법을 알려줘"
```

## API 엔드포인트

### `POST /generate`

RAG 기반 질문 응답

**요청 헤더:**
```
x-api-key: YOUR_API_KEY
```

**요청 바디:**
```json
{
  "prompt": "질문 내용",
  "temperature": 0.1,
  "max_tokens": 2048,
  "top_k": 8,
  "fetch_k": 30,
  "debug": false
}
```

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `top_k` | 8 | 최종 반환할 문서 청크 수 |
| `fetch_k` | 30 | MMR 후보로 가져올 청크 수 |
| `debug` | false | `true`이면 검색된 원문 청크 포함 반환 |

**응답 예시 (debug=false):**
```json
{
  "response": "온도는 작물의 광합성 속도에 직접 영향을 미칩니다 [근거 1]. 최적 온도 범위를 벗어나면 효소 활성이 저하되어 [근거 2]..."
}
```

**응답 예시 (debug=true):**
```json
{
  "response": "...",
  "model": "gemma4:31b",
  "retrieved_count": 8,
  "rag_check": {
    "retrieved_docs": 8,
    "has_evidence_refs": true,
    "likely_rag_answer": true
  },
  "sources": [
    {
      "rank": 1,
      "source": "2017 작물관리_OCR.pdf",
      "page": 42,
      "preview": "..."
    }
  ]
}
```

### `GET /debug/search`

문서 검색 결과만 확인 (LLM 추론 없음)

```bash
curl "http://localhost:8000/debug/search?q=온도관리&top_k=5" \
  -H "x-api-key: YOUR_API_KEY"
```

### `GET /health`

서버 상태 확인

```bash
curl http://localhost:8000/health -H "x-api-key: YOUR_API_KEY"
```

## RAG 동작 방식

```
질문: "담배가루이 방제법은?"
    │
    ▼ OllamaEmbeddings (embeddinggemma)로 질문 임베딩
    │
    ▼ MMR 검색 (fetch_k=30 중 다양성 고려해 top_k=8 선택)
[근거1] [근거2] ... [근거8]  (source + page 메타데이터 포함)
    │
    ▼ ChatPromptTemplate에 context로 주입
    │
    ▼ ChatOllama (gemma4:31b, num_ctx=32768)
    │
답변 (근거 번호 [근거 1] [근거 2] 인용 포함)
```

**MMR (Maximal Marginal Relevance):**  
단순 유사도 검색과 달리 **다양성**을 고려해 중복 내용이 적은 청크를 선택합니다.  
`fetch_k`개를 후보로 가져온 뒤 `top_k`개로 줄입니다.

## 시스템 프롬프트 설계

모델이 문서 외 정보를 사용하지 못하도록 엄격히 제한합니다.

```
절대 규칙:
1. 반드시 제공된 context 안의 내용만 근거로 답변한다.
2. context에 없는 배경지식, 모델의 사전지식을 사용하지 않는다.
3. context에서 확인되지 않는 내용은 "문서에서 확인되지 않습니다"라고 답한다.
4. 답변에는 반드시 [근거 번호]를 인용한다.
5. 여러 근거가 충돌하면 충돌한다고 명시한다.
```

## api.py vs api_RAG.py 비교

| 항목 | `api.py` | `api_RAG.py` |
|------|----------|--------------|
| 방식 | Ollama 직접 호출 | RAG + Ollama |
| 문서 검색 | 없음 | ChromaDB MMR 검색 |
| 답변 근거 | 없음 | PDF 출처 + 페이지 인용 |
| 용도 | 빠른 테스트 | 문서 기반 정확한 답변 |

## 설정값 변경

`api_RAG.py` 상단에서 수정합니다.

```python
LLM_MODEL    = "gemma4:31b"       # 사용할 LLM 모델
EMBED_MODEL  = "embeddinggemma"   # 임베딩 모델 (nomic-embed-text로 대체 가능)
API_KEY      = "your-api-key"     # API 인증 키
```

## 주의사항

- **Ollama가 실행 중이어야** 합니다 (`ollama serve`).
- `ingest.py`를 실행하기 전에 `docs/` 폴더에 PDF가 있어야 합니다.
- PDF를 추가하거나 변경하면 `ingest.py`를 다시 실행해야 합니다 (기존 DB 삭제 후 재생성).
- `gemma4:31b`는 대용량 모델로 충분한 VRAM(권장 24GB+) 또는 RAM이 필요합니다.
