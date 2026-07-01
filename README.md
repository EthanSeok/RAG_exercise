### 2026-계절학기 인공지능및빅데이터 실습

# RAG Exercise — 온실 해충 분류 AI 시스템

온실 해충 이미지를 분류하고 PDF 문서 기반 방제 정보를 제공하는 두 가지 RAG 구현 사례입니다.  
**같은 목적, 다른 접근 방식**으로 LLM + RAG 아키텍처를 비교 학습할 수 있습니다.

---

## 두 구현체 비교

| | `claude_MCP_RAG` | `gemmar_RAG` |
|---|---|---|
| **LLM** | Claude (Anthropic API) | Gemma4:31b (Ollama, 로컬) |
| **인터페이스** | Claude Desktop 채팅 | FastAPI REST API |
| **연결 방식** | MCP 프로토콜 (stdio) | HTTP |
| **임베딩** | ChromaDB 기본 임베딩 | embeddinggemma (Ollama) |
| **검색 방식** | 코사인 유사도 | MMR (다양성 고려) |
| **이미지 분류** | ViT / CNN (PyTorch) | 없음 |
| **실행 환경** | 로컬 (Claude Desktop 연동) | 로컬 서버 (외부 접근 가능) |

---

## 공통 RAG 파이프라인

```
PDF 문서
  │
  ▼ 텍스트 추출 → 청킹 → 임베딩
벡터 DB (ChromaDB)
  │
  ▼ 질문과 유사한 청크 검색
LLM에 컨텍스트로 주입
  │
  ▼
근거 인용 포함 답변
```

---

## claude_MCP_RAG

Claude Desktop에서 자연어로 해충 분류와 문서 검색을 수행하는 **MCP 서버**.

**구조:**
```
Claude Desktop → MCP (stdio) → server.py → ViT/CNN 분류기
                                         → ChromaDB 검색
```

**핵심 파일:**

| 파일 | 역할 |
|------|------|
| `server.py` | MCP 서버 + 도구 3개 정의 |
| `classifier.py` | ViT / CNN 딥러닝 추론 |
| `rag.py` | PDF 청킹, 색인, 검색 |

**빠른 시작:**

```bash
uv sync
```

`%APPDATA%\Claude\claude_desktop_config.json` 에 추가:

```json
{
  "mcpServers": {
    "pest-classifier": {
      "command": "uv",
      "args": ["--directory", "C:\\path\\to\\claude_MCP_RAG", "run", "python", "server.py"],
      "env": { "PYTHONUTF8": "1" }
    }
  }
}
```

Claude Desktop 재시작 후 `uploads/` 폴더에 이미지를 저장하고:

```
uploads 폴더에 해충 이미지 저장했어. classify_pest 도구로 분류해줘.
```

---

## gemmar_RAG

로컬 Gemma4 모델에 RAG를 결합한 **FastAPI 서버**.

**구조:**
```
HTTP 클라이언트 → FastAPI → MMR 검색 (ChromaDB) → Gemma4:31b (Ollama)
```

**핵심 파일:**

| 파일 | 역할 |
|------|------|
| `ingest.py` | PDF 색인 (최초 1회) |
| `api_RAG.py` | RAG + FastAPI 서버 |
| `api.py` | RAG 없이 Gemma4 직접 호출 (비교용) |
| `ask.py` | CLI 테스트 |

**빠른 시작:**

```bash
# 1. 모델 준비
ollama pull gemma4:31b
ollama pull embeddinggemma

# 2. PDF 색인
python ingest.py

# 3. 서버 실행
python api_RAG.py

# 4. 질문
curl -X POST http://localhost:8000/generate \
  -H "x-api-key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "담배가루이 방제법은?"}'
```

---

## 프로젝트 구조

```
RAG_exercise/
├── claude_MCP_RAG/        # MCP + Claude + ChromaDB
│   ├── server.py
│   ├── classifier.py
│   ├── rag.py
│   ├── uploads/           # 분류할 이미지
│   └── reference/         # 해충 참고 PDF (직접 추가)
│
├── gemmar_RAG/            # FastAPI + Gemma4 + ChromaDB
│   ├── ingest.py
│   ├── api_RAG.py
│   ├── api.py
│   ├── ask.py
│   └── docs/              # RAG용 PDF (직접 추가)
│
└── pest_classification/   # 딥러닝 모델 학습 결과
    └── output/
        ├── vit-aug/vit_base_12.pth
        └── cnn-aug/cnn_base_44.pth
```
