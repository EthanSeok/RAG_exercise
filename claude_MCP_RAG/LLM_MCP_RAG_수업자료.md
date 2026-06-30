# LLM + MCP + RAG 실전 구축 수업자료
## 온실 해충 분류 시스템으로 배우는 AI 에이전트 아키텍처

---

## 1. 전체 시스템 개요

이 프로젝트는 세 가지 핵심 기술을 결합한 **AI 에이전트 시스템**입니다.

```
사용자 (Claude Desktop)
        │
        │ 자연어 질문 + 해충 이미지
        ▼
┌───────────────────┐
│   Claude (LLM)    │  ← 대화 이해, 도구 호출 결정, 답변 생성
└────────┬──────────┘
         │ MCP 프로토콜 (JSON-RPC over stdio)
         ▼
┌───────────────────┐
│   MCP 서버        │  ← 로컬 머신에서 실행되는 Python 프로세스
│   (server.py)     │
└────────┬──────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌───────┐
│ 딥러닝 │  │  RAG  │
│ 분류기 │  │ 검색기 │
│(ViT/  │  │(ChromaDB│
│  CNN) │  │+ PDF) │
└───────┘  └───────┘
```

| 기술 | 역할 | 이 프로젝트에서 |
|------|------|----------------|
| **LLM** | 자연어 이해 + 추론 + 답변 생성 | Claude (claude-sonnet-4-6) |
| **MCP** | LLM과 외부 도구를 연결하는 표준 프로토콜 | FastMCP 라이브러리 |
| **RAG** | 외부 문서를 검색해 LLM에 컨텍스트 제공 | ChromaDB + PDF 해충 자료 |

---

## 2. 핵심 개념 설명

### 2-1. LLM (Large Language Model)

LLM은 텍스트를 이해하고 생성하는 언어 모델입니다.

**이 시스템에서 Claude가 하는 일:**
- 사용자의 자연어 요청을 이해한다
- 어떤 도구(tool)를 호출할지 결정한다
- 도구에서 받은 결과를 바탕으로 최종 답변을 작성한다

**LLM의 한계 (이 프로젝트에서 직접 확인한 것):**
- 로컬 파일에 직접 접근할 수 없다
- 채팅에 첨부된 이미지를 도구에 바이트로 전달할 수 없다
- 학습 데이터에 없는 최신/전문 정보는 모른다 → RAG로 보완

---

### 2-2. MCP (Model Context Protocol)

Anthropic이 2024년에 발표한 오픈 표준 프로토콜입니다.  
LLM이 외부 시스템(파일, DB, API 등)을 **표준화된 방식**으로 사용할 수 있게 합니다.

**MCP 이전:**
```
Claude ──(제각각 API)──► 파일시스템
Claude ──(제각각 API)──► 데이터베이스
Claude ──(제각각 API)──► 외부 서비스
```

**MCP 이후:**
```
Claude ──(MCP 표준)──► MCP 서버 A (파일시스템)
                  └──► MCP 서버 B (데이터베이스)
                  └──► MCP 서버 C (해충 분류기) ← 우리가 만든 것
```

**MCP 통신 방식 (stdio transport):**
```
Claude Desktop  ──stdin──►  python server.py
               ◄─stdout──
```

- Claude Desktop이 `server.py`를 **서브프로세스**로 실행
- JSON-RPC 형식의 메시지를 stdin/stdout으로 주고받음
- 서버가 로컬 머신에서 실행되므로 **로컬 파일, GPU, 모델 가중치**에 접근 가능

**MCP 서버가 제공하는 것:**

| 개념 | 설명 | 예시 |
|------|------|------|
| **Tool** | LLM이 호출할 수 있는 함수 | `classify_pest()`, `search_pest_info()` |
| **Resource** | LLM이 읽을 수 있는 데이터 | 파일, DB 레코드 |
| **Prompt** | 재사용 가능한 프롬프트 템플릿 | - |

---

### 2-3. RAG (Retrieval-Augmented Generation)

LLM의 지식 한계를 외부 문서 검색으로 보완하는 기법입니다.

**RAG 없이:**
```
사용자: "담배가루이 방제법을 알려줘"
LLM: "학습 데이터 기반으로 일반적인 내용 답변..."  ← 최신 자료 반영 안 됨
```

**RAG 있이:**
```
사용자: "담배가루이 방제법을 알려줘"
     ↓
① 검색: ChromaDB에서 "담배가루이 방제" 관련 PDF 청크 검색
② 컨텍스트 주입: 검색 결과를 LLM에 전달
③ 생성: LLM이 문서 내용을 바탕으로 정확한 답변 생성
```

**RAG 파이프라인 (이 프로젝트):**

```
[오프라인 인덱싱]
reference/*.PDF
      │
      ▼ pypdf로 텍스트 추출
텍스트 전체
      │
      ▼ 500자 단위로 청킹 (overlap 50자)
[청크1][청크2][청크3]...
      │
      ▼ 임베딩 (ChromaDB 기본 임베딩 함수)
벡터 DB (chroma_db/)

[온라인 검색]
쿼리: "담배가루이"
      │
      ▼ 코사인 유사도 검색
관련 청크 top-k 반환
```

---

## 3. 프로젝트 구조

```
claude_MCP_RAG/
├── server.py          # MCP 서버 진입점 — 도구 정의
├── classifier.py      # 딥러닝 추론 로직 (ViT / CNN)
├── rag.py             # PDF 색인 및 벡터 검색 로직
├── pyproject.toml     # 의존성 정의
├── uploads/           # 분류할 이미지를 여기에 저장
├── reference/         # 해충 참고 PDF 문서
└── chroma_db/         # ChromaDB 벡터 인덱스 (자동 생성)
```

---

## 4. 코드 상세 분석

### 4-1. MCP 서버 (`server.py`)

```python
from mcp.server.fastmcp import FastMCP

# 서버 인스턴스 생성 — instructions는 Claude에게 전달되는 시스템 지침
mcp = FastMCP(
    "pest-classifier",
    instructions="온실 해충 분류 서버입니다. 해충 관련 요청이 오면 즉시 classify_pest 도구를 호출하세요..."
)

# @mcp.tool() 데코레이터로 도구 등록
@mcp.tool()
def classify_pest(image_path: Optional[str] = None, model_type: str = "vit") -> dict:
    """
    docstring이 곧 Claude에게 전달되는 도구 설명입니다.
    Claude는 이 설명을 보고 언제 이 도구를 쓸지 결정합니다.
    """
    ...

if __name__ == "__main__":
    mcp.run()  # stdio transport로 MCP 서버 시작
```

**핵심 설계 포인트:**
- `@mcp.tool()` 데코레이터 하나로 함수가 Claude가 호출 가능한 도구가 됨
- 함수의 **타입 힌트**가 JSON Schema로 변환되어 Claude에게 전달됨
- **docstring**이 Claude에게 도구 사용법 설명이 됨 → 잘 작성할수록 Claude가 올바르게 사용

---

### 4-2. 딥러닝 분류기 (`classifier.py`)

두 가지 모델을 지원합니다:

**① Vision Transformer (ViT) — 기본값**
```python
class InsectModel(nn.Module):
    def __init__(self, num_classes):
        self.model = timm.create_model(
            "vit_base_patch16_224",  # 224×224 패치 기반 ViT
            pretrained=False,
            num_classes=num_classes  # 4클래스 분류
        )
```

**② Custom CNN**
```python
class CustomConvNet(nn.Module):
    # Conv → BN → LeakyReLU → MaxPool 블록 5개
    # Global Average Pooling으로 최종 분류
```

**추론 흐름:**
```
이미지 파일 경로
      │
      ▼ cv2.imread()
BGR numpy 배열
      │
      ▼ Albumentations: Resize(224,224) + ToTensor
Float32 텐서 [1, 3, 224, 224]
      │
      ▼ model.forward() + softmax
클래스별 확률 [정상, 담배가루이성충, 담배가루이유충, 애못털진딧물]
      │
      ▼ argmax
{"predicted_class": "애못털진딧물", "confidence": 0.9999, ...}
```

**클래스:**
| 라벨 | 설명 |
|------|------|
| 정상 | 해충 없음 |
| 담배가루이 성충 | Bemisia tabaci 성충 |
| 담배가루이 유충 | Bemisia tabaci 유충 |
| 애못털진딧물 | Aphis gossypii |

---

### 4-3. RAG 검색기 (`rag.py`)

```python
# 청킹: 긴 PDF 텍스트를 겹치는 윈도우로 분할
def _chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])   # 500자
        start += CHUNK_SIZE - CHUNK_OVERLAP               # 50자 오버랩
    return chunks

# 검색: 쿼리와 코사인 유사도가 가장 높은 청크 반환
def search_reference(query: str, top_k: int = 3) -> str:
    col = _get_collection()
    results = col.query(query_texts=[query], n_results=top_k)
    ...
```

**오버랩(overlap)이 필요한 이유:**
```
청크1: [.........500자.......|50자]
청크2:                  [50자|...500자...]
                         ↑ 겹치는 부분
```
청크 경계에서 문장이 잘리면 의미가 손상될 수 있어 앞 청크와 50자를 공유합니다.

---

## 5. 시스템 등록 방법

### Claude Desktop 설정 (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "pest-classifier": {
      "command": "uv",
      "args": [
        "--directory", "C:\\code\\RAG_exercise\\claude_MCP_RAG",
        "run", "python", "server.py"
      ],
      "env": {
        "PYTHONUTF8": "1"
      }
    }
  }
}
```

**각 필드 의미:**
- `command`: MCP 서버를 시작하는 실행 파일
- `args`: 명령 인수 (`uv run python server.py` 실행)
- `env`: 환경 변수 (`PYTHONUTF8=1`은 Windows에서 JSON 파싱 오류 방지)

Claude Desktop을 재시작하면 `server.py`가 서브프로세스로 실행되고,  
Claude가 이 서버의 도구 목록을 가져와 대화 중에 자동으로 호출합니다.

---

## 6. 실제 동작 흐름 (시퀀스)

```
사용자: "uploads 폴더에 이미지 저장했어. 해충 분류해줘."
    │
    ▼
Claude (LLM)
  - 해충 분류 요청임을 인식
  - pest-classifier 서버의 classify_pest 도구를 호출하기로 결정
    │
    ▼ MCP tools/call 요청
MCP 서버 (server.py)
  - classify_pest() 실행
  - uploads/ 에서 최신 이미지 파일 자동 선택
  - classifier.py의 classify_image() 호출
    │
    ▼ 딥러닝 추론
classifier.py
  - ViT 모델로 이미지 분류
  - {"predicted_class": "애못털진딧물", "confidence": 1.0, ...} 반환
    │
    ▼ MCP tools/call 응답
Claude (LLM)
  - 분류 결과 수신
  - search_pest_info("애못털진딧물") 도구 호출 결정
    │
    ▼ MCP tools/call 요청
MCP 서버 (server.py)
  - search_pest_info() 실행
  - ChromaDB에서 "애못털진딧물" 관련 PDF 청크 검색
  - 관련 문서 3개 반환
    │
    ▼ MCP tools/call 응답
Claude (LLM)
  - 분류 결과 + 문서 내용을 종합
  - 자연어 답변 생성
    │
    ▼
사용자: "이미지에서 애못털진딧물이 감지되었습니다 (신뢰도 99.99%). 방제 방법은..."
```

---

## 7. 의존성 및 환경 설정

### `pyproject.toml`

```toml
[project]
name = "pest-mcp"
dependencies = [
    "mcp[cli]>=1.3.0",        # MCP 서버 프레임워크
    "torch>=2.0.0",            # 딥러닝 추론
    "timm>=0.9.0",             # ViT 사전학습 모델
    "albumentations>=1.3.0",   # 이미지 전처리
    "opencv-python>=4.7.0",    # 이미지 읽기/변환
    "numpy>=1.24.0",           # 수치 연산
    "pypdf>=3.0.0",            # PDF 텍스트 추출
    "chromadb>=0.4.0",         # 벡터 DB
    "Pillow>=9.0.0",           # 이미지 처리
]
```

### 환경 설치

```bash
# uv로 가상환경 생성 및 의존성 설치
uv sync

# 서버 단독 실행 테스트
uv run python server.py

# 분류 단독 테스트
uv run python -c "from classifier import classify_image; print(classify_image())"
```

---

## 8. 확장 아이디어

이 프로젝트를 기반으로 할 수 있는 확장:

| 확장 | 방법 |
|------|------|
| 해충 종류 추가 | 모델 재학습 + `PEST_LABELS` 수정 |
| 실시간 카메라 입력 | `uploads/` 폴더를 감시하는 별도 프로세스 추가 |
| 웹 UI | FastAPI로 REST API 래핑, 채팅 업로드 지원 |
| 다국어 지원 | `search_pest_info`의 쿼리를 다국어로 확장 |
| 더 많은 참고 문서 | `reference/` 폴더에 PDF 추가 후 `reindex_pdfs(force=True)` 호출 |

---

## 9. 핵심 정리

```
LLM  = 언어 이해 + 추론 + 생성     (무엇을 할지 결정)
MCP  = LLM ↔ 로컬 시스템 표준 연결  (어떻게 실행할지 연결)
RAG  = 외부 문서 → LLM 컨텍스트     (무엇을 알고 있는지 보완)

셋의 결합 = 전문 지식을 갖추고, 로컬 환경에 접근하며, 자연어로 대화하는 AI 에이전트
```

**LLM 단독의 한계:**
- 최신 전문 지식 없음 → **RAG**로 보완
- 로컬 파일/GPU/DB 접근 불가 → **MCP**로 보완
- 채팅 이미지를 도구에 전달 불가 → **파일 기반 워크플로우**로 설계

---

*이 문서는 `claude_MCP_RAG` 프로젝트의 실제 개발 과정을 기반으로 작성되었습니다.*
