# 온실 해충 분류 MCP 서버

Claude Desktop에서 자연어로 온실 해충 이미지를 분류하고 PDF 문서 기반 방제 정보를 검색하는 **MCP(Model Context Protocol) 서버**입니다.

## 시스템 구조

```
사용자 (Claude Desktop)
        │ 자연어 요청
        ▼
   Claude (LLM)          — 도구 호출 결정 + 답변 생성
        │ MCP (JSON-RPC over stdio)
        ▼
   MCP 서버 (server.py)  — 로컬 머신에서 실행
        │
   ┌────┴────┐
   ▼         ▼
딥러닝 분류기   RAG 검색기
(ViT / CNN)   (ChromaDB + PDF)
```

## 제공 도구 (MCP Tools)

| 도구 | 설명 |
|------|------|
| `classify_pest` | 이미지 파일로 해충 분류 (ViT 또는 CNN 선택 가능) |
| `search_pest_info` | PDF 문서에서 해충 관련 정보 검색 |
| `reindex_pdfs` | `reference/` 폴더의 PDF를 다시 색인 |

**분류 가능한 해충:**
- 정상 (해충 없음)
- 담배가루이 성충 (*Bemisia tabaci*)
- 담배가루이 유충
- 애못털진딧물 (*Aphis gossypii*)

## 요구사항

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) 패키지 매니저
- [Claude Desktop](https://claude.ai/download)
- 학습된 모델 가중치:
  - `../pest_classification/output/vit-aug/vit_base_12.pth`
  - `../pest_classification/output/cnn-aug/cnn_base_44.pth`

## 설치

```bash
# 의존성 설치
uv sync
```

## Claude Desktop 연동

`%APPDATA%\Claude\claude_desktop_config.json` 에 아래 내용을 추가합니다.

```json
{
  "mcpServers": {
    "pest-classifier": {
      "command": "uv",
      "args": [
        "--directory", "C:\\path\\to\\claude_MCP_RAG",
        "run", "python", "server.py"
      ],
      "env": {
        "PYTHONUTF8": "1"
      }
    }
  }
}
```

> `PYTHONUTF8=1` — Windows에서 stdin/stdout UTF-8 BOM 문제 방지

Claude Desktop을 재시작하면 서버가 자동으로 연결됩니다.

## 사용 방법

1. 분류할 이미지를 `uploads/` 폴더에 저장합니다.
2. Claude Desktop에서 다음과 같이 요청합니다.

```
uploads 폴더에 해충 이미지 저장했어. classify_pest 도구로 분류해줘.
```

Claude가 자동으로:
1. `classify_pest()` 호출 → 딥러닝 모델로 해충 종류 + 신뢰도 반환
2. `search_pest_info()` 호출 → 해당 해충의 방제 정보를 PDF에서 검색
3. 결과를 종합해 자연어로 설명

### 분류 결과 예시

```json
{
  "predicted_class": "애못털진딧물",
  "confidence": 0.9999,
  "probabilities": {
    "정상": 0.0,
    "담배가루이 성충": 0.0,
    "담배가루이 유충": 0.0,
    "애못털진딧물": 0.9999
  },
  "model_used": "VIT"
}
```

## RAG 문서 추가

`reference/` 폴더에 PDF 파일을 추가한 후:

```
Claude: reindex_pdfs 도구로 문서를 다시 색인해줘.
```

또는 강제 재생성:

```python
reindex_pdfs(force=True)
```

## 프로젝트 구조

```
claude_MCP_RAG/
├── server.py       # MCP 서버 진입점, 도구 정의
├── classifier.py   # ViT / CNN 딥러닝 추론
├── rag.py          # PDF 청킹, ChromaDB 색인 및 검색
├── pyproject.toml  # 의존성
├── uploads/        # 분류할 이미지 보관 폴더
├── reference/      # 해충 참고 PDF (직접 추가)
└── chroma_db/      # 벡터 DB (자동 생성)
```

## 의존성

```toml
mcp[cli]>=1.3.0        # MCP 서버 프레임워크
torch>=2.0.0           # 딥러닝 추론
timm>=0.9.0            # ViT 사전학습 모델
albumentations>=1.3.0  # 이미지 전처리
opencv-python>=4.7.0   # 이미지 읽기
pypdf>=3.0.0           # PDF 텍스트 추출
chromadb>=0.4.0        # 벡터 DB
```

## 알려진 한계

- **채팅 첨부 이미지 미지원**: Claude는 채팅에서 받은 이미지를 MCP 도구에 바이트로 전달할 수 없습니다. 이미지는 반드시 `uploads/` 폴더에 파일로 저장해야 합니다.
- **MCP 서버는 로컬 전용**: 이 서버는 사용자 컴퓨터에서 직접 실행되며 외부에서 접근할 수 없습니다.
