import sys
from pathlib import Path
from typing import Optional

# ensure sibling modules (classifier, rag) are importable regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP
from classifier import classify_image, get_latest_upload, UPLOADS_DIR
from rag import search_reference, reindex

mcp = FastMCP(
    "pest-classifier",
    instructions=(
        "온실 해충 분류 서버입니다. 해충 관련 요청이 오면 즉시 classify_pest 도구를 호출하세요.\n\n"
        "도구 호출 순서:\n"
        "1. classify_pest() — image_path 생략 시 uploads/ 폴더 최신 파일을 자동 사용합니다.\n"
        "2. search_pest_info(predicted_class) — 분류 결과로 참고 문서를 검색합니다.\n"
        "3. 결과를 사용자에게 설명합니다.\n\n"
        "중요: 이 서버는 사용자 로컬 머신에서 실행됩니다. "
        "파일 읽기는 이 서버가 직접 수행하므로 로컬 경로를 그대로 image_path 에 전달하면 됩니다. "
        f"이미지를 찾을 수 없으면 사용자에게 {UPLOADS_DIR} 폴더에 파일을 저장해달라고 안내하세요."
    ),
)


@mcp.tool()
def classify_pest(image_path: Optional[str] = None, model_type: str = "vit") -> dict:
    """
    온실 해충 이미지를 분류합니다.

    image_path 를 생략하면 uploads/ 폴더의 가장 최근 이미지를 자동으로 사용합니다.

    Args:
        image_path: 이미지 파일의 절대 경로 (생략 시 uploads/ 폴더 최신 파일 자동 선택)
        model_type: 사용할 모델 — "vit" (Vision Transformer, 기본값) 또는 "cnn"

    Returns:
        predicted_class: 예측된 해충 종류 (정상/담배가루이 성충/담배가루이 유충/애못털진딧물)
        confidence: 예측 신뢰도 (0~1)
        probabilities: 각 클래스별 확률
        model_used: 실제 사용된 모델
        image_used: 실제 분류에 사용된 이미지 경로
    """
    try:
        result = classify_image(image_path, model_type)
        result["image_used"] = image_path if image_path else str(get_latest_upload())
        return result
    except ValueError as e:
        msg = str(e)
        if "이미지 파일이 없습니다" in msg:
            return {
                "status": "이미지 없음",
                "사용자_안내": (
                    "채팅 첨부 이미지는 이 도구에서 지원되지 않습니다. "
                    "아래 폴더에 이미지 파일을 저장(복사)한 뒤 다시 요청해 주세요:\n"
                    f"{UPLOADS_DIR}\n\n"
                    "저장 후 '해충 분류해줘' 라고 입력하면 바로 분류합니다."
                ),
            }
        return {"status": "오류", "메시지": msg}


@mcp.tool()
def search_pest_info(query: str, top_k: int = 3) -> str:
    """
    해충 관련 참고 PDF 문서에서 정보를 검색합니다.

    Args:
        query: 검색 키워드 (예: "담배가루이", "진딧물 방제")
        top_k: 반환할 최대 문서 단락 수 (기본값: 3)

    Returns:
        참고 문서에서 검색된 관련 내용
    """
    return search_reference(query, top_k)


@mcp.tool()
def reindex_pdfs(force: bool = False) -> str:
    """
    reference 폴더의 PDF를 다시 색인합니다.

    Args:
        force: True 이면 기존 색인을 삭제하고 처음부터 재생성합니다.

    Returns:
        색인 결과 메시지
    """
    return reindex(force)


if __name__ == "__main__":
    mcp.run()
