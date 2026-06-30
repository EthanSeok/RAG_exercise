from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import requests
import uvicorn


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4:31b"

API_HOST = "0.0.0.0"
API_PORT = 8000
PUBLIC_URL = "http://113.198.38.104:8000"

API_KEY = "password 지정"


app = FastAPI(title="Gemma4 API")


class PromptRequest(BaseModel):
    prompt: str
    temperature: float = 0.2
    max_tokens: int = 2048


def verify_api_key(x_api_key: str | None):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )


@app.post("/generate")
def generate(
    req: PromptRequest,
    x_api_key: str | None = Header(default=None),
):
    verify_api_key(x_api_key)

    payload = {
        "model": MODEL_NAME,
        "prompt": req.prompt,
        "stream": False,
        "options": {
            "temperature": req.temperature,
            "num_predict": req.max_tokens,
        },
    }

    res = requests.post(OLLAMA_URL, json=payload, timeout=300)
    res.raise_for_status()

    data = res.json()

    return {
        "model": MODEL_NAME,
        "response": data.get("response", ""),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "local": f"http://127.0.0.1:{API_PORT}",
        "public": PUBLIC_URL,
    }


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
    )