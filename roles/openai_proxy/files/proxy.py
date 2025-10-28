import os, time, requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

TRITON_HTTP = os.environ.get("TRITON_HTTP", "http://triton-llm.default.svc.cluster.local:8000")
ACTIVE_MODEL = os.environ.get("ACTIVE_MODEL", "")

app = FastAPI(title="OpenAI Proxy + Model Manager (PP=2)")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    model = req.model or os.environ.get("ACTIVE_MODEL") or ACTIVE_MODEL
    if not model:
        raise HTTPException(400, "No active model set. Call /admin/set_model first or include model.")
    # Minimal Triton infer stub: BYTES in/out; align with your backend's I/O names.
    payload = {"inputs": [{"name": "text_input", "shape": [1], "datatype": "BYTES", "data": [req.messages[-1].content]}]}
    r = requests.post(f"{TRITON_HTTP}/v2/models/{model}/infer", json=payload, timeout=600)
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    out = r.json()
    text = ""
    for o in out.get("outputs", []):
        if o.get("name") == "text_output":
            data = o.get("data", [])
            text = data[0] if data else ""
            break
    return {
        "id": "chatcmpl-proxy",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text or "[no output from backend]"},
            "finish_reason": "stop"
        }]
    }

@app.get("/admin/set_model")
def set_model(name: str):
    try:
        old = os.environ.get("ACTIVE_MODEL", "")
        if old:
            requests.post(f"{TRITON_HTTP}/v2/repository/models/{old}/unload", timeout=120)
    except Exception:
        pass
    r = requests.post(f"{TRITON_HTTP}/v2/repository/models/{name}/load", timeout=120)
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    for _ in range(180):
        s = requests.get(f"{TRITON_HTTP}/v2/models/{name}", timeout=10)
        if s.status_code == 200:
            os.environ["ACTIVE_MODEL"] = name
            return {"ok": True, "active_model": name}
        time.sleep(2)
    raise HTTPException(504, "Model did not become ready in time.")

@app.get("/admin/active_model")
def active_model():
    return {"active_model": os.environ.get("ACTIVE_MODEL", "")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
