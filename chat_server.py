"""
OpenAI-compatible bridge server for the LangGraph dynamic supervisor.

NextChat (https://github.com/ChatGPTNextWeb/NextChat) talks to any backend
configured as a "Custom Endpoint" using the same request/response shape as
OpenAI's Chat Completions API: POST {base_url}/v1/chat/completions, with
{"messages": [...], "stream": bool, "model": str, ...} in and either a
`chat.completion` JSON object or a `text/event-stream` of
`chat.completion.chunk` objects back out.

This server implements just enough of that contract to let NextChat be
used as a chat UI in front of `graph` from langgraph_dynamic_supervisor.py,
instead of typing into the terminal. Each request's full message history
is passed into graph.invoke() so multi-turn context is preserved; the
graph's own (blocking) answer is then sent back as a single chunk when
streaming is requested - there's no real token-by-token generation here,
NextChat just animates the one chunk it receives.

Route handlers below are plain `def`, not `async def`, so FastAPI runs
them in its worker thread pool - graph.invoke() blocks on a network call
to Anthropic, and this keeps that from stalling the whole server.
"""

import json
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from langgraph_dynamic_supervisor import graph

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "langgraph-supervisor"


def _message_text(content) -> str:
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def _to_lc_messages(messages):
    return [(m.get("role"), _message_text(m.get("content"))) for m in messages if _message_text(m.get("content"))]


@app.post("/v1/chat/completions")
def chat_completions(payload: dict):
    lc_messages = _to_lc_messages(payload.get("messages", []))
    answer = graph.invoke({"messages": lc_messages})["messages"][-1].content if lc_messages else "(empty message)"

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if payload.get("stream"):
        def sse():
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_NAME,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
