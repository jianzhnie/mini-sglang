"""FastAPI server frontend with OpenAI-compatible API endpoints.

Request/response models live in ``schemas`` and SSE frame construction in
``streaming``; this module wires them to FastAPI routes and the frontend
manager.
"""

from __future__ import annotations

__all__ = [
    "app",
    "ChatCompletionRequest",
    "ChatMessage",
    "CompletionRequest",
    "init_frontend",
]
import asyncio
import queue
import time
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from minisgl.config import SamplingParams, ServerArgs
from minisgl.scheduler.scheduler import Scheduler
from minisgl.server import streaming
from minisgl.server.manager import FrontendManager
from minisgl.server.schemas import ChatCompletionRequest, ChatMessage, CompletionRequest

if TYPE_CHECKING:
    from minisgl.tokenizer import TokenizerWorker

app = FastAPI(title="Mini-SGLang", version="0.1.0")

# Seconds to wait for the next token before timing out a request.
REQUEST_TIMEOUT = 120.0


class GenerationError(Exception):
    """Internal: a request died mid-run (mapped to HTTP 500)."""


# Global frontend manager set after initialization.
_frontend: FrontendManager | None = None


def init_frontend(
    server_args: ServerArgs,
    scheduler: Scheduler,
    tokenizer_worker: "TokenizerWorker",
) -> FrontendManager:
    global _frontend
    _frontend = FrontendManager(server_args, scheduler, tokenizer_worker)
    _frontend.start()
    return _frontend


async def _collect_all_tokens(
    uid: int, result_queue: queue.Queue
) -> tuple[str, str, int]:
    """Collect the full output of a non-streaming request.

    Returns (text, finish_reason, completion_token_count). Raises
    HTTPException on timeout or abort.
    """
    token_ids: list[int] = []
    finish_reason = "stop"

    def _collect() -> None:
        nonlocal finish_reason
        while True:
            token_id, finished, reason = result_queue.get(timeout=REQUEST_TIMEOUT)
            if reason == "abort":
                # The scheduler refused the request up front (e.g. prompt
                # longer than max_seq_len, or un-satisfiable KV demand). No
                # token was ever produced; surface it as a client error rather
                # than a misleading empty 200.
                raise RuntimeError("Request aborted by the scheduler")
            if reason == "error":
                # The request died mid-run (forward/sample failure). No usable
                # token was produced; distinguish from the 400 abort case.
                raise GenerationError("Request failed during generation")
            token_ids.append(token_id)
            if finished:
                finish_reason = reason
                break

    try:
        await asyncio.to_thread(_collect)
    except queue.Empty:
        _frontend.abort_request(uid)
        raise HTTPException(status_code=504, detail="Generation timed out") from None
    except GenerationError as e:
        raise HTTPException(status_code=500, detail=str(e)) from None
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    finally:
        _frontend.remove_result(uid)

    # Re-decode the full token list at once: per-token decode would corrupt
    # multi-byte UTF-8 characters split across tokens.
    return _frontend.tokenizer.decode(token_ids), finish_reason, len(token_ids)


def _submit_or_503(input_ids: list[int], sampling_params: SamplingParams) -> int:
    """Submit a request and return its UID, or raise 503 if already closed."""
    assert _frontend is not None
    uid = _frontend.submit_request(input_ids, sampling_params)
    if _frontend.get_result_queue(uid) is None:
        # The request was already aborted+cleaned up between submit and here
        # (scheduler thread ran ahead). Fail cleanly instead of crashing on a
        # None queue.
        _frontend.abort_request(uid)
        raise HTTPException(status_code=503, detail="Request already closed")
    return uid


def _sampling_params(
    temperature: float, top_p: float, top_k: int, max_tokens: int, ignore_eos: bool
) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible Chat Completions API with SSE streaming."""
    if _frontend is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Apply chat template + tokenize.
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    prompt_text = _frontend.tokenizer.apply_chat_template(messages)
    input_ids = _frontend.tokenizer.encode(prompt_text)

    sampling_params = _sampling_params(
        request.temperature, request.top_p, request.top_k, request.max_tokens, request.ignore_eos
    )
    uid = _submit_or_503(input_ids, sampling_params)
    result_queue = _frontend.get_result_queue(uid)

    if request.stream:
        return StreamingResponse(
            streaming.stream_response(
                _frontend,
                uid,
                result_queue,
                "chat",
                request.model,
                len(input_ids),
                lambda content: streaming.content_chunk(uid, request.model, "chat", content),
            ),
            media_type="text/event-stream",
        )

    text, finish_reason, completion_tokens = await _collect_all_tokens(uid, result_queue)
    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": streaming.usage(len(input_ids), completion_tokens),
    }


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """OpenAI-compatible Completions API with SSE streaming."""
    if _frontend is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    input_ids = _frontend.tokenizer.encode(request.prompt)
    sampling_params = _sampling_params(
        request.temperature, request.top_p, request.top_k, request.max_tokens, request.ignore_eos
    )
    uid = _submit_or_503(input_ids, sampling_params)
    result_queue = _frontend.get_result_queue(uid)

    if request.stream:
        return StreamingResponse(
            streaming.stream_response(
                _frontend,
                uid,
                result_queue,
                "completion",
                request.model,
                len(input_ids),
                lambda content: streaming.content_chunk(uid, request.model, "completion", content),
            ),
            media_type="text/event-stream",
        )

    text, finish_reason, completion_tokens = await _collect_all_tokens(uid, result_queue)
    return {
        "id": f"cmpl-{uid}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
        "usage": streaming.usage(len(input_ids), completion_tokens),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
