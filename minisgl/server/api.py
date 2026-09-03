"""FastAPI server frontend with OpenAI-compatible API endpoints."""

__all__ = [
    "app",
    "ChatMessage",
    "ChatCompletionRequest",
    "CompletionRequest",
    "init_frontend",
]
import asyncio
import json
import queue
import time
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from minisgl.config import SamplingParams, ServerArgs
from minisgl.scheduler.scheduler import Scheduler
from minisgl.server.manager import FrontendManager, IncrementalDetokenizer
from minisgl.utils.logger import logger

if TYPE_CHECKING:
    from minisgl.tokenizer import TokenizerWorker

app = FastAPI(title="Mini-SGLang", version="0.1.0")

# Seconds to wait for the next token before timing out a request.
REQUEST_TIMEOUT = 120.0


class ChatMessage(BaseModel):
    """A chat message with role and content."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = "default"
    messages: list[ChatMessage]
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 1024
    stream: bool = False
    ignore_eos: bool = False


class CompletionRequest(BaseModel):
    model: str = "default"
    prompt: str
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 1024
    stream: bool = False
    ignore_eos: bool = False


# Global frontend manager set after initialization
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


def _error_chunk(message: str) -> str:
    """Build an SSE error chunk."""
    return f"data: {json.dumps({'error': {'message': message}})}\n\n"


def _usage(prompt_tokens: int, completion_tokens: int) -> dict:
    """OpenAI-style token usage block."""
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


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
            token_ids.append(token_id)
            if finished:
                finish_reason = reason
                break

    try:
        await asyncio.to_thread(_collect)
    except queue.Empty:
        _frontend.scheduler.abort_request(uid)
        raise HTTPException(status_code=504, detail="Generation timed out") from None
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    finally:
        _frontend.remove_result(uid)

    # Re-decode the full token list at once: per-token decode would corrupt
    # multi-byte UTF-8 characters split across tokens.
    return _frontend.tokenizer.decode(token_ids), finish_reason, len(token_ids)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible Chat Completions API with SSE streaming."""
    if _frontend is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Apply chat template
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    prompt_text = _frontend.tokenizer.apply_chat_template(messages)

    # Tokenize
    input_ids = _frontend.tokenizer.encode(prompt_text)

    # Sampling params
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_tokens,
        ignore_eos=request.ignore_eos,
    )

    uid = _frontend.submit_request(input_ids, sampling_params)
    result_queue = _frontend.get_result_queue(uid)
    if result_queue is None:
        # The request was already aborted+cleaned up between submit and here
        # (scheduler thread ran ahead). Fail cleanly instead of crashing on a
        # None queue.
        _frontend.scheduler.abort_request(uid)
        raise HTTPException(status_code=503, detail="Request already closed")

    if request.stream:
        return StreamingResponse(
            _stream_chat_response(uid, result_queue, request.model, len(input_ids)),
            media_type="text/event-stream",
        )

    # Collect all tokens for non-streaming (runs blocking queue.get in a thread)
    text, finish_reason, completion_tokens = await _collect_all_tokens(
        uid, result_queue
    )
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
        "usage": _usage(len(input_ids), completion_tokens),
    }


def _chat_content_chunk(uid: int, model: str, content: str) -> str:
    """SSE content chunk for chat completions (no finish_reason yet)."""
    chunk = {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=True)}\n\n"


def _finish_chunk(
    uid: int, model: str, api: str, reason: str, prompt_tokens: int, completion_tokens: int
) -> str:
    """OpenAI terminal stream chunk: empty delta/text, finish_reason, usage."""
    if api == "chat":
        body = {
            "id": f"chatcmpl-{uid}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": reason}
            ],
        }
    else:
        body = {
            "id": f"cmpl-{uid}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "text": "", "finish_reason": reason}],
        }
    # usage is only carried on the final streamed chunk (OpenAI contract).
    body["usage"] = _usage(prompt_tokens, completion_tokens)
    return f"data: {json.dumps(body, ensure_ascii=True)}\n\n"


def _stream_chat_response(
    uid: int, result_queue: queue.Queue, model: str, prompt_tokens: int
):
    """SSE streaming generator for chat completions (sync, runs in thread pool)."""
    return _stream_response(
        uid,
        result_queue,
        "chat",
        model,
        prompt_tokens,
        lambda content: _chat_content_chunk(uid, model, content),
    )


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """OpenAI-compatible Completions API with SSE streaming."""
    if _frontend is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    input_ids = _frontend.tokenizer.encode(request.prompt)

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_tokens,
        ignore_eos=request.ignore_eos,
    )

    uid = _frontend.submit_request(input_ids, sampling_params)
    result_queue = _frontend.get_result_queue(uid)
    if result_queue is None:
        # See chat_completions: the request may have been cleaned up between
        # submit and here.
        _frontend.scheduler.abort_request(uid)
        raise HTTPException(status_code=503, detail="Request already closed")

    if request.stream:
        return StreamingResponse(
            _stream_completion_response(uid, result_queue, request.model, len(input_ids)),
            media_type="text/event-stream",
        )

    text, finish_reason, completion_tokens = await _collect_all_tokens(
        uid, result_queue
    )
    return {
        "id": f"cmpl-{uid}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
        "usage": _usage(len(input_ids), completion_tokens),
    }


def _completion_content_chunk(uid: int, model: str, content: str) -> str:
    """SSE content chunk for completions (no finish_reason yet)."""
    chunk = {
        "id": f"cmpl-{uid}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "text": content, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=True)}\n\n"


def _stream_completion_response(
    uid: int, result_queue: queue.Queue, model: str, prompt_tokens: int
):
    """SSE streaming generator for completions (sync, runs in thread pool)."""
    return _stream_response(
        uid,
        result_queue,
        "completion",
        model,
        prompt_tokens,
        lambda content: _completion_content_chunk(uid, model, content),
    )


def _stream_response(
    uid: int, result_queue: queue.Queue, api: str, model: str, prompt_tokens: int,
    make_content_chunk,
):
    """Shared SSE streaming core.

    Content chunks never carry a finish_reason; the token that finishes a
    request is followed by a separate terminal chunk holding finish_reason and
    usage (OpenAI's streaming contract), then a final [DONE].
    """
    detok = IncrementalDetokenizer(_frontend.tokenizer)
    completion_tokens = 0
    finish_reason: str | None = None
    try:
        while True:
            token_id, finished, reason = result_queue.get(timeout=REQUEST_TIMEOUT)
            if reason == "abort":
                yield _error_chunk("Request aborted by the scheduler")
                break
            completion_tokens += 1
            content = detok.add_token(token_id)
            if content:
                yield make_content_chunk(content)
            if finished:
                finish_reason = reason
                yield _finish_chunk(
                    uid, model, api, finish_reason, prompt_tokens, completion_tokens
                )
                break
    except queue.Empty:
        logger.warning(
            "Streaming request %s timed out after %ss; aborting", uid, REQUEST_TIMEOUT
        )
        _frontend.scheduler.abort_request(uid)
        yield _error_chunk("Timed out waiting for the next token")
    except GeneratorExit:
        # Client disconnected; make sure the request does not run forever.
        _frontend.scheduler.abort_request(uid)
        raise
    except Exception as exc:
        logger.warning("Streaming request %s failed: %s", uid, exc)
        yield _error_chunk(str(exc))
    finally:
        _frontend.remove_result(uid)
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health():
    return {"status": "ok"}
