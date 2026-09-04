"""SSE streaming core for the OpenAI-compatible API.

Pure serialization + stream driving: given a result queue and a frontend, it
emits OpenAI-style SSE frames (content chunks, a terminal chunk with
``finish_reason`` + ``usage``, then ``[DONE]``). No FastAPI dependency here, so
the frame shapes are unit-testable in isolation.
"""

from __future__ import annotations

import json
import queue
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from minisgl.server.manager import IncrementalDetokenizer
from minisgl.utils.logger import logger

if TYPE_CHECKING:
    from minisgl.server.manager import FrontendManager

__all__ = [
    "content_chunk",
    "error_chunk",
    "finish_chunk",
    "stream_response",
    "usage",
]


def usage(prompt_tokens: int, completion_tokens: int) -> dict:
    """OpenAI-style token usage block."""
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=True)}\n\n"


def error_chunk(message: str) -> str:
    """Build an SSE error chunk."""
    return _sse({"error": {"message": message}})


def content_chunk(uid: int, model: str, api: str, content: str) -> str:
    """SSE content chunk: chat carries ``delta.content``, completions ``text``.

    Content chunks never carry a finish_reason (OpenAI streaming contract).
    """
    if api == "chat":
        body = {
            "id": f"chatcmpl-{uid}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }
    else:
        body = {
            "id": f"cmpl-{uid}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "text": content, "finish_reason": None}],
        }
    return _sse(body)


def finish_chunk(
    uid: int,
    model: str,
    api: str,
    reason: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> str:
    """OpenAI terminal stream chunk: empty delta/text, finish_reason, usage."""
    if api == "chat":
        body = {
            "id": f"chatcmpl-{uid}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
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
    body["usage"] = usage(prompt_tokens, completion_tokens)
    return _sse(body)


def stream_response(
    frontend: FrontendManager,
    uid: int,
    result_queue: queue.Queue,
    api: str,
    model: str,
    prompt_tokens: int,
    make_content_chunk: Callable[[str], str],
    timeout: float = 120.0,
):
    """Drive a result queue into OpenAI-style SSE frames.

    Yields the frames for one request: content chunks as tokens arrive, a
    terminal chunk (finish_reason + usage) when the request finishes, and a
    final ``[DONE]``. Content chunks are produced by ``make_content_chunk``
    (which captures the per-API shape); the rest is API-agnostic here.
    """
    detok = IncrementalDetokenizer(frontend.tokenizer)
    completion_tokens = 0
    try:
        while True:
            token_id, finished, reason = result_queue.get(timeout=timeout)
            if reason == "abort":
                yield error_chunk("Request aborted by the scheduler")
                break
            if reason == "error":
                # The request died mid-run (forward/sample failure). No usable
                # token was produced for it; report and stop cleanly.
                logger.error("Request %s failed during generation", uid)
                yield error_chunk("Request failed during generation")
                break
            completion_tokens += 1
            content = detok.add_token(token_id)
            if content:
                yield make_content_chunk(content)
            if finished:
                yield finish_chunk(
                    uid, model, api, reason, prompt_tokens, completion_tokens
                )
                break
    except queue.Empty:
        logger.warning(
            "Streaming request %s timed out after %ss; aborting", uid, timeout
        )
        frontend.abort_request(uid)
        yield error_chunk("Timed out waiting for the next token")
    except GeneratorExit:
        # Client disconnected; make sure the request does not run forever.
        frontend.abort_request(uid)
        raise
    except Exception as exc:
        logger.warning("Streaming request %s failed: %s", uid, exc)
        yield error_chunk(str(exc))
    finally:
        frontend.remove_result(uid)
    yield "data: [DONE]\n\n"
