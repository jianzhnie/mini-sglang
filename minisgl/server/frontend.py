"""FastAPI server frontend with OpenAI-compatible API endpoints."""

__all__ = [
    "app",
    "ChatMessage",
    "ChatCompletionRequest",
    "CompletionRequest",
    "FrontendManager",
    "init_frontend",
]
import asyncio
import json
import queue
import threading
import time
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from minisgl.config import SamplingParams, ServerArgs
from minisgl.scheduler.scheduler import Scheduler
from minisgl.utils.logger import logger

if TYPE_CHECKING:
    from minisgl.models.tokenizer.worker import TokenizerWorker

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


class FrontendManager:
    """Manages the lifecycle of inference requests and SSE streaming."""

    def __init__(
        self,
        server_args: ServerArgs,
        scheduler: Scheduler,
        tokenizer_worker: "TokenizerWorker",
    ) -> None:
        self.args = server_args
        self.scheduler = scheduler
        self.tokenizer = tokenizer_worker
        self._results: dict[int, queue.Queue] = {}
        self._lock = threading.Lock()
        self._running = True

    def submit_request(
        self, input_ids: list[int], sampling_params: SamplingParams
    ) -> int:
        """Submit a tokenized request and return its UID.

        The result queue is registered under the same lock that process_step
        holds while distributing tokens, so the first token can never be
        produced before the queue exists.
        """
        with self._lock:
            uid = self.scheduler.add_request(input_ids, sampling_params)
            self._results[uid] = queue.Queue()
            return uid

    def get_result_queue(self, uid: int) -> queue.Queue:
        with self._lock:
            return self._results.get(uid)

    def remove_result(self, uid: int) -> None:
        with self._lock:
            self._results.pop(uid, None)

    def process_step(self) -> None:
        """Run one scheduler step and distribute results."""
        try:
            step_results = self.scheduler.step()
        except (RuntimeError, ValueError) as e:
            logger.error("Scheduler step error: %s", e)
            return
        # Hold the lock while distributing: remove_result() may run
        # concurrently from a streaming generator's finally block.
        with self._lock:
            for out in step_results:
                q = self._results.get(out.uid)
                if q is not None:
                    q.put((out.token_id, out.finished, out.finish_reason))

    def run_event_loop(self) -> None:
        """Background thread running the scheduler loop."""
        logger.info("Scheduler event loop started")
        while self._running:
            try:
                if self.scheduler.is_idle():
                    time.sleep(0.01)  # avoid a busy spin while idle
                    continue
                self.process_step()
            except Exception as exc:
                logger.error("Scheduler event loop error: %s", exc)

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run_event_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False


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


class IncrementalDetokenizer:
    """Incrementally decode a token stream into text deltas.

    Re-decodes the full token list on every token so multi-byte UTF-8
    characters split across tokens are never emitted as U+FFFD replacement
    characters.
    """

    def __init__(self, tokenizer: "TokenizerWorker") -> None:
        self.tokenizer = tokenizer
        self.token_ids: list[int] = []
        self.text = ""

    def add_token(self, token_id: int) -> str:
        """Append a token and return the newly produced text."""
        self.token_ids.append(token_id)
        new_text = self.tokenizer.decode(self.token_ids, skip_special_tokens=True)
        # A trailing U+FFFD may be an incomplete multi-byte character still
        # waiting for its remaining bytes; hold it back for the next token.
        stable = new_text.removesuffix("\ufffd")
        delta = stable[len(self.text) :]
        self.text = stable
        return delta


def _error_chunk(message: str) -> str:
    """Build an SSE error chunk."""
    return f"data: {json.dumps({'error': {'message': message}})}\n\n"


async def _collect_all_tokens(uid: int, result_queue: queue.Queue) -> tuple[str, str]:
    """Collect the full output of a non-streaming request.

    Returns (text, finish_reason). Raises HTTPException on timeout or abort.
    """
    token_ids: list[int] = []
    finish_reason = "stop"

    def _collect() -> None:
        nonlocal finish_reason
        while True:
            token_id, finished, reason = result_queue.get(timeout=REQUEST_TIMEOUT)
            if reason == "abort":
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
        raise HTTPException(status_code=500, detail=str(e)) from None
    finally:
        _frontend.remove_result(uid)

    return _frontend.tokenizer.decode(token_ids), finish_reason


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

    if request.stream:
        return StreamingResponse(
            _stream_chat_response(uid, result_queue, request.model),
            media_type="text/event-stream",
        )

    # Collect all tokens for non-streaming (runs blocking queue.get in a thread)
    text, finish_reason = await _collect_all_tokens(uid, result_queue)
    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
    }


def _stream_chat_response(uid: int, result_queue: queue.Queue, model: str):
    """SSE streaming generator for chat completions (sync, runs in thread pool)."""
    return _stream_response(
        uid,
        result_queue,
        lambda content, finished, reason: {
            "id": f"chatcmpl-{uid}",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": reason if finished else None,
                }
            ],
        },
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

    if request.stream:
        return StreamingResponse(
            _stream_completion_response(uid, result_queue, request.model),
            media_type="text/event-stream",
        )

    text, finish_reason = await _collect_all_tokens(uid, result_queue)
    return {
        "id": f"cmpl-{uid}",
        "object": "text_completion",
        "model": request.model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
    }


def _stream_completion_response(uid: int, result_queue: queue.Queue, model: str):
    """SSE streaming generator for completions (sync, runs in thread pool)."""
    return _stream_response(
        uid,
        result_queue,
        lambda content, finished, reason: {
            "id": f"cmpl-{uid}",
            "object": "text_completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": content,
                    "finish_reason": reason if finished else None,
                }
            ],
        },
    )


def _stream_response(uid: int, result_queue: queue.Queue, make_chunk):
    """Shared SSE streaming core; ``make_chunk`` shapes the per-API chunk dict."""
    detok = IncrementalDetokenizer(_frontend.tokenizer)
    try:
        while True:
            token_id, finished, finish_reason = result_queue.get(
                timeout=REQUEST_TIMEOUT
            )
            if finish_reason == "abort":
                yield _error_chunk("Request aborted by the scheduler")
                break
            chunk = make_chunk(detok.add_token(token_id), finished, finish_reason)
            yield f"data: {json.dumps(chunk, ensure_ascii=True)}\n\n"

            if finished:
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
