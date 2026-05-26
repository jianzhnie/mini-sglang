"""FastAPI server frontend with OpenAI-compatible API endpoints."""

import asyncio
import json
import queue
import threading
import time
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from minisgl.config import SamplingParams, ServerArgs
from minisgl.scheduler.scheduler import Scheduler
from minisgl.utils.logger import logger

app = FastAPI(title="Mini-SGLang", version="0.1.0")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: List[ChatMessage]
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 1024
    stream: bool = True
    ignore_eos: bool = False


class CompletionRequest(BaseModel):
    model: str = "default"
    prompt: str
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 1024
    stream: bool = True
    ignore_eos: bool = False


class FrontendManager:
    """Manages the lifecycle of inference requests and SSE streaming."""

    def __init__(self, server_args: ServerArgs, scheduler: Scheduler, tokenizer_worker):
        self.args = server_args
        self.scheduler = scheduler
        self.tokenizer = tokenizer_worker
        self._results: Dict[int, queue.Queue] = {}
        self._lock = threading.Lock()
        self._running = True

    def submit_request(self, input_ids: List[int], sampling_params: SamplingParams) -> int:
        """Submit a tokenized request and return its UID."""
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
            for uid, token_id, finished in step_results:
                if uid in self._results:
                    self._results[uid].put((token_id, finished))
        except Exception as e:
            logger.error(f"Scheduler step error: {e}")

    def run_event_loop(self) -> None:
        """Background thread running the scheduler loop."""
        logger.info("Scheduler event loop started")
        while self._running:
            if self.scheduler.is_idle():
                time.sleep(0.01)
                continue
            self.process_step()

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run_event_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False


# Global frontend manager set after initialization
_frontend: Optional[FrontendManager] = None


def init_frontend(server_args: ServerArgs, scheduler: Scheduler, tokenizer_worker) -> FrontendManager:
    global _frontend
    _frontend = FrontendManager(server_args, scheduler, tokenizer_worker)
    _frontend.start()
    return _frontend


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
    else:
        # Collect all tokens for non-streaming
        generated_tokens = []
        while True:
            token_id, finished = result_queue.get()
            generated_tokens.append(token_id)
            if finished:
                break

        text = _frontend.tokenizer.tokenizer.decode(generated_tokens)
        _frontend.remove_result(uid)
        return {
            "id": f"chatcmpl-{uid}",
            "object": "chat.completion",
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
        }


def _stream_chat_response(uid: int, result_queue: queue.Queue, model: str):
    """SSE streaming generator for chat completions (sync, runs in thread pool)."""
    try:
        while True:
            token_id, finished = result_queue.get()
            text = _frontend.tokenizer.decode(token_id)

            chunk = {
                "id": f"chatcmpl-{uid}",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": "stop" if finished else None,
                }],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=True)}\n\n"

            if finished:
                break
    finally:
        _frontend.remove_result(uid)
        yield "data: [DONE]\n\n"


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
    else:
        generated_tokens = []
        while True:
            token_id, finished = result_queue.get()
            generated_tokens.append(token_id)
            if finished:
                break
        text = _frontend.tokenizer.tokenizer.decode(generated_tokens)
        _frontend.remove_result(uid)
        return {
            "id": f"cmpl-{uid}",
            "object": "text_completion",
            "model": request.model,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        }


def _stream_completion_response(uid: int, result_queue: queue.Queue, model: str):
    """SSE streaming generator for completions (sync, runs in thread pool)."""
    try:
        while True:
            token_id, finished = result_queue.get()
            text = _frontend.tokenizer.decode(token_id)

            chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion",
                "model": model,
                "choices": [{
                    "index": 0,
                    "text": text,
                    "finish_reason": "stop" if finished else None,
                }],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=True)}\n\n"

            if finished:
                break
    finally:
        _frontend.remove_result(uid)
        yield "data: [DONE]\n\n"


@app.get("/health")
async def health():
    return {"status": "ok"}
