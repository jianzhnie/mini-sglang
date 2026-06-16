#!/usr/bin/env python3
"""Mini-SGLang OpenAI-compatible API server demo.

Quick start:
    python server_demo.py --model-path ~/hfhub/models/facebook/opt-125m
    python server_demo.py --model-path ~/hfhub/models/Qwen/Qwen3-0.6B

Then test in another terminal:
    curl http://127.0.0.1:8765/health
    curl http://127.0.0.1:8765/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"prompt":"The capital of France is","max_tokens":30,"stream":false}'
"""

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL = os.path.expanduser("~/hfhub/models/facebook/opt-125m")
HOST, PORT = "127.0.0.1", 8765


def _request(path: str, body: dict = None) -> dict:
    url = f"http://{HOST}:{PORT}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data,
        method="GET" if body is None else "POST",
    )
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as resp:
        content = resp.read().decode()
        return json.loads(content) if content else {}


def _stream_request(path: str, body: dict):
    url = f"http://{HOST}:{PORT}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            line = line.decode().strip()
            raw = line.removeprefix("data: ").strip()
            if raw and raw != "[DONE]":
                try:
                    chunk = json.loads(raw)
                    choices = chunk.get("choices", [])
                    if choices:
                        c = choices[0]
                        content = c.get("text") or c.get("delta", {}).get("content", "")
                        print(content, end="", flush=True)
                except json.JSONDecodeError:
                    pass
        print()


def start_server(model_path: str):
    import logging

    import uvicorn

    from minisgl.config import ModelArgs, ServerArgs
    from minisgl.engine.engine import Engine
    from minisgl.models.tokenizer.worker import TokenizerWorker
    from minisgl.scheduler.scheduler import Scheduler
    from minisgl.server.frontend import app, init_frontend
    from minisgl.utils.logger import setup_logger

    setup_logger(level=logging.INFO)

    args = ServerArgs(
        model_path=model_path,
        host=HOST,
        port=PORT,
        tp_size=1,
        attention_backend="fa",
        max_running_req=8,
        max_seq_len=512,
        page_size=16,
        memory_ratio=0.5,
    )
    model_args = ModelArgs.from_pretrained(model_path)
    tokenizer = TokenizerWorker(model_path)
    engine = Engine(args, model_args, tp_rank=0)
    scheduler = Scheduler(args, engine)
    init_frontend(args, scheduler, tokenizer)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    import multiprocessing

    parser = argparse.ArgumentParser(description="Mini-SGLang Server Demo")
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL,
        help="Path to HuggingFace model directory",
    )
    cli_args = parser.parse_args()
    model_path = cli_args.model_path

    print("=" * 60)
    print("  Mini-SGLang OpenAI API Server Demo")
    print(f"  Model: {model_path}")
    print("=" * 60)

    server_proc = multiprocessing.Process(
        target=start_server, args=(model_path,), daemon=True
    )
    server_proc.start()
    time.sleep(8)

    print("\n── Health Check ──")
    print(json.dumps(_request("/health"), indent=2))

    print("\n── POST /v1/completions (sync, 3 examples) ──")
    prompts = [
        "The capital of France is",
        "The answer to life, the universe, and everything is",
        "To be or not to be,",
    ]
    for prompt in prompts:
        resp = _request(
            "/v1/completions",
            {"prompt": prompt, "max_tokens": 20, "stream": False},
        )
        text = resp.get("choices", [{}])[0].get("text", "").replace("\n", "\\n")
        print(f"  {prompt!r}")
        print(f"  → {text!r}\n")

    print("── POST /v1/completions (SSE stream) ──")
    print("  Prompt: The capital of France is")
    print("  → ", end="", flush=True)
    _stream_request(
        "/v1/completions",
        {"prompt": "The capital of France is", "max_tokens": 15, "stream": True},
    )

    print("\n── POST /v1/chat/completions (sync) ──")
    resp = _request(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "The weather today is"}],
            "max_tokens": 15,
            "stream": False,
        },
    )
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    content = content.replace("\n", "\\n") if content else "(empty)"
    print("  User: The weather today is")
    print(f"  Assistant: {content}")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE — server stopping")
    print("=" * 60)
    server_proc.terminate()
    server_proc.join(timeout=5)
