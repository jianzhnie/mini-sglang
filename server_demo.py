#!/usr/bin/env python3
"""Mini-SGLang OpenAI-compatible API server deployment with OPT-125M.

Quick start:
    python3 server_demo.py

Then test in another terminal:
    curl http://127.0.0.1:8765/health
    curl http://127.0.0.1:8765/v1/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model":"opt","prompt":"The capital of France is","max_tokens":30,"stream":false}'
    curl http://127.0.0.1:8765/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model":"opt","messages":[{"role":"user","content":"The sky is"}],"max_tokens":20,"stream":false}'

Note: OPT-125M is a base (non-instruction-tuned) model. It works best
with /v1/completions (text continuation). For /v1/chat/completions,
use an instruction-tuned model like Qwen2-0.5B-Instruct.
"""

import os
import sys
import time
import json
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.expanduser("~/hfhub/models/facebook/opt-125m")
HOST, PORT = "127.0.0.1", 8765


def _request(path: str, body: dict = None) -> dict:
    url = f"http://{HOST}:{PORT}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST")
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


def start_server():
    from minisgl.config import ModelArgs, ServerArgs
    from minisgl.engine.engine import Engine
    from minisgl.scheduler.scheduler import Scheduler
    from minisgl.server.frontend import app, init_frontend
    from minisgl.models.tokenizer.worker import TokenizerWorker
    from minisgl.utils.logger import setup_logger, logger
    import uvicorn

    setup_logger(level="INFO")

    args = ServerArgs(
        model_path=MODEL_PATH, host=HOST, port=PORT,
        tp_size=1, attention_backend="fa",
        max_running_req=8, max_seq_len=512, page_size=16,
        memory_ratio=0.5,
    )
    model_args = ModelArgs.from_pretrained(MODEL_PATH)
    tokenizer = TokenizerWorker(MODEL_PATH)
    engine = Engine(args, model_args, tp_rank=0)
    scheduler = Scheduler(args, engine)
    init_frontend(args, scheduler, tokenizer)

    logger.info(f"API server at http://{HOST}:{PORT}")
    logger.info("Endpoints: /v1/completions  /v1/chat/completions  /health")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    import multiprocessing

    print("=" * 60)
    print("  Mini-SGLang OpenAI API Server Demo")
    print(f"  Model: {MODEL_PATH}")
    print("=" * 60)

    server_proc = multiprocessing.Process(target=start_server, daemon=True)
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
        resp = _request("/v1/completions", {
            "model": "opt", "prompt": prompt,
            "max_tokens": 20, "stream": False,
        })
        text = resp.get("choices", [{}])[0].get("text", "").replace("\n", "\\n")
        print(f"  {prompt!r}")
        print(f"  → {text!r}\n")

    print("── POST /v1/completions (SSE stream) ──")
    print(f"  Prompt: The capital of France is ")
    print("  → ", end="", flush=True)
    _stream_request("/v1/completions", {
        "model": "opt", "prompt": "The capital of France is ",
        "max_tokens": 15, "stream": True,
    })

    print("\n── POST /v1/chat/completions (sync) ──")
    print("  Note: OPT is a base model, output may be low quality.")
    resp = _request("/v1/chat/completions", {
        "model": "opt",
        "messages": [
            {"role": "user", "content": "The weather today is"},
        ],
        "max_tokens": 15, "stream": False,
    })
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    content = content.replace("\n", "\\n") if content else "(empty)"
    print(f"  User: The weather today is")
    print(f"  Assistant: {content}")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE — server stopping")
    print("=" * 60)
    server_proc.terminate()
    server_proc.join(timeout=5)
