#!/usr/bin/env python3
"""Mini-SGLang OpenAI-compatible API server demo.

Starts the real FastAPI server and exercises it end-to-end: health check,
sync completions, SSE streaming, chat completions.

Usage:
    python examples/server_demo.py
    python examples/server_demo.py --model-path /path/to/model

Then test:
    curl http://127.0.0.1:8765/health
    curl http://127.0.0.1:8765/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"prompt":"The capital of France is","max_tokens":10,"stream":false}'
"""

import json
import os
import sys
import time
import urllib.request

# The server runs in a spawned subprocess that re-imports this script, so the
# sys.path bootstrap below must run at module top level (before any import of
# _common or minisgl) — examples/ for _common, repo root for minisgl.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # examples/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # root

from _common import (  # noqa: E402
    banner,
    cli_main,
)

HOST, PORT = "127.0.0.1", 8765


def _request(path: str, body: dict | None = None) -> dict:
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


def _stream_request(path: str, body: dict) -> None:
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


def start_server(model_path: str) -> None:
    """Entry point of the spawned server subprocess."""
    from minisgl.config import ServerArgs
    from minisgl.server.serve import serve

    args = ServerArgs(
        model_path=model_path,
        host=HOST,
        port=PORT,
        tp_size=1,
        attention_backend="fa",
        max_running_req=4,
        max_seq_len=256,
        page_size=16,
        memory_ratio=0.5,
        cuda_graph_bs=0,
        log_level="INFO",
    )
    serve(args)


def main(model_path: str) -> None:
    import multiprocessing

    banner(
        "Mini-SGLang OpenAI API Server Demo",
        lines=[f"Model: {model_path}"],
    )

    server_proc = multiprocessing.Process(
        target=start_server, args=(model_path,), daemon=True
    )
    server_proc.start()

    print("\nWaiting for server to start...")
    for attempt in range(60):
        time.sleep(2)
        try:
            _request("/health")
            print(f"  Server ready after ~{(attempt + 1) * 2}s")
            break
        except Exception:
            pass
    else:
        print("ERROR: Server did not start in 120s")
        server_proc.terminate()
        sys.exit(1)

    try:
        print("\n── Health Check ──")
        print(json.dumps(_request("/health"), indent=2))

        print("\n── POST /v1/completions (sync) ──")
        for prompt in ["The capital of France is", "To be or not to be,"]:
            resp = _request(
                "/v1/completions",
                {"prompt": prompt, "max_tokens": 10, "stream": False},
            )
            text = resp.get("choices", [{}])[0].get("text", "").replace("\n", "\\n")
            print(f"  {prompt!r} → {text!r}")

        print("\n── POST /v1/completions (SSE stream) ──")
        print("  Prompt: The capital of France is")
        print("  → ", end="", flush=True)
        _stream_request(
            "/v1/completions",
            {"prompt": "The capital of France is", "max_tokens": 10, "stream": True},
        )

        print("\n── POST /v1/chat/completions (sync) ──")
        resp = _request(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "The weather today is"}],
                "max_tokens": 10,
                "stream": False,
            },
        )
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        content = content.replace("\n", "\\n") if content else "(empty)"
        print("  User: The weather today is")
        print(f"  Assistant: {content}")
    finally:
        print("\n" + "=" * 60)
        print("  DEMO COMPLETE — server stopping")
        print("=" * 60)
        server_proc.terminate()
        server_proc.join(timeout=5)


if __name__ == "__main__":
    cli_main("Mini-SGLang Server Demo", main)
