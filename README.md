# Mini-SGLang

<p align="center">
  <img src="docs/images/mini-sglang.png" alt="Mini-SGLang overview">
</p>

<p align="center">
  English | <a href="README.zh-CN.md">中文</a>
</p>

**Mini-SGLang** is a lightweight educational implementation of [SGLang](https://github.com/sgl-project/sglang), replicating the core mechanisms of a high-performance LLM inference framework in ~4,800 lines of Python. The project dissects every key component of a modern LLM serving system so developers can understand how an inference engine works, line by line.

## Features

- **Continuous Batching** — separate Prefill / Decode scheduling to maximize throughput
- **PagedAttention** — paged KV cache management, eliminating GPU memory fragmentation
- **RadixCache** — radix-tree prefix-aware caching with automatic shared-prefix reuse
- **CUDA / NPU Graph** — decode-phase kernel launch overhead elimination (NVIDIA CUDA and Huawei Ascend NPU)
- **Tensor Parallelism** — Column / Row Parallel Linear layers are ready; multi-process TP launch is not yet implemented (`--tp-size > 1` exits with an error)
- **Multi-device** — NVIDIA CUDA / Huawei Ascend NPU / CPU, with automatic detection priority
- **Pluggable Attention Backends** — FlashAttention (`fa`, auto-falls back to PyTorch SDPA when unavailable) / PyTorch SDPA (`pt`)
- **OpenAI-compatible API** — `/v1/chat/completions` + `/v1/completions` + SSE streaming

## Architecture

```
Client (HTTP/SSE)
    │
    ▼
Frontend (FastAPI) ── Tokenizer (HF tokenizer)
    │
    ▼
Scheduler ── PrefillManager + DecodeManager
    │
    ▼
Engine (Model Forward + Sampling)
    │
    ▼
KV Cache Pool + RadixCache + CUDA/NPU Graphs
```

### Request Lifecycle

1. **Ingress**: FastAPI receives a request → Tokenizer encodes → Scheduler enqueues
2. **Prefill**: allocate KV cache pages → forward all prompt tokens in parallel → produce the first output token
3. **Decode**: generate token by token → CUDA Graph replay → Sampler samples → append to input_ids
4. **Egress**: Scheduler → Detokenize → Frontend → SSE stream back to the client
5. **Termination**: EOS token, `max_tokens` reached, or client disconnect

## Quick Start

### Installation

```bash
pip install -e .
# or install dependencies manually
pip install torch transformers fastapi uvicorn safetensors
# optional: high-performance attention backend
pip install flash-attn
```

### Launching the Server

```bash
# Single GPU (CUDA)
python -m minisgl --model-path Qwen/Qwen3-0.6B --port 8000

# Huawei Ascend NPU
python -m minisgl --model-path Qwen/Qwen3-0.6B --device npu --attention-backend pt

# Interactive shell
python -m minisgl --model-path Qwen/Qwen3-0.6B --shell
```

> Note: `--tp-size > 1` is not supported yet — multi-process TP launch is not
> implemented (the Column/Row Parallel layer logic is ready), and passing it
> exits with an error.

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | (required) | HuggingFace model path |
| `--host` | `127.0.0.1` | API listen address |
| `--port` | `8000` | API port |
| `--tp-size` | `1` | Tensor parallelism size; `>1` exits with an error (multi-process TP launch not implemented) |
| `--device` | `auto` | Device type: `auto` / `cuda` / `npu` / `cpu` |
| `--memory-ratio` | `0.9` | Fraction of GPU memory available to the KV cache |
| `--max-running-req` | `256` | Maximum concurrent requests |
| `--max-seq-len` | `8192` | Maximum sequence length |
| `--page-size` | `16` | KV cache page size (tokens) |
| `--cuda-graph-bs` | `None` | Maximum batch size for graph capture (CUDA/NPU) |
| `--attention-backend` | `fa` | Attention backend: `fa` (FlashAttention, falls back to PyTorch SDPA) / `pt` (PyTorch SDPA) |
| `--dtype` | `auto` | Model precision: `auto` (reads config.json) / `float16` / `bfloat16` / `float32` (CPU forces float32) |
| `--trust-remote-code` | `False` | Trust custom code from HF models |
| `--log-level` | `INFO` | Log level: `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `--shell` | `False` | Interactive CLI mode |

### Python API

```python
from minisgl.engine.llm import LLM

llm = LLM(model_path="Qwen/Qwen3-0.6B")

# Single prompt
output = llm.generate("Hello, who are you?", max_tokens=128)

# Batch generation
outputs = llm.generate(["Hello", "What is AI?"], temperature=0.7)

# Chat interface
messages = [{"role": "user", "content": "What is 2+2?"}]
response = llm.chat(messages, max_tokens=128)
```

### API Endpoints

```bash
# Chat Completions
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'

# Text Completions
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Once upon a time","max_tokens":30,"stream":false}'

# Health Check
curl http://127.0.0.1:8000/health
```

## Examples

```
examples/
├── _common.py           # Shared helpers: model discovery, engine setup, generation loop
├── cpu_demo.py          # Dependency-free self-contained CPU demo (no model download)
├── offline_inference.py # Comprehensive offline inference: batch, LLM API, streaming, sampling strategies
├── benchmark.py         # Performance benchmark: prefill latency, decode throughput, end-to-end
├── server_demo.py       # OpenAI-compatible API server + automated tests
└── npu_inference.py     # Huawei Ascend NPU inference + multi-model tests
```

### Self-contained CPU Demo (no model download)

```bash
python examples/cpu_demo.py
```

Creates a temporary random-weight toy model and validates the full Engine → Scheduler → Generate pipeline.

### Comprehensive Offline Inference

```bash
python examples/offline_inference.py --model-path /path/to/model
```

Covers: Engine+Scheduler batch inference, the high-level LLM.generate()/chat() API, token-by-token streaming (with TTFT/throughput metrics), and sampling strategy comparison (greedy/temperature/top-p/top-k).

### Performance Benchmark

```bash
python examples/benchmark.py --model-path /path/to/model
```

### OpenAI-compatible API Server

```bash
python examples/server_demo.py --model-path /path/to/model
```

Starts the FastAPI server and automatically tests health check / sync completion / SSE streaming / chat.

### NPU Inference

```bash
python examples/npu_inference.py --model-path /path/to/model --device npu
python examples/npu_inference.py --models Qwen3-0.6B Qwen3-1.7B Qwen3-4B
```

## Huawei Ascend NPU Support

Mini-SGLang fully supports Huawei Ascend NPU inference via `torch_npu` for device-agnostic tensor computation.

### Requirements

- CANN 9.0.0+
- PyTorch 2.5+ with a matching torch_npu version
- Recommended Docker image: `torchtitan-npu:cann9.0.0-torch2.12.0`

### NPU Quick Start

```bash
# With Docker (recommended)
docker run --privileged --shm-size=16g \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /path/to/models:/models \
  torchtitan-npu:cann9.0.0-torch2.12.0 \
  python -m minisgl --model-path /models/Qwen3-0.6B --device npu --attention-backend pt

# NPU single-model inference
python examples/npu_inference.py --model-path /path/to/Qwen3-0.6B --device npu

# NPU multi-model batch test
python examples/npu_inference.py --models Qwen3-0.6B Qwen3-1.7B Qwen3-4B
```

### NPU Adaptation Notes

| Item | Approach |
|------|----------|
| Device detection | `torch.npu.is_available()` auto-detection, priority NPU > CUDA > CPU |
| Attention | PyTorch SDPA (`--attention-backend pt`), hardware-accelerated by torch_npu |
| Graph speedup | `torch.npu.NPUGraph()` instead of CUDA Graph (requires `TASK_QUEUE_ENABLE=1`) |
| Distributed comms | HCCL backend selected automatically (replaces NCCL) |
| Memory management | `torch.npu.mem_get_info()` for unified memory queries |
| Indexing | All advanced indexing uses `int64` (required by NPU) |

### NPU Performance Results

**Test environment**: `torchtitan-npu:cann9.0.0-torch2.12.0` | Ascend 910 (64GB HBM) | torch 2.12.0 + torch_npu 2.12.0rc1

| Model | Status | Load time | Prefill | Decode | Batch (3x) throughput |
|-------|--------|-----------|---------|--------|-----------------------|
| **Qwen3-0.6B** | PASS | 13.0s | 57.4 tok/s | 14.3 tok/s | 54.0 tok/s |
| **Qwen3-1.7B** | PASS | 13.0s | 201.4 tok/s | 13.5 tok/s | 36.0 tok/s |
| **Qwen3-4B** | PASS | 24.4s | 260.9 tok/s | 13.0 tok/s | 39.2 tok/s |

> All 3/3 models passed with semantically correct outputs. Eager-mode inference is stable and reliable.
>
> Test command: `python examples/npu_inference.py --models Qwen3-0.6B Qwen3-1.7B Qwen3-4B --max-tokens 30`

## Supported Models

Mini-SGLang supports the **Qwen3 family** of dense decoder-only transformers.

| Model | Architecture highlights | CUDA verified | NPU verified |
|-------|------------------------|---------------|--------------|
| **Qwen3** | RMSNorm + RoPE + SwiGLU + QK LayerNorm + GQA | Qwen3-0.6B | Qwen3-{0.6B, 1.7B, 4B} |
| **Qwen3-MoE** | Qwen3 + sparse MoE router (softmax → top-k → normalize, HF-aligned) | (unit tests) | (unit tests) |

## Running Tests

```bash
# Unit tests (99 tests, CPU-only, ~15s)
python tests/test_cpu_core.py

# Example smoke tests (cpu_demo needs no model; model cases take local model
# paths via MINISGL_TEST_MODELS and are skipped when unset)
python tests/test_examples.py

# NPU environment tests (requires Docker + Ascend hardware)
docker run --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v $(pwd):/workspace -w /workspace torchtitan-npu:cann9.0.0-torch2.12.0 \
  python tests/test_cpu_core.py
```

### Test Coverage

- **Model layers**: RMSNorm, RoPE, Linear, Embedding, attention backends, per-model forward passes
- **Caching**: KVCachePool alloc/free, RadixCache prefix/evict/remove, NaiveCacheManager
- **Scheduling**: PrefillManager, DecodeManager, BatchContext, Req lifecycle
- **Sampling**: greedy, top-k, top-p, temperature, edge cases
- **Devices**: NPU detection, device switching, memory queries, distributed backend selection
- **Integration**: end-to-end scheduling loop, multi-request concurrency, EOS termination, OpenAI API endpoints, SSE streaming
- **Frontend**: FrontendManager request submit/fetch/cleanup

## References

- [SGLang](https://github.com/sgl-project/sglang) — architectural blueprint
- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention reference
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — attention kernels
- [torch_npu](https://gitee.com/ascend/pytorch) — Huawei Ascend NPU PyTorch backend
