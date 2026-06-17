# Mini-SGLang

**Mini-SGLang** 是 [SGLang](https://github.com/sgl-project/sglang) 的轻量级教学实现，用 ~4,300 行 Python 完整复刻了一个高性能 LLM 推理框架的核心机制。项目拆解了现代 LLM 服务系统的每一个关键环节，让开发者能够逐行理解推理引擎的内部工作原理。

## 核心特性

- **Continuous Batching** — Prefill / Decode 两阶段分离调度，最大化吞吐
- **PagedAttention** — 页式 KV Cache 管理，消除显存碎片
- **RadixCache** — 基于 Radix Tree 的前缀感知缓存，自动复用公共前缀
- **CUDA Graph** — Decode 阶段 kernel launch overhead 优化
- **Tensor Parallelism** — Column / Row Parallel Linear + PyNCCL 多 GPU 扩展
- **可插拔 Attention 后端** — FlashAttention (`fa`) / FlashInfer (`fi`) / PyTorch fallback
- **OpenAI 兼容 API** — `/v1/chat/completions` + `/v1/completions` + SSE 流式返回

## 架构概览

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
KV Cache Pool + RadixCache + CUDA Graphs
```

### 请求生命周期

1. **接入**: FastAPI 接收请求 → Tokenizer encode → Scheduler 入队
2. **Prefill**: 分配 KV Cache pages → 全量 prompt tokens 并行 forward → 生成首个 output token
3. **Decode**: 逐 token 生成 → CUDA Graph 回放加速 → Sampler 采样 → 追加到 input_ids
4. **返回**: Scheduler → Detokenize → Frontend → SSE 流式返回客户端
5. **终止**: 遇到 EOS 或达到 `max_tokens` 或客户端断开

## 代码组织

```
minisgl/
├── config.py           # 配置：ServerArgs, ModelArgs, CacheArgs, SamplingParams
├── __init__.py          # CLI 入口（--shell / --port / --tp-size）
├── engine/
│   ├── engine.py        # 推理引擎：模型加载、forward、CUDA Graph
│   ├── context.py       # BatchContext：批次张量管理
│   ├── llm.py           # 高层 LLM API（离线推理）
│   ├── kvcache/
│   │   ├── pool.py      # KV Cache 页式内存池
│   │   ├── radix.py     # Radix Tree 前缀共享缓存
│   │   └── naive.py     # LRU 简化缓存（无前缀共享）
│   └── distributed/
│       └── pynccl.py    # NCCL 通信原语封装
├── models/
│   ├── decoder.py       # 共享基类：GatedMLP, RMSNormDecoderLayer, RMSNormForCausalLM
│   ├── opt.py           # OPT 模型（LayerNorm + 可学习位置编码 + ReLU FFN）
│   ├── qwen2.py         # Qwen2 模型（RMSNorm + Gated MLP + Q/K bias）
│   ├── qwen3.py         # Qwen3 模型（+ QK LayerNorm + GQA）
│   ├── qwen3_moe.py     # Qwen3-MoE 模型（MoE Router + FusedMoE）
│   ├── llama.py         # Llama 模型（经典架构）
│   ├── mistral.py       # Mistral 模型（+ Sliding Window Attention）
│   ├── registry.py      # 模型注册与自动检测
│   ├── layers/
│   │   ├── attention.py # BaseAttention 模板方法
│   │   ├── rms_norm.py  # RMSNorm（含 fused residual add）
│   │   ├── rope.py      # Rotary Position Embedding
│   │   ├── linear.py    # Column / Row Parallel Linear
│   │   └── embedding.py # Vocab Parallel Embedding
│   ├── attention/
│   │   └── backend.py   # FlashAttention / FlashInfer / PyTorch 后端
│   ├── moe/
│   │   └── fused_moe.py # FusedMoE Triton kernel + PyTorch 实现
│   └── tokenizer/
│       └── worker.py    # HF Tokenizer Worker
├── scheduler/
│   ├── scheduler.py     # 主调度器：prefill/decode 协调
│   ├── prefill.py       # PrefillManager：pending 队列 + 令牌预算
│   ├── decode.py        # DecodeManager：running 队列管理
│   └── batch.py         # Req / Batch 数据结构
├── sampling/
│   └── sampler.py       # 采样器：greedy / top-k / top-p / temperature
├── server/
│   └── frontend.py      # FastAPI 服务：SSE 流式 + OpenAI 兼容端点
└── utils/
    ├── device.py         # 设备管理 + 分布式初始化
    ├── logger.py         # 统一日志
    └── weights.py        # HuggingFace 权重加载 + TP 分片
```

### 模型继承结构

Qwen2 / Qwen3 / Llama / Mistral 共用 `decoder.py` 中的基类，消除重复代码：

```
RMSNormForCausalLM          ← Qwen2/Qwen3/Llama/Mistral/Qwen3MoE 继承
  └── RMSNormModel          ← 共享 embed → layers → norm 流程
        └── RMSNormDecoderLayer  ← 共享 LN → Attn → LN → MLP 流程
              └── GatedMLP       ← SwiGLU 实现，所有模型共用
```

每个模型文件只需定义自己独特的 Attention 类（~50 行）。

## 快速开始

### 安装

```bash
pip install -e .
# 或手动安装依赖
pip install torch transformers fastapi uvicorn safetensors
# 可选：高性能 attention 后端
pip install flash-attn flashinfer
```

### 启动服务

```bash
# 单 GPU
python -m minisgl --model-path Qwen/Qwen2.5-0.5B --port 8000

# 多 GPU Tensor Parallel
python -m minisgl --model-path Qwen/Qwen2.5-7B-Instruct --tp-size 4

# 交互式 Shell
python -m minisgl --model-path Qwen/Qwen3-0.6B --shell
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | (必填) | HuggingFace 模型路径 |
| `--host` | `127.0.0.1` | API 监听地址 |
| `--port` | `8000` | API 端口 |
| `--tp-size` | `1` | Tensor Parallelism GPU 数量 |
| `--memory-ratio` | `0.9` | KV Cache 可用显存比例 |
| `--max-running-req` | `256` | 最大并发请求数 |
| `--max-seq-len` | `8192` | 最大序列长度 |
| `--page-size` | `16` | KV Cache 页大小（tokens） |
| `--attention-backend` | `fa` | 注意力后端：`fa` / `fi` / `fa,fi` |
| `--dtype` | `auto` | 模型精度：`auto` / `float16` / `bfloat16` |
| `--shell` | `False` | 交互式 CLI 模式 |

### Python API

```python
from minisgl.engine.llm import LLM

llm = LLM(model_path="Qwen/Qwen3-0.6B")

# 单条生成
output = llm.generate("Hello, who are you?", max_tokens=128)

# 批量生成
outputs = llm.generate(["Hello", "What is AI?"], temperature=0.7)

# Chat 接口
messages = [{"role": "user", "content": "What is 2+2?"}]
response = llm.chat(messages, max_tokens=128)
```

### API 端点

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

## 示例

### CPU 自包含 Demo（无需下载模型）

```bash
python examples/cpu_demo.py
```

创建一个临时的随机权重小模型，验证完整的 Engine → Scheduler → Generate 管线。

### Engine + Scheduler 直接使用

```bash
python examples.py --model-path /path/to/model
```

### 批量推理（并发多 Prompt）

```bash
python examples/batch_inference.py --model-path /path/to/model
```

多条 prompt 同时提交，Scheduler 自动 batch 处理，各 prompt 输出独立不干扰。

### 高层 LLM API

```bash
python examples/llm_generate.py --model-path /path/to/model
```

演示 `LLM.generate()` 文本续写和 `LLM.chat()` 对话接口。

### OpenAI 兼容 API 服务

```bash
python server_demo.py --model-path /path/to/model
```

启动 FastAPI 服务器，自动测试 health check / sync completion / SSE streaming / chat。

## 支持的模型

| 模型 | 架构要点 | 已验证 |
|------|---------|--------|
| **OPT** | LayerNorm + 可学习位置编码 + ReLU FFN | OPT-125M |
| **Qwen2** | RMSNorm + RoPE + Gated MLP (SwiGLU) + Q/K bias | Qwen2.5-0.5B |
| **Qwen3** | Qwen2 + QK LayerNorm + GQA | Qwen3-0.6B |
| **Qwen3-MoE** | Qwen3 + MoE Router + FusedMoE | (单元测试) |
| **Llama** | 经典架构 + tie_word_embeddings | (单元测试) |
| **Mistral** | Llama + Sliding Window Attention | (单元测试) |

## 运行测试

```bash
# 单元测试（45 个测试，CPU 即可，~25s）
python tests/test_cpu_core.py

# 多模型集成测试（跨 OPT / Qwen2 / Qwen3 三种架构）
python tests/test_examples.py
```

### 测试覆盖

- **模型层**: RMSNorm, RoPE, Linear, Embedding, Attention Backend, 各模型 forward pass
- **缓存**: KVCachePool alloc/free, RadixCache prefix/evict/remove, NaiveCacheManager LRU
- **调度**: PrefillManager, DecodeManager, BatchContext, Req 生命周期
- **采样**: Greedy, Top-K, Top-P, Temperature
- **集成**: 多 Prompt 批量推理, OpenAI API 端点, SSE 流式

## 参考项目

- [SGLang](https://github.com/sgl-project/sglang) — 架构蓝本
- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention 参考
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — Attention kernel
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — Attention kernel
