# Mini-SGLang

**Mini-SGLang** 是 [SGLang](https://github.com/sgl-project/sglang) 的轻量级教学实现，用 ~4,300 行 Python 完整复刻了一个高性能 LLM 推理框架的核心机制。项目拆解了现代 LLM 服务系统的每一个关键环节，让开发者能够逐行理解推理引擎的内部工作原理。

## 核心特性

- **Continuous Batching** — Prefill / Decode 两阶段分离调度，最大化吞吐
- **PagedAttention** — 页式 KV Cache 管理，消除显存碎片
- **RadixCache** — 基于 Radix Tree 的前缀感知缓存，自动复用公共前缀
- **CUDA / NPU Graph** — Decode 阶段 kernel launch overhead 优化（支持 NVIDIA CUDA 和华为 Ascend NPU）
- **Tensor Parallelism** — Column / Row Parallel Linear + NCCL/HCCL 多卡扩展
- **多设备支持** — NVIDIA CUDA / 华为 Ascend NPU / CPU，自动检测优先级
- **可插拔 Attention 后端** — FlashAttention (`fa`) / FlashInfer (`fi`) / PyTorch SDPA (`pt`)
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
KV Cache Pool + RadixCache + CUDA/NPU Graphs
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
    ├── device.py         # 设备管理（CUDA/NPU/CPU）+ 分布式初始化（NCCL/HCCL）
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
# 单 GPU (CUDA)
python -m minisgl --model-path Qwen/Qwen2.5-0.5B --port 8000

# 华为 Ascend NPU
python -m minisgl --model-path Qwen/Qwen3-0.6B --device npu --attention-backend pt

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
| `--tp-size` | `1` | Tensor Parallelism 卡数 |
| `--device` | `auto` | 设备类型：`auto` / `cuda` / `npu` / `cpu` |
| `--memory-ratio` | `0.9` | KV Cache 可用显存比例 |
| `--max-running-req` | `256` | 最大并发请求数 |
| `--max-seq-len` | `8192` | 最大序列长度 |
| `--page-size` | `16` | KV Cache 页大小（tokens） |
| `--attention-backend` | `fa` | 注意力后端：`fa` / `fi` / `fa,fi` / `pt` |
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

### 采样策略对比 Demo（无需下载模型）

```bash
python examples/sampling_demo.py
```

对比 greedy / temperature / top-k / top-p 不同采样策略对生成多样性的影响。

### 流式生成（含性能统计）

```bash
python examples/streaming_demo.py --model-path /path/to/model
```

### 性能基准测试

```bash
python examples/benchmark.py --model-path /path/to/model
```

### 多轮对话

```bash
python examples/multi_turn_chat.py --model-path /path/to/model
```

## 华为 Ascend NPU 支持

Mini-SGLang 完整支持华为昇腾 NPU 推理，通过 `torch_npu` 实现设备无关的张量计算。

### 环境要求

- CANN 9.0.0+
- PyTorch 2.5+ 与匹配的 torch_npu 版本
- 推荐使用 Docker 镜像：`torchtitan-npu:cann9.0.0-torch2.12.0`

### NPU 快速启动

```bash
# 使用 Docker（推荐）
docker run --privileged --shm-size=16g \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /path/to/models:/models \
  torchtitan-npu:cann9.0.0-torch2.12.0 \
  python -m minisgl --model-path /models/Qwen3-0.6B --device npu --attention-backend pt

# NPU 推理 Demo
python examples/npu_demo.py --model-path /path/to/Qwen3-0.6B --device npu

# 多模型 NPU 测试
python examples/npu_multi_model_test.py --models Qwen3-0.6B Qwen2.5-0.5B Qwen2.5-1.5B
```

### NPU 适配要点

| 项目 | 适配方案 |
|------|---------|
| 设备检测 | `torch.npu.is_available()` 自动检测，优先级 NPU > CUDA > CPU |
| Attention | 使用 PyTorch SDPA（`--attention-backend pt`），torch_npu 提供硬件加速 |
| Graph 加速 | `torch.npu.NPUGraph()` 替代 CUDA Graph（需 `TASK_QUEUE_ENABLE=1`） |
| 分布式通信 | HCCL 后端自动选择（替代 NCCL） |
| 内存管理 | `torch.npu.mem_get_info()` 统一显存查询 |
| 索引操作 | 所有高级索引使用 `int64`（NPU 要求） |

### NPU 性能测试结果

**测试环境**: `torchtitan-npu:cann9.0.0-torch2.12.0` | Ascend 910 (64GB HBM) | torch 2.12.0 + torch_npu 2.12.0rc1

| 模型 | 状态 | 加载时间 | Prefill | Decode | 批量(3x)吞吐 |
|------|------|---------|---------|--------|-------------|
| **Qwen2.5-0.5B** | PASS | 5.1s | 522.1 tok/s | 31.6 tok/s | 91.9 tok/s |
| **Qwen2.5-1.5B** | PASS | 15.3s | 451.8 tok/s | 24.4 tok/s | 73.8 tok/s |
| **Qwen2.5-3B** | PASS | 43.3s | 47.1 tok/s | 13.4 tok/s | 52.5 tok/s |
| **Qwen3-0.6B** | PASS | 11.1s | 68.7 tok/s | 18.0 tok/s | 67.8 tok/s |
| **Qwen3-1.7B** | PASS | 13.2s | 440.9 tok/s | 24.1 tok/s | 69.9 tok/s |
| **Qwen3-4B** | PASS | 31.7s | 41.5 tok/s | 10.5 tok/s | 39.3 tok/s |

> 6/6 模型全部通过，生成结果语义正确。Eager 模式推理稳定可靠。

## 支持的模型

| 模型 | 架构要点 | CUDA 验证 | NPU 验证 |
|------|---------|-----------|----------|
| **OPT** | LayerNorm + 可学习位置编码 + ReLU FFN | OPT-125M | OPT-125M |
| **Qwen2** | RMSNorm + RoPE + Gated MLP (SwiGLU) + Q/K bias | Qwen2.5-0.5B | Qwen2.5-{0.5B, 1.5B, 3B} |
| **Qwen3** | Qwen2 + QK LayerNorm + GQA | Qwen3-0.6B | Qwen3-{0.6B, 1.7B, 4B} |
| **Qwen3-MoE** | Qwen3 + MoE Router + FusedMoE | (单元测试) | (单元测试) |
| **Llama** | 经典架构 + tie_word_embeddings | (单元测试) | (单元测试) |
| **Mistral** | Llama + Sliding Window Attention | (单元测试) | (单元测试) |

## 运行测试

```bash
# 单元测试（66 个测试，CPU 即可，~30s）
python tests/test_cpu_core.py

# 多模型集成测试（跨 OPT / Qwen2 / Qwen3 三种架构）
python tests/test_examples.py

# NPU 环境测试（需要 Docker + Ascend 硬件）
docker run --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v $(pwd):/workspace -w /workspace torchtitan-npu:cann9.0.0-torch2.12.0 \
  python tests/test_cpu_core.py
```

### 测试覆盖

- **模型层**: RMSNorm, RoPE, Linear, Embedding, Attention Backend, 各模型 forward pass
- **缓存**: KVCachePool alloc/free, RadixCache prefix/evict/remove, NaiveCacheManager LRU
- **调度**: PrefillManager, DecodeManager, BatchContext, Req 生命周期
- **采样**: Greedy, Top-K, Top-P, Temperature, 边界情况
- **设备**: NPU 检测, 设备切换, 内存查询, 分布式后端选择
- **集成**: 端到端调度循环, 多请求并发, EOS 终止, OpenAI API 端点, SSE 流式
- **前端**: FrontendManager 请求提交/获取/清理

## 参考项目

- [SGLang](https://github.com/sgl-project/sglang) — 架构蓝本
- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention 参考
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — Attention kernel
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — Attention kernel
- [torch_npu](https://gitee.com/ascend/pytorch) — 华为 Ascend NPU PyTorch 后端
