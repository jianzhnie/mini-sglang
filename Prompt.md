# Mini-SGLang

## 项目定位

**Mini-SGLang** 是 [SGLang](https://github.com/sgl-project/sglang) 的轻量级教学实现，目标是用 ~8,000 行 Python 完整复刻一个高性能 LLM 推理框架。它拆解了现代 LLM 服务系统的每一个关键环节，让开发者能够逐行理解推理引擎的内部工作原理。

**解决的核心问题：**
- 如何在 GPU 上高效运行大语言模型推理（Prefill/Decode 分离、CUDA Graph 加速）
- 如何在多个请求间共享 GPU 显存和计算资源（Continuous Batching、PagedAttention、Radix Cache）
- 如何在单 GPU 显存受限的情况下服务长上下文（KV Cache 分页管理、前缀共享）
- 如何跨多 GPU 扩展推理能力（Tensor Parallelism）

**目标用户：**
- 学习和研究 LLM 推理系统的开发者、研究者
- 需要一个轻量、可定制推理引擎的产品团队
- 想理解 SGLang/vLLM 内部原理的工程师

---

## 整体架构

Mini-SGLang 采用**多进程 + 消息传递**架构，核心由以下进程组成：

```
Client (HTTP/SSE)
    │
    ▼
Frontend (FastAPI) ──ZMQ── Tokenizer (HF tokenizer worker)
    │                           │
    │ TokenizeMsg               │ DetokenizeMsg
    ▼                           ▼
Scheduler (Python) ──TCP/broadcast── Scheduler (其他 TP ranks)
    │
    ▼
Engine (CUDA Stream) — 模型 forward + 采样
```

- **Frontend**: FastAPI HTTP 服务，接收 `/v1/chat/completions` 和 `/v1/completions` 请求，分配 UID，将文本 tokenize 后发给 Scheduler，并将结果以 SSE 流式返回客户端
- **Tokenizer**: 独立进程，调用 HuggingFace tokenizer 进行 encode/decode，通过 ZMQ 与 Frontend 和 Scheduler 通信
- **Scheduler**: 每个 TP rank 一个 Scheduler 实例，rank0 的 Scheduler 负责调度决策（prefill/decode 批处理），并通过 TCP（gloo backend）广播到所有 rank
- **Engine**: 持有模型权重、KV Cache、Attention Backend，执行实际的 forward 计算和采样

---

## 代码组织

```bash
minisgl/engine：  实现 执行引擎，KV 缓存管理， 分布式通信 模块
minisgl/models：  实现 模型架构，MoE 后端， 神经网络层， 注意力后端， 分词/解码 worker
minisgl/scheduler：调度核心， Prefill & Decode
minisgl/sampling: 实现 采样策略
minisgl/server:   API 服务 + 启动
minisgl/utils：   工具集（权重加载，日志，device 帮助函数）
minisgl/config.py: 参数配置模块
```
---

## 核心数据流（请求生命周期）

下面以一次完整的 `/v1/chat/completions` 请求为例，描述请求从接入到返回的全过程：

### 1. 接入与分词
1. FastAPI 收到 POST 请求（JSON），FrontendManager 分配唯一的 `uid`
2. 构造 `TokenizeMsg(uid, text)`，通过 ZMQ PUSH socket 发送到 Tokenizer 进程
3. TokenizerWorker 调用 HF tokenizer 将文本转为 token IDs

### 2. 调度
4. Tokenizer 构造 `UserMsg(uid, input_ids, sampling_params)`，通过 ZMQ 发给 Scheduler Rank0
5. Scheduler Rank0 通过 TCP（gloo broadcast）将请求信息广播到所有 TP Rank
6. 每个 rank 的 PrefillManager 将请求加入 pending 队列

### 3. Prefill（预填充）
7. PrefillManager 从 pending 队列取出请求，检查 KV Cache 可用空间、table 槽位、token budget
8. 分配 KV Cache pages（优先通过 RadixCache 复用已有前缀）
9. 将 prompt tokens 打包成 `Batch`（phase="prefill"），构造 `ForwardInput`：
   - `input_ids`: token 索引
   - `positions`: 位置编码
   - `write_loc`: KV Cache 写入位置（page table 索引）
10. Engine 执行一次完整 forward：**所有 prompt tokens 并行处理**
11. 生成第一个 output token，将 KV Cache 状态写入 cache pool
12. 请求从 PrefillManager 移入 DecodeManager（状态：pending → running）

### 4. Decode（逐 token 生成）
13. DecodeManager 将所有 running 请求打包成 `Batch`（phase="decode"）
14. 每个 decode step，Engine 仅处理**一个**新 token（利用已缓存的 KV）
15. 通过 **CUDA Graph** 回放加速 decode（消除 kernel launch overhead）
16. Sampler 对 logits 做 top-p / top-k / temperature 采样得到下一个 token
17. 新 token 追加到请求的 input_ids，更新 page table

### 5. 拆词与返回
18. Scheduler Rank0 将生成的 token 封装为 `DetokenizeMsg(uid, token_id, finished)` 发给 DetokenizerWorker
19. Detokenizer 将 token 解码为文本，通过 ZMQ 发给 FrontendManager
20. FrontendManager 以 **SSE (Server-Sent Events)** 格式流式返回给客户端
21. 步骤 13-20 循环直到：遇到 EOS token 或达到 `max_tokens` 或客户端断开

---

## 关键数据结构

### Req（请求）
```python
@dataclass
class Req:
    input_ids: Tensor          # CPU tensor，累积的 token 序列
    table_idx: int             # 请求在 page table 中的行索引
    cached_len: int            # 已缓存 KV 的 token 数（前缀匹配后）
    output_len: int            # 还需生成的 token 数
    uid: int                   # 全局唯一请求 ID
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle  # 指向分配的 KV Cache 块
```

### Batch（批次）
```python
@dataclass
class Batch:
    reqs: List[Req]            # 本次 forward 包含的请求
    phase: Literal["prefill", "decode"]
    # 由 Context 管理的派生字段：
    # input_ids, positions, write_loc (page table indices)
```

### SamplingParams（采样参数）
```python
@dataclass
class SamplingParams:
    temperature: float = 0.0   # 0 = 贪心解码
    top_k: int = -1            # -1 = 不限制
    top_p: float = 1.0         # 1.0 = 不限制
    ignore_eos: bool = False
    max_tokens: int = 1024
```

---

## 关键架构决策

### Attention 后端（可插拔）
- **FlashInfer** (`fi`): 推荐，支持 Prefill/Decode 分离优化
- **FlashAttention** (`fa`): 通用后端，兼容性好
- 支持 **Hybrid Backend**: Prefill 和 Decode 使用不同的后端（如 `fa,fi`），最大化各自场景的吞吐

### KV Cache 管理
- **PagedAttention**: 将 KV Cache 划分为固定大小的 page（默认 16/32/64 tokens）
- **RadixCache**: 基于 Radix Tree 的前缀感知缓存管理器，自动检测和复用公共前缀
- **NaiveCache**: 简化版 LRU 缓存，用于无前缀共享场景
- Page Table 是一个 `(max_running_req + 1, max_seq_len)` 的 int32 tensor，存储每个 token 位置对应的 KV Cache page 索引

### 分布式策略
- **Tensor Parallelism (TP)**: 将权重矩阵按列/行切分到多个 GPU
  - `ColumnParallelLinear`: 沿 hidden_dim 切分（gather output）
  - `RowParallelLinear`: 沿 input_dim 切分（all-reduce output）
  - `VocabParallelEmbedding`: 沿 vocab_dim 切分
- **PyNCCL**: 用 Python 封装的 NCCL 通信原语（all-reduce, all-gather, broadcast）
- Scheduler Rank0 做决策后通过 TCP 广播，确保所有 rank 的 batch 顺序一致

### CUDA Graph 优化
- 对 decode 阶段的每种 batch size（1, 2, ..., max_cuda_graph_bs）预先捕获 CUDA Graph
- 运行时直接 replay（消除 kernel launch overhead）
- Prefill 阶段因输入长度变化大，不使用 CUDA Graph

### 内存管理
- 模型在 `meta` device 上初始化（不分配实际显存），加载权重时再分配
- KV Cache 根据剩余显存动态计算可分配的 page 数量
- 通过 `memory_ratio` 参数控制 KV Cache 可使用的显存比例

---

## 支持的模型

### 基础要求
- **Qwen2** — Qwen2ForCausalLM（推荐首选实现）
- **Qwen3** — Qwen3ForCausalLM（支持 GQA + QK Norm）
- **Qwen3-MoE** — Qwen3MoEForCausalLM（含 MoE Router + FusedMoE）
- **Llama** — LlamaForCausalLM（最经典的架构参考）
- **Mistral** — MistralForCausalLM（含 Sliding Window Attention）

### 模型架构需支持的组件
| 组件 | 说明 |
|------|------|
| RMSNorm | 含 fused residual add 优化 |
| RoPE | 旋转位置编码，支持动态序列长度 |
| GQA (Grouped Query Attention) | Q heads 数可以不同于 KV heads 数 |
| Gated MLP (SwiGLU) | gate_proj + up_proj + down_proj 三矩阵 |
| MoE (Mixture of Experts) | Top-k routing + Fused Triton kernel |
| QK LayerNorm | Qwen3 特有的 attention Q/K normalization |
| Tensor Parallel | Column/Row parallel linear, Vocab parallel embedding |
| Tie Word Embeddings | lm_head 与 embed_tokens 共享权重 |

---

## API 端点

### `/v1/chat/completions` (POST)
兼容 OpenAI Chat Completions API：
```json
{
  "model": "qwen2",
  "messages": [{"role": "user", "content": "Hello"}],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 1024,
  "stream": true
}
```

### `/v1/completions` (POST)
兼容 OpenAI Completions API（文本补全模式）。

### 响应格式
- `stream=true`（默认）：SSE 流式返回 `data: {"choices": [{"delta": {"content": "..."}}]}`
- `stream=false`：单次 JSON 响应


### 启动方式
```bash
# 启动服务（TP=1）
python -m minisgl --model-path Qwen/Qwen2-0.5B-Instruct --port 8000

# 多 GPU Tensor Parallel
python -m minisgl --model-path Qwen/Qwen2-7B-Instruct --tp-size 4

# 交互式 CLI shell
python -m minisgl --model-path Qwen/Qwen2-0.5B-Instruct --shell
```

---

### 代码格式规范

所有 Python 代码必须遵循以下规范：

1. **Type Hints**: 所有公开函数/方法必须添加完整的类型标注（参数、返回值、泛型类型参数）。使用 `list[T]` 而非 `List[T]`（Python 3.10+ 语法）。复杂类型用 `typing` 模块的 `TypeAlias` 或自定义 `Protocol` 表达。
2. **Docstrings**: 公开 API（类和公开方法）使用 Google 风格的 docstring，包含 `Args`、`Returns`、`Raises` 段落。内部/私有方法如有非显而易见的逻辑需加单行注释说明 **Why**（而非 What）。
3. **代码组织**: 每个 `.py` 文件只负责一个明确的职责；import 顺序为 stdlib → third-party → first-party（由 `isort` 自动管理）；模块级 `__all__` 用于控制公开接口。
4. **格式化**: 使用 `ruff format` 统一格式化（行宽 88、4 空格缩进），`ruff check` 进行 lint。CI 中通过 pre-commit hook 自动执行，不通过则不允许提交。
5. **命名约定**: 类名 `PascalCase`，函数/变量 `snake_case`，常量 `UPPER_SNAKE_CASE`，私有属性以单下划线 `_` 开头。避免单字母变量名（循环索引 `i`、`j` 除外）。
6. **最佳实践**: 优先使用 `dataclass` 定义数据结构；上下文管理器管理资源生命周期；避免可变默认参数；`if __name__ == "__main__":` 保护入口代码。



## 参考项目
- [SGLang](https://github.com/sgl-project/sglang) — 架构蓝本
- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention 参考
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — Attention kernel
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — Attention kernel
- [mini-vllm](https://github.com/jianzhnie/mini-vllm) — 姊妹教学项目
