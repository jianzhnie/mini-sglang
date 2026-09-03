# Mini-SGLang

## 项目定位

**Mini-SGLang** 是 [SGLang](https://github.com/sgl-project/sglang) 的轻量级教学实现，目标是用 ~4,800 行 Python 完整复刻一个高性能 LLM 推理框架。它拆解了现代 LLM 服务系统的每一个关键环节，让开发者能够逐行理解推理引擎的内部工作原理。

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

Mini-SGLang 采用**单进程**架构（教学简化：不做 SGLang 的多进程 ZMQ 拆分），核心组件：

```
Client (HTTP/SSE)
    │
    ▼
Frontend (FastAPI) ── TokenizerWorker (HF tokenizer，进程内调用)
    │
    ▼
Scheduler (后台线程，单实例)
    │
    ▼
Engine (CUDA Stream) — 模型 forward + 采样
```

- **Frontend**: FastAPI HTTP 服务，接收 `/v1/chat/completions` 和 `/v1/completions` 请求，分配 UID，将文本 tokenize 后发给 Scheduler，并将结果以 SSE 流式返回客户端
- **Tokenizer**: 进程内 `TokenizerWorker`，调用 HuggingFace tokenizer 进行 encode/decode（流式返回时使用增量 detokenize）
- **Scheduler**: 单实例，负责调度决策（prefill/decode 批处理）。TP>1 的多进程启动（每 rank 一个 Scheduler + TCP 广播）尚未实现，CLI 会拒绝 `--tp-size > 1`
- **Engine**: 持有模型权重、KV Cache、Attention Backend，执行实际的 forward 计算和采样

---

## 代码组织

```bash
minisgl/engine：  实现 执行引擎，KV 缓存管理， 分布式通信 模块
minisgl/models：  实现 模型架构，神经网络层，注意力后端
minisgl/scheduler：调度核心， Prefill & Decode
minisgl/sampling.py: 实现 采样策略
minisgl/server:   API 服务 + 启动
minisgl/utils：   工具集（权重加载，日志，device 帮助函数）
minisgl/config.py: 参数配置模块
minisgl/tokenizer.py: 分词/解码 worker（HF tokenizer，进程内调用）
```
---

## 核心数据流（请求生命周期）

下面以一次完整的 `/v1/chat/completions` 请求为例，描述请求从接入到返回的全过程：

### 1. 接入与分词
1. FastAPI 收到 POST 请求（JSON），FrontendManager 分配唯一的 `uid`
2. 进程内 TokenizerWorker 调用 HF tokenizer 将文本转为 token IDs

### 2. 调度
3. FrontendManager 直接调用 `Scheduler.add_request(uid, input_ids, sampling_params)`
4. PrefillManager 将请求加入 pending 队列

### 3. Prefill（预填充）
5. PrefillManager 从 pending 队列取出请求，检查 KV Cache 可用空间、token budget
6. 分配 KV Cache pages（优先通过 RadixCache 复用已有前缀，命中部分走 extend attention）
7. 将 prompt tokens 打包成 `Batch`（phase="prefill"），由 BatchContext 填充：
   - `input_ids`: token 索引
   - `positions`: 位置编码
   - `attn_meta`: 类型化的 `AttentionMetadata`（KV 写入位置 `write_loc`、页表 `block_table` / `req_to_token`、varlen 边界 `cu_seqlens_q` 等）
   - `logits_indices`: 每个请求最后一个未缓存 token 的索引（lm_head 只算这些位置）
8. Engine 执行一次完整 forward：**所有 prompt tokens 并行处理**（lm_head 只算每个请求最后一个位置）
9. 生成第一个 output token，并立即检查终止条件（EOS / max_tokens / max_seq_len）
10. 请求移入 running 列表（状态：pending → running）

### 4. Decode（逐 token 生成）
11. DecodeManager 将所有 running 请求打包成 `Batch`（phase="decode"）
12. 每个 decode step，Engine 仅处理**一个**新 token（利用已缓存的 KV）
13. 通过 **CUDA Graph** 回放加速 decode（消除 kernel launch overhead）
14. Sampler 对 logits 做 top-p / top-k / temperature 采样得到下一个 token
15. 新 token 追加到请求的 input_ids，更新 page table

### 5. 拆词与返回
16. Scheduler 的 `step()` 返回 `list[OutputToken]`（dataclass，字段为 `uid` / `token_id` / `finished` / `finish_reason`）给 FrontendManager
17. FrontendManager 用增量 detokenize 将 token 解码为文本片段
18. FrontendManager 以 **SSE (Server-Sent Events)** 格式流式返回给客户端（`stream=true` 时；默认 `stream=false` 单次返回）
19. 步骤 11-18 循环直到：遇到 EOS token 或达到 `max_tokens` 或客户端断开（断开会触发 `Scheduler.abort_request`）

---

## 关键数据结构

### Req（请求）
```python
@dataclass(slots=True)
class Req:
    input_ids: list[int]       # 累积的 token 序列（prompt + 已生成）
    cached_len: int            # 已缓存 KV 的 token 数（前缀匹配后）
    output_len: int            # 已生成的 token 数
    uid: int                   # 全局唯一请求 ID
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle  # 指向分配的 KV Cache 页（含共享前缀页）
    status: SequenceStatus     # WAITING / RUNNING / FINISHED
```

### Batch（批次）
```python
@dataclass(slots=True)
class Batch:
    reqs: list[Req]            # 本次 forward 包含的请求
    phase: Literal["prefill", "decode"]
    # 由 BatchContext (prefill) / DecodeManager (decode) 填充的派生字段：
    input_ids: torch.Tensor        # (total_tokens,)
    positions: torch.Tensor        # (total_tokens,)
    attn_meta: AttentionMetadata   # 批次级 attention 输入（见下）
    logits_indices: torch.Tensor   # prefill：各请求最后一个未缓存 token 的索引
```

### AttentionMetadata（批次级 attention 输入）
```python
@dataclass(slots=True)
class AttentionMetadata:
    forward_mode: str          # "prefill" | "decode"
    write_loc: Tensor          # (total_tokens,) KV Cache 写入槽位（-1 跳过）
    cu_seqlens_q: Tensor       # prefill varlen 边界
    prefix_lens: Tensor        # 每请求已缓存前缀长度（extend attention）
    block_table: Tensor        # (num_reqs, max_blocks) 页表（FA 后端方言）
    req_to_token: Tensor       # (num_reqs, max_seq_len) 页表（PT 后端方言）
    cache_seqlens: Tensor      # decode：含当前 token 的总长度
    max_seqlen: int            # Python int，避免 backend 里 .item() 主机同步
```
模型 forward 签名统一为 `forward(input_ids, positions, attn_meta=None, logits_indices=None)`。
KV Cache 张量不随 forward 传递：每个 attention 层在 Engine 启动时通过
`set_kv_cache()` 绑定自己在 KV pool 中的切片，forward 时按 `write_loc` 写入。

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
- **FlashAttention** (`fa`): 通用后端，兼容性好
- **PyTorch SDPA** (`pt`): 纯 PyTorch 参考实现（CPU/NPU 也用它），支持 extend attention
- **FlashInfer** (`fi`): 尚未实现——选择后转调 `fa`；`fa,fi`（hybrid）目前同样全部走 `fa`

### KV Cache 管理
- **PagedAttention**: 将 KV Cache 划分为固定大小的 page（默认 16 tokens）
- **RadixCache**: 基于 Radix Tree 的前缀感知缓存管理器，自动检测和复用公共前缀；页所有权归树，请求结束只摘引用，页由 evict() 按需回收
- **NaiveCache**: 简化缓存，不做前缀共享、不保留页，用于对照
- Page Table（`req_to_token`）是 `(num_reqs, max_seq_len)` 的 int32 tensor，存储每个 token 位置对应的 KV Cache 槽位索引

### 分布式策略
- **Tensor Parallelism (TP)**: 将权重矩阵按列/行切分到多个 GPU（层逻辑已实现；多进程启动尚未实现，CLI 拒绝 `--tp-size > 1`）
  - `ColumnParallelLinear`: 沿 hidden_dim 切分（weight/bias 都分片；lm_head 显式 gather output）
  - `RowParallelLinear`: 沿 input_dim 切分（all-reduce output）
  - `VocabParallelEmbedding`: 沿 vocab_dim 切分
- **Collectives**: NCCL/HCCL 双后端通信原语（all-reduce, all-gather, broadcast，位于 `minisgl/engine/collectives.py`）；非分布式时为 no-op

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
- **Qwen3** — Qwen3ForCausalLM（支持 GQA + QK Norm）
- **Qwen3-MoE** — Qwen3MoEForCausalLM（含 MoE Router + 融合 expert 权重）

### 模型架构需支持的组件
| 组件 | 说明 |
|------|------|
| RMSNorm | 含 fused residual add 优化 |
| RoPE | 旋转位置编码，支持动态序列长度 |
| GQA (Grouped Query Attention) | Q heads 数可以不同于 KV heads 数 |
| Gated MLP (SwiGLU) | gate_proj + up_proj + down_proj 三矩阵 |
| MoE (Mixture of Experts) | 融合 expert 权重张量 + HF 对齐路由（softmax → top-k → 归一），纯 PyTorch 实现 |
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
- `stream=false`（默认）：单次 JSON 响应
- `stream=true`：SSE 流式返回 `data: {"choices": [{"delta": {"content": "..."}}]}`


### 启动方式
```bash
# 启动服务（TP=1）
python -m minisgl --model-path Qwen/Qwen3-0.6B --port 8000

# 交互式 CLI shell
python -m minisgl --model-path Qwen/Qwen3-0.6B --shell
```

> 注：`--tp-size > 1` 暂不支持（多进程 TP 启动未实现），传入会直接报错退出。

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
