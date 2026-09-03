# 4,800 行 Python 手搓一个 SGLang：把 LLM 推理引擎扒光了给你看

> 项目地址：https://github.com/jianzhnie/mini-sglang

## 一、为什么读不懂 vLLM / SGLang 的源码？

想搞懂 LLM 推理引擎，几乎所有人的路径都是：打开 vLLM 或 SGLang 的仓库 → 被十几万行代码劝退 → 回去看博客。

问题不在于你，而在于生产级框架里塞了太多"工程负重"：分布式 RPC、多进程编排、CUDA kernel 模板元编程、各种硬件兜底……**核心机制可能只占 5%，但它被埋在 95% 的工程代码下面。**

所以我写了 **Mini-SGLang**：用 **~4,800 行纯 Python** 完整复刻 SGLang 的核心机制。不是玩具 demo，是一个真的能跑模型、能起 OpenAI 兼容服务、能上 NPU 的推理引擎——只是每一行都为你阅读而写。

## 二、它实现了什么？

现代 LLM 服务系统的关键环节，一个不少：

| 机制 | 说明 |
|------|------|
| **Continuous Batching** | Prefill / Decode 两阶段分离调度，最大化吞吐 |
| **PagedAttention** | 页式 KV Cache 管理，消除显存碎片 |
| **RadixCache** | Radix Tree 前缀感知缓存，公共前缀自动复用（含完整的页共享与 extend attention） |
| **CUDA / NPU Graph** | Decode 阶段 kernel launch 开销优化，NVIDIA 和华为昇腾都支持 |
| **Tensor Parallelism** | Column / Row Parallel Linear + NCCL/HCCL 通信原语 |
| **可插拔 Attention 后端** | FlashAttention / PyTorch SDPA 一键切换 |
| **OpenAI 兼容 API** | `/v1/chat/completions` + SSE 流式返回 |

模型支持：Qwen3 / Qwen3-MoE（RMSNorm + RoPE + SwiGLU + QK LayerNorm + GQA）。

## 三、代码读起来是什么体验？

整个项目的结构就是一张推理引擎的解剖图：

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

几个我特别满意的设计：

**1. 类型化的 AttentionMetadata，拒绝 kwargs 黑盒。** 很多教学项目喜欢 `**kwargs` 一路透传，读者根本搞不清数据从哪来、到哪去。Mini-SGLang 里每个 batch 的 attention 输入（页表、写入位置、序列边界……）都是一个显式的 dataclass，由 Scheduler 侧构建、backend 消费，一眼看穿。

**2. 模型继承体系消掉了 90% 的重复代码。** Qwen3 / Qwen3-MoE 共用一套基类：

```
RMSNormForCausalLM          ← 各模型继承
  └── RMSNormModel          ← 共享 embed → layers → norm
        └── RMSNormDecoderLayer  ← 共享 LN → Attn → LN → MLP
              └── GatedMLP       ← SwiGLU，所有模型共用
```

每个模型文件只需写自己独特的 Attention 类（MoE 再多写一个 Router/Expert 块）。

**3. 不是"能跑就行"，是真的对。** 99 个单元测试全绿，RadixCache 前缀共享与全量 prefill 的 logits 级数值一致性、double-free 防护、CUDA Graph 的静态 buffer 正确性……这些生产级引擎里最容易出错的点，都有回归测试盯着。

## 四、三分钟跑起来

```bash
pip install -e .

# 起 OpenAI 兼容服务
python -m minisgl --model-path Qwen/Qwen3-0.6B --port 8000

# 或者不起服务，离线批量推理
python examples/offline_inference.py --model-path /path/to/model
```

连模型都不想下载？有个零依赖的 CPU demo，随机权重小模型验证完整管线：

```bash
python examples/cpu_demo.py
```

## 五、不只是 CUDA：华为昇腾 NPU 也跑通了

通过 `torch_npu` 做了完整的设备无关适配，Ascend 910 上 Qwen3 系列全部实测通过：

| 模型 | Prefill | Decode | 批量(3x)吞吐 |
|------|---------|--------|-------------|
| Qwen3-0.6B | 57.4 tok/s | 14.3 tok/s | 54.0 tok/s |
| Qwen3-1.7B | 201.4 tok/s | 13.5 tok/s | 36.0 tok/s |
| Qwen3-4B | 260.9 tok/s | 13.0 tok/s | 39.2 tok/s |

（测试环境：Ascend 910 64GB，torch 2.12.0 + torch_npu 2.12.0rc1，eager 模式）

## 六、谁适合读这个项目？

- **想转行/深入 LLM Infra 的工程师**：从调度器到 CUDA Graph，一条请求的生命周期全部摊开
- **准备面试的同学**：PagedAttention、RadixCache、Continuous Batching 不再是八股文名词，是你读过的代码
- **想做二次开发的研究者**：改采样策略、加新模型、换 attention 后端，都有清晰的扩展点
- **NPU 开发者**：torch_npu 适配的完整参考实现

## 结语

理解一个系统的最好方式，是把它重新发明一遍——或者，读一个被认真重新发明过的版本。

如果这个项目对你有帮助，欢迎 Star / Fork / PR：

**https://github.com/jianzhnie/mini-sglang**

---

*参考项目：[SGLang](https://github.com/sgl-project/sglang)（架构蓝本）、[vLLM](https://github.com/vllm-project/vllm)（PagedAttention 参考）、[FlashAttention](https://github.com/Dao-AILab/flash-attention) / [FlashInfer](https://github.com/flashinfer-ai/flashinfer)（Attention kernel）*
