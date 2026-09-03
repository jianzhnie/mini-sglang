"""Core inference engine: holds model, KV cache, and runs forward + sampling."""

from __future__ import annotations

import contextlib
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minisgl.config import ModelArgs, ServerArgs
    from minisgl.models.attention.metadata import AttentionMetadata
    from minisgl.scheduler.batch import Batch, Req

__all__ = ["Engine"]
import torch

from minisgl.engine.batch_context import BatchContext
from minisgl.engine.graph import GraphRunner
from minisgl.engine.kvcache.pool import KVCachePool
from minisgl.models.attention.dispatcher import AttentionBackend
from minisgl.models.attention.layer import BaseAttention
from minisgl.models.layers.rope import RotaryEmbedding
from minisgl.sampling import Sampler
from minisgl.utils.device import (
    get_device,
    get_device_type,
    init_distributed,
    mem_get_info,
    synchronize,
)
from minisgl.utils.logger import logger
from minisgl.utils.weights import load_hf_weights, load_weights_parallel

# Teaching simplification: on CPU there is no free-memory query, so the KV
# pool is sized against this fixed budget.
_CPU_KV_CACHE_BYTES = 512 * 1024 * 1024  # 512 MB


def _path_exists(path: str) -> bool:
    from pathlib import Path

    return Path(path).is_dir() and (Path(path) / "config.json").exists()


def _resolve_dtype(dtype_str: str, model_path: str, device_type: str) -> torch.dtype:
    """Resolve the target model dtype from the --dtype CLI value.

    'auto' reads torch_dtype from the model's config.json (float32 fallback).
    CPU only supports float32 reliably: anything else is forced to float32.
    """
    if dtype_str == "auto":
        import json
        from pathlib import Path

        config_file = Path(model_path) / "config.json"
        dtype_str = "float32"
        if config_file.exists():
            with config_file.open() as f:
                dtype_str = json.load(f).get("torch_dtype", "float32")

    dtype = getattr(torch, dtype_str, None)
    if not isinstance(dtype, torch.dtype):
        logger.warning("Unknown dtype %r; falling back to float32", dtype_str)
        dtype = torch.float32

    if device_type == "cpu" and dtype != torch.float32:
        logger.warning(
            "dtype %s is not fully supported on CPU; falling back to float32", dtype
        )
        dtype = torch.float32
    return dtype


class Engine:
    """Core inference engine.

    Holds the model, KV cache pool, and runs forward + sampling. Decode
    optionally runs on captured execution graphs (see GraphRunner).
    Supports CUDA, NPU (Ascend), and CPU devices.
    Supports context manager protocol for resource cleanup.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        model_args: ModelArgs,
        tp_rank: int = 0,
    ) -> None:
        self.server_args = server_args
        self.model_args = model_args
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size
        self.device = get_device()
        self._device_type = get_device_type()

        if self.tp_size > 1:
            init_distributed(tp_rank=tp_rank, tp_size=self.tp_size)
            if model_args.num_kv_heads % self.tp_size != 0:
                # Teaching simplification: KV heads are not truly divisible,
                # so ranks replicate heads (max(1, ...) in each attention
                # module and the KV pool) instead of erroring out.
                logger.warning(
                    "num_kv_heads=%d not divisible by tp_size=%d; KV heads will "
                    "be replicated across ranks (no exact sharding)",
                    model_args.num_kv_heads,
                    self.tp_size,
                )

        AttentionBackend.configure(server_args.attention_backend)

        # Clamp max_seq_len to the model's trained context window. RoPE tables
        # and the paged KV pool / req_to_token buffers are all sized off this
        # value, and learned-position models (OPT embed_positions) hard-fail
        # past max_position_embeddings — so an oversized --max-seq-len would
        # only surface later as an index error mid-generation. Engine is always
        # built before the Scheduler (which shares server_args), so clamping
        # here propagates everywhere.
        max_pos = model_args.max_position_embeddings
        if max_pos and server_args.max_seq_len > max_pos:
            logger.warning(
                "max_seq_len=%d exceeds the model's max_position_embeddings=%d; "
                "clamping to %d",
                server_args.max_seq_len,
                max_pos,
                max_pos,
            )
            server_args.max_seq_len = max_pos

        self.model, self._model_type = self._create_model()
        self.dtype = _resolve_dtype(
            server_args.dtype, server_args.model_path, self.device.type
        )
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        # No autograd during capture/replay (or any inference forward).
        self.model.requires_grad_(False)

        self._prebuild_rope()
        self._load_weights()

        self.kv_cache_pool = self._allocate_kv_cache()
        self._assign_kv_cache()

        self.batch_context = BatchContext(
            server_args.max_running_req,
            server_args.max_seq_len,
            server_args.page_size,
            self.device,
        )

        self.sampler = Sampler()

        self.graph_runner: GraphRunner | None = None
        if (
            server_args.cuda_graph_bs
            and server_args.cuda_graph_bs > 0
            and self.device.type in ("cuda", "npu")
        ):
            self.graph_runner = GraphRunner(self)

        logger.info(
            "Engine initialized on rank %d (device=%s)", tp_rank, self._device_type
        )

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *args: object) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Release accelerator resources."""
        if self.graph_runner is not None:
            self.graph_runner.clear()
        # The per-layer k_cache/v_cache references are views into the pool
        # buffer; drop them too or the pool's memory stays referenced and
        # cannot be released. The graph pad handle is released together with
        # the pool.
        if hasattr(self, "model"):
            for module in self.model.modules():
                if isinstance(module, BaseAttention):
                    module.k_cache = None
                    module.v_cache = None
        if hasattr(self, "kv_cache_pool"):
            del self.kv_cache_pool
        synchronize()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.cleanup()

    def _create_model(self) -> tuple[torch.nn.Module, str]:
        """Create the model and return it together with its detected type."""
        from minisgl.models.registry import create_model, detect_model_type

        model_type = detect_model_type(self.server_args.model_path)
        return create_model(self.model_args, model_type), model_type

    def _prebuild_rope(self) -> None:
        # Pre-build RoPE cos/sin tables on the compute device so per-layer
        # forwards avoid host-device syncs and stay graph-capturable.
        # RotaryEmbedding is not an nn.Module, so it never shows up in
        # modules(); reach it via the attention modules' `rotary_emb` attr.
        # OPT has no RoPE — the loop then simply finds nothing. prebuild()
        # is idempotent, so per-layer instances are fine.
        for module in self.model.modules():
            rotary = getattr(module, "rotary_emb", None)
            if isinstance(rotary, RotaryEmbedding):
                rotary.prebuild(self.server_args.max_seq_len, self.device)

    def _load_weights(self) -> None:
        """Load HF weights from the model path, if it exists on disk."""
        from minisgl.models.registry import get_remap_fn

        model_path = self.server_args.model_path
        if not (model_path and _path_exists(model_path)):
            return
        state_dict = load_hf_weights(model_path)
        model_type = self._model_type
        remap_fn = get_remap_fn(model_type)
        loaded = load_weights_parallel(
            self.model,
            state_dict,
            self.tp_rank,
            self.tp_size,
            remap_fn=remap_fn,
        )
        logger.info("Loaded %d weights (model_type=%s)", loaded, model_type)
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights(state_dict)
        # Fused-expert MoE models (e.g. Qwen3MoE) aggregate HF's
        # per-expert weights (mlp.experts.{i}.*) into fused tensors
        # themselves; load_weights_parallel cannot match those keys.
        load_hf_experts = getattr(self.model, "load_hf_experts", None)
        if load_hf_experts is not None:
            n_expert = load_hf_experts(state_dict)
            logger.info("Loaded %d fused expert weights", n_expert)

    def _allocate_kv_cache(self) -> KVCachePool:
        """Allocate KV cache based on available accelerator memory."""
        args = self.server_args
        ma = self.model_args

        if self.device.type in ("cuda", "npu"):
            free_mem, total_mem = mem_get_info(self.device)
            used_mem = total_mem - free_mem
            available_mem = int(total_mem * args.memory_ratio - used_mem)
            if available_mem <= 0:
                # The model already fills the budget (memory_ratio over-used).
                # Silently allocating a 1-page pool turns every later request
                # into "needs more pages than the pool" aborts — fail loudly
                # instead so the operator can raise memory_ratio or use a
                # smaller model.
                logger.error(
                    "No memory left for KV cache: total=%d used=%d (ratio=%.2f). "
                    "Lower the model size or free GPU memory.",
                    total_mem,
                    used_mem,
                    args.memory_ratio,
                )
                raise RuntimeError("KV cache allocation failed: no free memory")
        else:
            available_mem = _CPU_KV_CACHE_BYTES

        dtype_itemsize = self.model.lm_head.weight.dtype.itemsize
        # Match the attention modules: replicate KV heads when num_kv_heads
        # is not divisible by tp_size (never allocate a 0-head buffer).
        num_kv_heads_per_rank = max(1, ma.num_kv_heads // self.tp_size)
        bytes_per_page = (
            2
            * ma.num_layers
            * args.page_size
            * num_kv_heads_per_rank
            * ma.head_dim
            * dtype_itemsize
        )

        num_pages = max(1, available_mem // bytes_per_page)
        max_pages_needed = args.max_running_req * max(
            1,
            args.max_seq_len // args.page_size + 1,
        )
        num_pages = min(num_pages, max_pages_needed)

        logger.info("Allocating KV cache: %d pages", num_pages)
        return KVCachePool(
            num_layers=ma.num_layers,
            num_pages=num_pages,
            page_size=args.page_size,
            num_kv_heads=num_kv_heads_per_rank,
            head_dim=ma.head_dim,
            dtype=self.model.lm_head.weight.dtype,
            device=self.device,
        )

    def _assign_kv_cache(self) -> None:
        """Bind each attention layer to its slice of the paged KV cache pool.

        The pool buffer is (num_layers, num_pages, page_size, num_kv_heads,
        head_dim); modules() yields the layers' attention modules in layer
        order, so layer i gets slice i.
        """
        k_all, v_all = self.kv_cache_pool.get_all_kv_cache()
        layer_id = 0
        for module in self.model.modules():
            if isinstance(module, BaseAttention):
                module.set_kv_cache(k_all[layer_id], v_all[layer_id])
                layer_id += 1

    def _run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_meta: AttentionMetadata | None = None,
        logits_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single entry point for model forward.

        The KV cache no longer travels through the call — each attention
        layer holds its own slice (see _assign_kv_cache); only the per-batch
        AttentionMetadata is passed in.
        """
        return self.model(
            input_ids=input_ids,
            positions=positions,
            attn_meta=attn_meta,
            logits_indices=logits_indices,
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        """Run model forward pass on a batch."""
        if batch.phase == "prefill":
            self.batch_context.prepare(batch)
            with torch.inference_mode():
                return self._run_model(
                    batch.input_ids,
                    batch.positions,
                    batch.attn_meta,
                    batch.logits_indices,
                )

        # Decode: try execution graph first, fall back to eager
        if self.graph_runner is not None:
            logits = self.graph_runner.replay(batch)
            if logits is not None:
                return logits
        with torch.inference_mode():
            return self._run_model(
                batch.input_ids,
                batch.positions,
                batch.attn_meta,
            )

    def sample(self, logits: torch.Tensor, batch: Batch) -> list[int]:
        """Sample next tokens from logits.

        Groups requests with identical sampling params for batched sampling.
        """
        # Normalize to (num_reqs, vocab_size): decode may produce
        # (num_reqs, 1, vocab_size) or, for a single request, (vocab_size,).
        # Prefill already yields (num_reqs, vocab_size) — forward gathered each
        # request's last-position logits via logits_indices, so the
        # normalization below is a no-op for it.
        if logits.dim() == 3 and logits.shape[1] == 1:
            logits = logits.squeeze(1)  # (num_reqs, vocab_size)
        elif logits.dim() == 1:
            logits = logits.unsqueeze(0)  # (1, vocab_size)
        return self._sample_batched(logits, batch.reqs)

    def _sample_batched(self, logits: torch.Tensor, reqs: list[Req]) -> list[int]:
        """Batch sample by grouping requests with identical sampling params.

        Fast path: if all requests are greedy, skip grouping entirely.
        """
        if all(req.sampling_params.temperature <= 0.0 for req in reqs):
            return logits.argmax(dim=-1).tolist()

        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, req in enumerate(reqs):
            key = (
                req.sampling_params.temperature,
                req.sampling_params.top_k,
                req.sampling_params.top_p,
            )
            groups[key].append(i)

        token_ids = [0] * len(reqs)
        for indices in groups.values():
            batch_logits = logits[indices]
            params = reqs[indices[0]].sampling_params
            tokens = self.sampler.sample(batch_logits, params).tolist()
            for j, idx in enumerate(indices):
                token_ids[idx] = tokens[j]

        return token_ids
