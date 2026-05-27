"""Core inference engine: holds model, KV cache, and runs forward + sampling."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minisgl.config import ModelArgs, ServerArgs
    from minisgl.scheduler.batch import Batch

__all__ = ["Engine"]
import torch

from minisgl.engine.context import BatchContext
from minisgl.engine.kvcache.pool import KVCachePool
from minisgl.models.attention.backend import AttentionBackend
from minisgl.sampling.sampler import Sampler
from minisgl.utils.device import get_device, init_distributed
from minisgl.utils.logger import logger
from minisgl.utils.weights import load_hf_weights, load_weights_parallel


def _path_exists(path: str) -> bool:
    from pathlib import Path

    return Path(path).is_dir() and (Path(path) / "config.json").exists()


def _get_remap_fn(model_type: str):
    """Return a key remapping function for the given model type."""
    if model_type == "opt":

        def _remap(name: str) -> str:
            return name.replace("model.", "model.decoder.")

        return _remap
    return None


class Engine:
    """Core inference engine.

    Holds the model, KV cache pool, CUDA graphs, and runs forward + sampling.
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

        if self.tp_size > 1:
            init_distributed(tp_rank=tp_rank, tp_size=self.tp_size, backend="nccl")

        AttentionBackend.configure(server_args.attention_backend)

        self.model = self._create_model()
        self.model.to(self.device)
        self.model.eval()

        # Load weights only if model path exists
        if server_args.model_path and _path_exists(server_args.model_path):
            state_dict = load_hf_weights(server_args.model_path)
            model_type = self._model_type
            remap_fn = _get_remap_fn(model_type)
            loaded = load_weights_parallel(
                self.model,
                state_dict,
                tp_rank,
                self.tp_size,
                remap_fn=remap_fn,
            )
            logger.info(f"Loaded {loaded} weights (model_type={model_type})")
            if hasattr(self.model, "tie_weights"):
                self.model.tie_weights(state_dict)

        self.kv_cache_pool = self._allocate_kv_cache()
        self._assign_kv_cache()

        self.batch_context = BatchContext(
            server_args.max_running_req,
            server_args.max_seq_len,
            server_args.page_size,
            self.device,
        )

        self.sampler = Sampler(model_args.vocab_size)

        self.cuda_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._graph_inputs: dict[int, tuple] = {}
        self._graph_outputs: dict[int, torch.Tensor] = {}
        if (
            server_args.cuda_graph_bs
            and server_args.cuda_graph_bs > 0
            and torch.cuda.is_available()
        ):
            self._capture_cuda_graphs()

        logger.info(f"Engine initialized on rank {tp_rank}")

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Release GPU resources."""
        self.cuda_graphs.clear()
        self._graph_inputs.clear()
        self._graph_outputs.clear()
        if hasattr(self, "kv_cache_pool"):
            del self.kv_cache_pool
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.cleanup()

    def _create_model(self) -> torch.nn.Module:
        from minisgl.models.registry import create_model, detect_model_type

        model_type = detect_model_type(self.server_args.model_path)
        self._model_type = model_type
        return create_model(self.model_args, model_type)

    def _allocate_kv_cache(self) -> KVCachePool:
        """Allocate KV cache based on available GPU memory."""
        args = self.server_args
        ma = self.model_args

        if torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.mem_get_info(self.device)
            used_mem = total_mem - free_mem
            available_mem = int(total_mem * args.memory_ratio - used_mem)
        else:
            # CPU: use a reasonable default
            available_mem = 512 * 1024 * 1024  # 512 MB

        dtype_itemsize = self.model.lm_head.weight.dtype.itemsize
        bytes_per_page = (
            2
            * ma.num_layers
            * args.page_size
            * ma.num_kv_heads
            * ma.head_dim
            * dtype_itemsize
            // self.tp_size
        )

        num_pages = max(1, available_mem // bytes_per_page)
        max_pages_needed = args.max_running_req * max(
            1,
            args.max_seq_len // args.page_size + 1,
        )
        num_pages = min(num_pages, max_pages_needed)

        logger.info(f"Allocating KV cache: {num_pages} pages")
        return KVCachePool(
            num_layers=ma.num_layers,
            num_pages=num_pages,
            page_size=args.page_size,
            num_kv_heads=ma.num_kv_heads // self.tp_size,
            head_dim=ma.head_dim,
            dtype=self.model.lm_head.weight.dtype,
            device=self.device,
        )

    def _assign_kv_cache(self) -> None:
        """Assign KV cache slices to each model layer."""
        # Store as (num_layers, num_pages, page_size, num_kv_heads, head_dim)
        k_all, v_all = self.kv_cache_pool.get_all_kv_cache()
        self.k_cache = k_all
        self.v_cache = v_all

    def forward(self, batch: Batch) -> torch.Tensor:
        """Run model forward pass on a batch."""
        if batch.phase == "prefill":
            self.batch_context.prepare(batch)
            with torch.inference_mode():
                return self.model(
                    input_ids=batch.input_ids,
                    positions=batch.positions,
                    k_cache=self.k_cache,
                    v_cache=self.v_cache,
                    write_loc=batch.write_loc,
                    cu_seqlens_q=batch.cu_seqlens_q,
                )

        # Decode: try CUDA graph first, fall back to eager
        bs = len(batch.reqs)
        graph = self._find_graph(bs)
        if graph is not None and batch.write_loc is not None:
            graph_bs, ins, outs = graph
            with torch.inference_mode():
                ins["input_ids"][:bs].copy_(batch.input_ids)
                ins["positions"][:bs].copy_(batch.positions)
                ins["write_loc"][:bs].copy_(batch.write_loc)
                self.cuda_graphs[graph_bs].replay()
                return outs[:bs]
        else:
            with torch.inference_mode():
                return self.model(
                    input_ids=batch.input_ids,
                    positions=batch.positions,
                    k_cache=self.k_cache,
                    v_cache=self.v_cache,
                    write_loc=batch.write_loc,
                    cache_seqlens=batch.cache_seqlens,
                    block_table=batch.block_table,
                    req_to_token=batch.req_to_token,
                )

    def _find_graph(self, batch_size: int) -> tuple | None:
        """Find a CUDA graph large enough for the given batch size."""
        for bs in sorted(self.cuda_graphs.keys()):
            if bs >= batch_size:
                return (bs, self._graph_inputs[bs], self._graph_outputs[bs])
        return None

    def sample(self, logits: torch.Tensor, batch: Batch) -> list:
        """Sample next tokens from logits.

        Groups requests with identical sampling params for batched sampling.
        """
        if batch.phase == "decode":
            if logits.dim() == 3 and logits.shape[1] == 1:
                logits = logits.squeeze(1)  # (num_reqs, vocab_size)
            elif logits.dim() == 1:
                logits = logits.unsqueeze(0)  # (1, vocab_size)
            return self._sample_batched(logits, batch.reqs)

        # Prefill: collect last-position logits per request
        last_logits = []
        offset = 0
        for req in batch.reqs:
            req_len = req.uncached_len
            last_logits.append(logits[offset + req_len - 1 : offset + req_len])
            offset += req_len

        if last_logits:
            batched = torch.cat(last_logits, dim=0)  # (num_reqs, vocab_size)
            return self._sample_batched(batched, batch.reqs)
        return []

    def _sample_batched(self, logits: torch.Tensor, reqs: list) -> list:
        """Batch sample by grouping requests with identical sampling params."""
        from collections import defaultdict

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
            # All requests in this group share the same sampling params
            params = reqs[indices[0]].sampling_params
            tokens = self.sampler.sample(batch_logits, params)
            for j, idx in enumerate(indices):
                token_ids[idx] = tokens[j].item()

        return token_ids

    def _capture_cuda_graphs(self) -> None:
        """Capture CUDA graphs for decode phase."""
        if not torch.cuda.is_available():
            logger.info("CUDA not available, skipping graph capture")
            return

        max_bs = min(self.server_args.cuda_graph_bs, self.server_args.max_running_req)
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        batch_sizes = [bs for bs in batch_sizes if bs <= max_bs]

        logger.info(f"Capturing CUDA graphs for batch sizes: {batch_sizes}")

        for bs in batch_sizes:
            self._capture_graph_safe(bs)

    def _capture_graph_safe(self, batch_size: int) -> None:
        """Capture a CUDA graph, logging but not raising on failure."""
        try:
            self._capture_graph(batch_size)
        except RuntimeError as e:
            logger.warning(f"Failed to capture CUDA graph for bs={batch_size}: {e}")

    def _capture_graph(self, batch_size: int) -> None:
        """Capture a single CUDA graph for a given batch size."""
        device = self.device

        input_ids = torch.ones(batch_size, dtype=torch.long, device=device)
        positions = torch.zeros(batch_size, dtype=torch.long, device=device)
        write_loc = torch.zeros(batch_size, dtype=torch.int32, device=device)

        # Warmup
        for _ in range(3):
            self.model(
                input_ids=input_ids,
                positions=positions,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                write_loc=write_loc,
            )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.model(
                input_ids=input_ids,
                positions=positions,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                write_loc=write_loc,
            )

        self.cuda_graphs[batch_size] = graph
        self._graph_inputs[batch_size] = {
            "input_ids": input_ids,
            "positions": positions,
            "write_loc": write_loc,
        }
        self._graph_outputs[batch_size] = output
