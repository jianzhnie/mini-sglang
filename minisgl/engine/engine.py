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
from minisgl.utils.device import (
    get_device,
    get_device_type,
    init_distributed,
    is_accelerator_available,
    mem_get_info,
    synchronize,
)
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

    Holds the model, KV cache pool, execution graphs, and runs forward + sampling.
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
            logger.info("Loaded %d weights (model_type=%s)", loaded, model_type)
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

        self._graphs: dict[int, object] = {}
        self._graph_inputs: dict[int, tuple] = {}
        self._graph_outputs: dict[int, torch.Tensor] = {}
        if (
            server_args.cuda_graph_bs
            and server_args.cuda_graph_bs > 0
            and is_accelerator_available()
        ):
            self._capture_graphs()

        logger.info(
            "Engine initialized on rank %d (device=%s)", tp_rank, self._device_type
        )

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Release accelerator resources."""
        self._graphs.clear()
        self._graph_inputs.clear()
        self._graph_outputs.clear()
        if hasattr(self, "kv_cache_pool"):
            del self.kv_cache_pool
        synchronize()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.cleanup()

    def _create_model(self) -> torch.nn.Module:
        from minisgl.models.registry import create_model, detect_model_type

        model_type = detect_model_type(self.server_args.model_path)
        self._model_type = model_type
        return create_model(self.model_args, model_type)

    def _allocate_kv_cache(self) -> KVCachePool:
        """Allocate KV cache based on available accelerator memory."""
        args = self.server_args
        ma = self.model_args

        if is_accelerator_available():
            free_mem, total_mem = mem_get_info(self.device)
            used_mem = total_mem - free_mem
            available_mem = int(total_mem * args.memory_ratio - used_mem)
        else:
            available_mem = 512 * 1024 * 1024  # 512 MB for CPU

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

        logger.info("Allocating KV cache: %d pages", num_pages)
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

        # Decode: try execution graph first, fall back to eager
        bs = len(batch.reqs)
        graph = self._find_graph(bs)
        if graph is not None and batch.write_loc is not None:
            graph_bs, ins, outs = graph
            with torch.inference_mode():
                ins["input_ids"][:bs].copy_(batch.input_ids.squeeze(1))
                ins["positions"][:bs].copy_(batch.positions.squeeze(1))
                ins["write_loc"][:bs].copy_(batch.write_loc)
                self._graphs[graph_bs].replay()
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
        """Find an execution graph large enough for the given batch size."""
        for bs in sorted(self._graphs.keys()):
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
        """Batch sample by grouping requests with identical sampling params.

        Fast path: if all requests are greedy, skip grouping entirely.
        """
        if all(req.sampling_params.temperature <= 0.0 for req in reqs):
            return logits.argmax(dim=-1).tolist()

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
            params = reqs[indices[0]].sampling_params
            tokens = self.sampler.sample(batch_logits, params).tolist()
            for j, idx in enumerate(indices):
                token_ids[idx] = tokens[j]

        return token_ids

    def _capture_graphs(self) -> None:
        """Capture execution graphs for decode phase (CUDA or NPU)."""
        if not is_accelerator_available():
            logger.info("No accelerator available, skipping graph capture")
            return

        max_bs = min(self.server_args.cuda_graph_bs, self.server_args.max_running_req)
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        batch_sizes = [bs for bs in batch_sizes if bs <= max_bs]

        logger.info(
            "Capturing %s graphs for batch sizes: %s",
            self._device_type.upper(),
            batch_sizes,
        )

        for bs in batch_sizes:
            self._capture_graph_safe(bs)

    def _capture_graph_safe(self, batch_size: int) -> None:
        """Capture an execution graph, logging but not raising on failure."""
        try:
            self._capture_graph(batch_size)
        except RuntimeError as e:
            logger.warning("Failed to capture graph for bs=%d: %s", batch_size, e)

    def _capture_graph(self, batch_size: int) -> None:
        """Capture a single execution graph for a given batch size.

        Supports both CUDA graphs and NPU graphs (via torch.npu).
        """
        device = self.device

        input_ids = torch.ones(batch_size, dtype=torch.long, device=device)
        positions = torch.zeros(batch_size, dtype=torch.long, device=device)
        write_loc = torch.zeros(batch_size, dtype=torch.int32, device=device)

        # Warmup
        for _ in range(3):
            with torch.inference_mode():
                self.model(
                    input_ids=input_ids,
                    positions=positions,
                    k_cache=self.k_cache,
                    v_cache=self.v_cache,
                    write_loc=write_loc,
                )

        if self._device_type == "npu":
            import os

            os.environ.setdefault("TASK_QUEUE_ENABLE", "1")
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    k_cache=self.k_cache,
                    v_cache=self.v_cache,
                    write_loc=write_loc,
                )
        else:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    k_cache=self.k_cache,
                    v_cache=self.v_cache,
                    write_loc=write_loc,
                )

        self._graphs[batch_size] = graph
        self._graph_inputs[batch_size] = {
            "input_ids": input_ids,
            "positions": positions,
            "write_loc": write_loc,
        }
        self._graph_outputs[batch_size] = output
