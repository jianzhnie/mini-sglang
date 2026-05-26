"""Core inference engine: holds model, KV cache, and runs forward + sampling."""

from __future__ import annotations

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
        if (
            server_args.cuda_graph_bs
            and server_args.cuda_graph_bs > 0
            and torch.cuda.is_available()
        ):
            self._capture_cuda_graphs()

        logger.info(f"Engine initialized on rank {tp_rank}")

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
            )

    def sample(self, logits: torch.Tensor, batch: Batch) -> list:
        """Sample next tokens from logits."""
        if batch.phase == "decode" and logits.dim() == 3 and logits.shape[1] == 1:
            # Decode logits: (num_reqs, 1, vocab_size)
            logits = logits.squeeze(1)
            token_ids = []
            for i, req in enumerate(batch.reqs):
                next_token = self.sampler.sample(logits[i : i + 1], req.sampling_params)
                token_ids.append(next_token.item())
            return token_ids

        # Prefill or naive decode: logits are (total_tokens, vocab_size)
        token_ids = []
        offset = 0
        for req in batch.reqs:
            req_len = len(req.input_ids)
            seq_logits = logits[offset : offset + req_len]
            last_logits = seq_logits[-1:]  # Only sample from last position
            next_token = self.sampler.sample(last_logits, req.sampling_params)
            token_ids.append(next_token.item())
            offset += req_len
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
            self.model(
                input_ids=input_ids,
                positions=positions,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                write_loc=write_loc,
            )

        self.cuda_graphs[batch_size] = graph

    def replay_cuda_graph(self, batch_size: int) -> None:
        """Replay a captured CUDA graph for decode."""
        if batch_size in self.cuda_graphs:
            self.cuda_graphs[batch_size].replay()
        else:
            # Find closest larger graph
            for bs in sorted(self.cuda_graphs.keys()):
                if bs >= batch_size:
                    self.cuda_graphs[bs].replay()
                    return
