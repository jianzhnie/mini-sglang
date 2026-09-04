"""Core inference engine: holds model, KV cache, and runs forward + sampling."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minisgl.config import ModelArgs, ServerArgs
    from minisgl.models.attention.metadata import AttentionMetadata
    from minisgl.scheduler.batch import Batch

__all__ = ["Engine"]
import torch

from minisgl.engine import model_runner
from minisgl.models.attention.dispatcher import AttentionBackend
from minisgl.models.attention.layer import BaseAttention
from minisgl.sampling import Sampler
from minisgl.utils.device import (
    get_device,
    get_device_type,
    init_distributed,
    synchronize,
)
from minisgl.utils.logger import logger


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

        self._maybe_init_distributed(model_args)
        AttentionBackend.configure(server_args.attention_backend)
        self._clamp_max_seq_len(model_args)
        self._validate_model_path(server_args.model_path)

        # --- Model assembly (delegated to model_runner for testability) ---
        model, model_type = model_runner.detect_and_create_model(
            server_args.model_path, model_args
        )
        self._model_type = model_type
        self.dtype = model_runner.resolve_dtype(
            server_args.dtype, server_args.model_path, self.device.type
        )
        model.to(device=self.device, dtype=self.dtype)
        model.eval()
        # No autograd during capture/replay (or any inference forward).
        model.requires_grad_(False)

        model_runner.prebuild_rope(model, server_args.max_seq_len, self.device)

        loaded = model_runner.load_model_weights(
            model, server_args.model_path, self.tp_rank, self.tp_size
        )
        logger.info("Loaded %d weights (model_type=%s)", loaded, self._model_type)

        # --- KV cache allocation (delegated to KVCacheAllocator) ---
        from minisgl.engine.kvcache.allocator import KVCacheAllocator

        allocator = KVCacheAllocator(server_args, model_args)
        self.kv_cache_pool = allocator.allocate(model, self.device, self.tp_size)

        # --- Execution + sampling (delegated to ModelRunner / Sampler) ---
        self.runner = model_runner.ModelRunner(model, server_args, self.device)
        self.model = model  # public alias kept for tests / GraphRunner coupling
        self.batch_context = self.runner.batch_context
        self.graph_runner = self.runner.graph_runner

        self.sampler = Sampler()

        logger.info(
            "Engine initialized on rank %d (device=%s)", tp_rank, self._device_type
        )

    def _maybe_init_distributed(self, model_args: ModelArgs) -> None:
        """Initialize the process group when tensor parallelism is requested."""
        if self.tp_size <= 1:
            return
        init_distributed(tp_rank=self.tp_rank, tp_size=self.tp_size)
        if model_args.num_kv_heads % self.tp_size != 0:
            # Teaching simplification: KV heads are not truly divisible, so
            # ranks replicate heads (max(1, ...) in each attention module and
            # the KV pool) instead of erroring out.
            logger.warning(
                "num_kv_heads=%d not divisible by tp_size=%d; KV heads will "
                "be replicated across ranks (no exact sharding)",
                model_args.num_kv_heads,
                self.tp_size,
            )

    def _clamp_max_seq_len(self, model_args: ModelArgs) -> None:
        """Clamp max_seq_len to the model's trained context window.

        RoPE tables and the paged KV pool / req_to_token buffers are all sized
        off this value, so an oversized --max-seq-len would only surface later
        as an index error mid-generation. Engine is always built before the
        Scheduler (which shares server_args), so clamping here propagates
        everywhere.
        """
        args = self.server_args
        max_pos = model_args.max_position_embeddings
        if max_pos and args.max_seq_len > max_pos:
            logger.warning(
                "max_seq_len=%d exceeds the model's max_position_embeddings=%d; "
                "clamping to %d",
                args.max_seq_len,
                max_pos,
                max_pos,
            )
            args.max_seq_len = max_pos

    @staticmethod
    def _validate_model_path(model_path: str) -> None:
        """Fail fast with a clear message when the model dir is unusable.

        Catches the common typo / un-downloaded case up front instead of an
        opaque FileNotFoundError deep inside registry/config loading.
        """
        from pathlib import Path

        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Model path does not exist: {model_path!r}. "
                "Pass a directory containing config.json."
            )
        if not p.is_dir():
            raise NotADirectoryError(
                f"Model path is not a directory: {model_path!r}."
            )
        if not (p / "config.json").is_file():
            raise FileNotFoundError(
                f"Model path has no config.json: {model_path!r}. "
                "This does not look like a HuggingFace model directory."
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

    def _run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_meta: AttentionMetadata | None = None,
        logits_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single forward entry (alias to ModelRunner, kept for GraphRunner)."""
        return self.runner._run_model(
            input_ids, positions, attn_meta, logits_indices
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        """Run a model forward pass on a scheduler batch (via ModelRunner)."""
        return self.runner.forward(batch)

    def sample(self, logits: torch.Tensor, batch: Batch) -> list[int]:
        """Sample next tokens from logits.

        Groups requests with identical sampling params for batched sampling
        (see Sampler.sample_batch).
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

        params = [req.sampling_params for req in batch.reqs]
        return self.sampler.sample_batch(logits, params)
