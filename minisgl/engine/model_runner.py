"""Model construction, weight loading, and execution for the inference engine.

Consolidates everything about the model except the Engine's per-request
sampling glue:

- **assembly**: ``detect_and_create_model`` / ``resolve_dtype`` /
  ``prebuild_rope`` — build a model from a HF directory and prepare it;
- **weights**: ``load_model_weights`` — load HF weights (incl. MoE fused
  experts and tied embeddings);
- **execution**: ``ModelRunner`` — owns the model and the CUDA/NPU decode
  graphs, and runs forward passes over scheduler batches.

The Engine delegates to this module and keeps thin public aliases (``model``,
``graph_runner``, ``forward``) for compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import torch.nn as nn

    from minisgl.config import ModelArgs, ServerArgs
    from minisgl.models.attention.metadata import AttentionMetadata
    from minisgl.scheduler.batch import Batch

__all__ = [
    "ModelRunner",
    "detect_and_create_model",
    "load_model_weights",
    "prebuild_rope",
    "resolve_dtype",
]


def detect_and_create_model(
    model_path: str, model_args: "ModelArgs"
) -> tuple["nn.Module", str]:
    """Create the model from the model dir's config and return (model, type).

    ``model_path`` is used only to auto-detect the architecture
    (``detect_model_type`` reads its ``config.json``); instantiation then uses
    the already-parsed ``model_args``.
    """
    from minisgl.models.registry import create_model, detect_model_type

    model_type = detect_model_type(model_path)
    return create_model(model_args, model_type), model_type


def resolve_dtype(dtype_str: str, model_path: str, device_type: str) -> torch.dtype:
    """Resolve the target model dtype from the --dtype CLI value.

    'auto' reads torch_dtype from the model's config.json (float32 fallback).
    CPU only supports float32 reliably: anything else is forced to float32.
    """
    import json
    from pathlib import Path

    if dtype_str == "auto":
        config_file = Path(model_path) / "config.json"
        dtype_str = "float32"
        if config_file.exists():
            with config_file.open() as f:
                dtype_str = json.load(f).get("torch_dtype", "float32")

    dtype = getattr(torch, dtype_str, None)
    if not isinstance(dtype, torch.dtype):
        from minisgl.utils.logger import logger

        logger.warning("Unknown dtype %r; falling back to float32", dtype_str)
        dtype = torch.float32

    if device_type == "cpu" and dtype != torch.float32:
        from minisgl.utils.logger import logger

        logger.warning(
            "dtype %s is not fully supported on CPU; falling back to float32", dtype
        )
        dtype = torch.float32
    return dtype


def prebuild_rope(model, max_seq_len: int, device) -> None:
    """Pre-build each layer's RoPE cos/sin tables on the compute device.

    Pre-building keeps per-layer forwards free of host-device syncs and CUDA
    graph-capturable. RotaryEmbedding is not an nn.Module, so it never shows
    up in ``modules()``; reach it via the attention modules' ``rotary_emb``
    attribute. ``prebuild()`` is idempotent, so per-layer instances are fine.
    """
    from minisgl.models.layers.rope import RotaryEmbedding

    for module in model.modules():
        rotary = getattr(module, "rotary_emb", None)
        if isinstance(rotary, RotaryEmbedding):
            rotary.prebuild(max_seq_len, device)


def load_model_weights(
    model,
    model_path: str,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> int:
    """Load HF weights into ``model`` from ``model_path``, if it exists.

    Handles the Qwen3 family's two weight layouts:
    - plain parameters, loaded verbatim via ``load_weights_parallel``;
    - fused-expert MoE tensors (Qwen3MoE), whose per-expert HF keys are
      aggregated by the model's own ``load_hf_experts`` hook.
    Also applies ``tie_weights`` when the model declares it.

    Returns the number of plain parameters loaded.
    """
    from pathlib import Path

    from minisgl.utils.logger import logger
    from minisgl.utils.weights import load_hf_weights, load_weights_parallel

    def _path_exists(path: str) -> bool:
        return Path(path).is_dir() and (Path(path) / "config.json").exists()

    if not (model_path and _path_exists(model_path)):
        return 0

    state_dict = load_hf_weights(model_path)
    # The Qwen3 family loads HF keys verbatim — no remapping needed.
    loaded = load_weights_parallel(
        model,
        state_dict,
        tp_rank,
        tp_size,
    )
    if hasattr(model, "tie_weights"):
        model.tie_weights(state_dict)
    load_hf_experts = getattr(model, "load_hf_experts", None)
    if load_hf_experts is not None:
        n_expert = load_hf_experts(state_dict)
        logger.info("Loaded %d fused expert weights", n_expert)
    return loaded


class ModelRunner:
    """Owns the model and executes forward passes over scheduler batches.

    Attributes mirror what callers expect on the Engine: ``model``,
    ``device``, ``server_args``, ``graph_runner``, ``batch_context``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        server_args: "ServerArgs",
        device: torch.device,
    ) -> None:
        from minisgl.engine.batch_context import BatchContext

        self.model = model
        self.device = device
        self.server_args = server_args

        self.batch_context = BatchContext(
            server_args.max_running_req,
            server_args.max_seq_len,
            server_args.page_size,
            device,
        )

        # Decode acceleration: capture CUDA/NPU graphs for the decode path.
        from minisgl.engine.graph import GraphRunner

        self.graph_runner: GraphRunner | None = None
        if (
            server_args.cuda_graph_bs
            and server_args.cuda_graph_bs > 0
            and device.type in ("cuda", "npu")
        ):
            self.graph_runner = GraphRunner(self)

    def _run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_meta: "AttentionMetadata | None" = None,
        logits_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single entry point for the model forward.

        The KV cache is bound to each attention layer (see the allocator's
        ``bind_layers``), so only the per-batch AttentionMetadata travels
        through the call.
        """
        return self.model(
            input_ids=input_ids,
            positions=positions,
            attn_meta=attn_meta,
            logits_indices=logits_indices,
        )

    def forward(self, batch: "Batch") -> torch.Tensor:
        """Run a model forward pass on a scheduler batch."""
        if batch.phase == "prefill":
            self.batch_context.prepare(batch)
            with torch.inference_mode():
                return self._run_model(
                    batch.input_ids,
                    batch.positions,
                    batch.attn_meta,
                    batch.logits_indices,
                )

        # Decode: try the captured execution graph first, else eager.
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

    def clear_graphs(self) -> None:
        """Drop captured graphs and their static buffers."""
        if self.graph_runner is not None:
            self.graph_runner.clear()
