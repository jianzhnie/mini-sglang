"""CUDA/NPU execution graph capture and replay for the decode phase."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minisgl.engine.engine import Engine
    from minisgl.scheduler.batch import Batch

__all__ = ["GraphRunner"]
import torch

from minisgl.engine.kvcache.pool import BaseCacheHandle
from minisgl.models.attention.metadata import AttentionMetadata
from minisgl.utils.logger import logger


class GraphRunner:
    """Captures and replays execution graphs (CUDA or NPU) for decode batches.

    Holds the captured graphs plus their static input/output buffers, and a
    trash page reserved from the KV pool: padding rows of a padded graph
    batch write/read KV into this page instead of polluting real requests'
    pages. The trash page is released together with the pool at cleanup.
    """

    def __init__(self, engine: "Engine") -> None:
        self.engine = engine
        self.device = engine.device
        self.args = engine.server_args
        self._run_model = engine._run_model

        self.graphs: dict[int, object] = {}
        self.inputs: dict[int, dict] = {}
        self.outputs: dict[int, torch.Tensor] = {}
        self.pad_handle: BaseCacheHandle | None = None
        self.pad_page_id = 0
        self.pad_loc = 0

        self._capture_graphs()

    def clear(self) -> None:
        """Drop all captured graphs and their static buffers."""
        self.graphs.clear()
        self.inputs.clear()
        self.outputs.clear()

    def replay(self, batch: Batch) -> torch.Tensor | None:
        """Replay the smallest captured graph that fits the batch.

        Returns None when no captured graph is large enough; the caller then
        falls back to eager execution.
        """
        bs = len(batch.reqs)
        graph_bs = next((s for s in sorted(self.graphs) if s >= bs), None)
        if graph_bs is None:
            return None
        ins = self.inputs[graph_bs]
        meta = ins["attn_meta"]
        with torch.inference_mode():
            ins["input_ids"][:bs].copy_(batch.input_ids)
            ins["positions"][:bs].copy_(batch.positions)
            meta.write_loc[:bs].copy_(batch.attn_meta.write_loc)
            meta.cache_seqlens[:bs].copy_(batch.attn_meta.cache_seqlens)
            meta.block_table[:bs].copy_(batch.attn_meta.block_table)
            meta.req_to_token[:bs].copy_(batch.attn_meta.req_to_token)
            # Padding rows point at the reserved trash page so their KV
            # writes and reads never touch real requests' pages.
            if bs < graph_bs:
                meta.write_loc[bs:].fill_(self.pad_loc)
                meta.cache_seqlens[bs:].fill_(1)
                meta.block_table[bs:].fill_(self.pad_page_id)
                meta.req_to_token[bs:].fill_(self.pad_loc)
            self.graphs[graph_bs].replay()
            # Clone: the output is a static buffer the next replay overwrites.
            return self.outputs[graph_bs][:bs].clone()

    def _capture_graphs(self) -> None:
        """Capture execution graphs for the decode phase (CUDA or NPU)."""
        args = self.args
        max_bs = min(args.cuda_graph_bs, args.max_running_req)
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        batch_sizes = [bs for bs in batch_sizes if bs <= max_bs]

        # Reserve one trash page from the pool: padding rows of padded graph
        # batches write/read KV into this page instead of polluting real
        # requests. Released together with the pool at cleanup.
        try:
            self.pad_handle = self.engine.kv_cache_pool.alloc(1)
        except RuntimeError as e:
            logger.warning(
                "Cannot reserve a pad page for graphs (%s); skipping capture", e
            )
            return
        self.pad_page_id = self.pad_handle.page_ids[0]
        self.pad_loc = self.pad_page_id * args.page_size

        logger.info(
            "Capturing %s graphs for batch sizes: %s",
            self.engine._device_type.upper(),
            batch_sizes,
        )

        for bs in batch_sizes:
            self._capture_graph_safe(bs)

    def _capture_graph_safe(self, batch_size: int) -> None:
        """Capture an execution graph, logging but not raising on failure.

        On failure the engine simply falls back to eager execution for this
        batch size (decode works without graphs, just slower).
        """
        try:
            self._capture_graph(batch_size)
        except RuntimeError as e:
            logger.warning(
                "Failed to capture graph for bs=%d: %s. "
                "Falling back to eager execution for this batch size.",
                batch_size,
                e,
            )

    def _capture_graph(self, batch_size: int) -> None:
        """Capture a single execution graph for a given batch size.

        Supports both CUDA graphs and NPU graphs (via torch.npu). Static
        input buffers mirror the real decode metadata produced by
        DecodeManager (same shapes/dtypes), so replay only needs copy_
        into these buffers. All entries point at the trash pad page.
        """
        device = self.device
        args = self.args
        max_blocks = (args.max_seq_len + args.page_size - 1) // args.page_size
        pad_page = self.pad_page_id
        pad_loc = self.pad_loc

        input_ids = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        positions = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        write_loc = torch.full((batch_size,), pad_loc, dtype=torch.int32, device=device)
        cache_seqlens = torch.ones(batch_size, dtype=torch.int32, device=device)
        block_table = torch.full(
            (batch_size, max_blocks), pad_page, dtype=torch.int32, device=device
        )
        req_to_token = torch.full(
            (batch_size, args.max_seq_len), pad_loc, dtype=torch.int32, device=device
        )
        # One resident metadata object wrapping the static buffers: the
        # captured graph reads through it, and replay only copy_()s new
        # values into the same buffers (the object itself never changes).
        attn_meta = AttentionMetadata(
            forward_mode="decode",
            write_loc=write_loc,
            block_table=block_table,
            req_to_token=req_to_token,
            cache_seqlens=cache_seqlens,
            # Static Python int: keeps the captured graph free of .item()
            # host syncs in the PyTorch attention backend.
            max_seqlen=args.max_seq_len,
        )

        def run_decode() -> torch.Tensor:
            return self._run_model(input_ids, positions, attn_meta)

        with torch.inference_mode():
            # Warmup
            for _ in range(3):
                run_decode()

            if self.engine._device_type == "npu":
                import os

                os.environ.setdefault("TASK_QUEUE_ENABLE", "1")
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    output = run_decode()
            else:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = run_decode()

        self.graphs[batch_size] = graph
        self.inputs[batch_size] = {
            "input_ids": input_ids,
            "positions": positions,
            "attn_meta": attn_meta,
        }
        self.outputs[batch_size] = output
