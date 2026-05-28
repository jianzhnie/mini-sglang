"""Batch context management for engine forward pass.

Prepares derived tensors from Batch metadata:
- input_ids: concatenated token IDs
- positions: position encodings
- write_loc: KV cache write locations (page table indices)
- req_to_token: page table (num_reqs, max_seq_len)
- cu_seqlens_q: cumulative sequence lengths for FlashAttention varlen
"""

__all__ = ["BatchContext"]
import torch

from minisgl.scheduler.batch import Batch


class BatchContext:
    """Manages derived tensors needed for a forward pass."""

    def __init__(
        self,
        max_running_req: int,
        max_seq_len: int,
        page_size: int,
        device: torch.device,
    ) -> None:
        self.max_running_req = max_running_req
        self.max_seq_len = max_seq_len
        self.page_size = page_size
        self.device = device

    def prepare(self, batch: Batch) -> None:
        """Fill derived tensors from batch metadata."""
        reqs = batch.reqs

        all_input_ids = []
        all_positions = []
        write_loc = []
        seq_lengths = []

        for req in reqs:
            uncached_tokens = req.input_ids[req.cached_len :]
            all_input_ids.extend(uncached_tokens)
            all_positions.extend(range(req.cached_len, len(req.input_ids)))
            seq_lengths.append(len(uncached_tokens))

            if req.cache_handle:
                pages = req.cache_handle.page_ids
                for pos in range(req.cached_len, len(req.input_ids)):
                    page_idx = pos // self.page_size
                    if page_idx < len(pages):
                        loc = pages[page_idx] * self.page_size + (pos % self.page_size)
                        write_loc.append(loc)
                    else:
                        write_loc.append(-1)
            else:
                write_loc.extend([-1] * req.uncached_len)

        batch.input_ids = torch.tensor(
            all_input_ids,
            dtype=torch.long,
            device=self.device,
        )
        batch.positions = torch.tensor(
            all_positions,
            dtype=torch.long,
            device=self.device,
        )

        if write_loc:
            batch.write_loc = torch.tensor(
                write_loc,
                dtype=torch.int32,
                device=self.device,
            )

        # Build cu_seqlens for FlashAttention varlen
        if len(seq_lengths) > 1:
            cu = torch.tensor(
                [0] + torch.tensor(seq_lengths).cumsum(0).tolist(),
                dtype=torch.int32,
                device=self.device,
            )
            batch.cu_seqlens_q = cu
        else:
            batch.cu_seqlens_q = torch.tensor(
                [0, seq_lengths[0]] if seq_lengths else [0],
                dtype=torch.int32,
                device=self.device,
            )

        # Build req_to_token (page table): (num_reqs, max_seq_len)
        num_reqs = len(reqs)
        table = torch.full(
            (num_reqs, self.max_seq_len),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        for i, req in enumerate(reqs):
            handle = req.cache_handle
            if handle is not None:
                pages = handle.page_ids
                total = len(req.input_ids)
                for p_idx, page_id in enumerate(pages):
                    start = p_idx * self.page_size
                    end = min((p_idx + 1) * self.page_size, total)
                    if start >= total:
                        break
                    count = end - start
                    table[i, start:end] = page_id * self.page_size + torch.arange(
                        count, dtype=torch.int32, device=self.device
                    )

        batch.req_to_token = table
