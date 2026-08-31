"""Batch context management for engine forward pass.

Prepares derived tensors from Batch metadata:
- input_ids: concatenated token IDs
- positions: position encodings
- write_loc: KV cache write locations (page table indices)
- req_to_token: page table (num_reqs, max_seq_len)
- cu_seqlens_q: cumulative sequence lengths for FlashAttention varlen
- prefix_lens: cached prefix length per request (extend attention)
- block_table: page IDs per request (paged KV cache attention)
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
            start_pos = req.cached_len
            end_pos = len(req.input_ids)
            all_positions.extend(range(start_pos, end_pos))
            seq_lengths.append(len(uncached_tokens))

            if req.cache_handle:
                pages = req.cache_handle.page_ids
                for pos in range(start_pos, end_pos):
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

        seq_lens_t = torch.tensor(seq_lengths, dtype=torch.int32)
        cu = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32)
        cu[1:] = seq_lens_t.cumsum(0)
        batch.cu_seqlens_q = cu.to(device=self.device)
        # Max uncached length as a Python int — backends use it for FA varlen
        # sizing without a host sync (.item()).
        batch.max_seqlen = max(seq_lengths) if seq_lengths else 0

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

        # Cached prefix length per request (extend attention reads these KV
        # entries from the shared prefix pages in req_to_token).
        batch.prefix_lens = torch.tensor(
            [req.cached_len for req in reqs],
            dtype=torch.int32,
            device=self.device,
        )

        # Full page table (shared prefix pages first) for backends that
        # address the KV cache by page ID (e.g. flash_attn_with_kvcache).
        max_blocks = (self.max_seq_len + self.page_size - 1) // self.page_size
        block_table = torch.full(
            (num_reqs, max_blocks), -1, dtype=torch.int32, device=self.device
        )
        for i, req in enumerate(reqs):
            handle = req.cache_handle
            if handle is not None:
                for j, page_id in enumerate(handle.page_ids):
                    if j >= max_blocks:
                        break
                    block_table[i, j] = page_id
        batch.block_table = block_table
