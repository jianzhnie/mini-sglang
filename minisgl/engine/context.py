"""Batch context management for engine forward pass.

Prepares derived tensors from Batch metadata:
- input_ids: concatenated token IDs
- positions: position encodings
- write_loc: KV cache write locations (page table indices)
- req_to_token: page table (num_reqs, max_seq_len)
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
    ):
        self.max_running_req = max_running_req
        self.max_seq_len = max_seq_len
        self.page_size = page_size
        self.device = device

    def prepare(self, batch: Batch) -> None:
        """Fill derived tensors from batch metadata."""
        reqs = batch.reqs

        # Collect all token info
        all_input_ids = []
        all_positions = []
        write_loc = []

        for req in reqs:
            uncached_tokens = req.input_ids[req.cached_len :]
            all_input_ids.extend(uncached_tokens)
            all_positions.extend(range(req.cached_len, len(req.input_ids)))

            # Map positions to KV cache page indices
            if req.cache_handle:
                for pos in range(req.cached_len, len(req.input_ids)):
                    page_idx = pos // self.page_size
                    offset = pos % self.page_size
                    if page_idx < len(req.cache_handle.page_ids):
                        loc = (
                            req.cache_handle.page_ids[page_idx] * self.page_size
                            + offset
                        )
                        write_loc.append(loc)
                    else:
                        write_loc.append(-1)
            else:
                write_loc.extend([-1] * req.uncached_len)

        batch.input_ids = torch.tensor(
            all_input_ids, dtype=torch.long, device=self.device
        )
        batch.positions = torch.tensor(
            all_positions, dtype=torch.long, device=self.device
        )

        if write_loc:
            batch.write_loc = torch.tensor(
                write_loc, dtype=torch.int32, device=self.device
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
            if req.cache_handle:
                for pos in range(len(req.input_ids)):
                    page_idx = pos // self.page_size
                    if page_idx < len(req.cache_handle.page_ids):
                        offset = pos % self.page_size
                        table[i, pos] = (
                            req.cache_handle.page_ids[page_idx] * self.page_size
                            + offset
                        )

        batch.req_to_token = table
