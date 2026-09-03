"""Batch context management for engine forward pass.

Prepares derived tensors from Batch metadata:
- input_ids: concatenated token IDs
- positions: position encodings
- attn_meta: typed AttentionMetadata — write_loc (KV write slots),
  req_to_token / block_table (page tables), cu_seqlens_q (varlen
  boundaries), prefix_lens (cached prefix per request), max_seqlen
- logits_indices: last-uncached-token index per request (prefill lm_head)
"""

__all__ = ["BatchContext"]
import torch

from minisgl.models.attention.metadata import AttentionMetadata
from minisgl.scheduler.batch import Batch, Req


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
        """Fill derived tensors from batch metadata.

        Every tensor is built on the CPU first and uploaded in one shot:
        creating many small tensors directly on the accelerator (or writing
        per-page slices into a device-side table) is a slow path.
        `non_blocking=True` only matters for CUDA pinned copies and is
        harmless elsewhere.
        """
        reqs = batch.reqs

        all_input_ids = []
        all_positions = []
        write_loc_parts = []
        seq_lengths = []
        # Per-request index of the last uncached token in the flat batch.
        # Prefill only needs lm_head logits at these positions.
        logits_indices = []

        offset = 0
        for req in reqs:
            uncached_tokens = req.input_ids[req.cached_len :]
            all_input_ids.extend(uncached_tokens)
            start_pos = req.cached_len
            end_pos = len(req.input_ids)
            all_positions.extend(range(start_pos, end_pos))
            seq_lengths.append(len(uncached_tokens))
            logits_indices.append(offset + len(uncached_tokens) - 1)
            offset += len(uncached_tokens)

            if req.cache_handle:
                # Vectorized write_loc (same trick as _build_req_to_token):
                # position p maps to page_ids[p // ps] * ps + p % ps, with the
                # -1 sentinel kept for positions beyond the allocated pages.
                pages = torch.tensor(req.cache_handle.page_ids, dtype=torch.int32)
                pos = torch.arange(start_pos, end_pos, dtype=torch.int32)
                page_idx = pos // self.page_size
                valid = page_idx < len(pages)
                locs = torch.full_like(pos, -1)
                locs[valid] = (
                    pages[page_idx[valid]] * self.page_size
                    + pos[valid] % self.page_size
                )
                write_loc_parts.append(locs)
            else:
                write_loc_parts.append(
                    torch.full((req.uncached_len,), -1, dtype=torch.int32)
                )

        batch.input_ids = torch.tensor(all_input_ids, dtype=torch.long).to(
            self.device, non_blocking=True
        )
        batch.positions = torch.tensor(all_positions, dtype=torch.long).to(
            self.device, non_blocking=True
        )

        write_loc = None
        if write_loc_parts:
            write_loc = torch.cat(write_loc_parts).to(self.device, non_blocking=True)

        batch.logits_indices = torch.tensor(logits_indices, dtype=torch.long).to(
            self.device, non_blocking=True
        )

        seq_lens_t = torch.tensor(seq_lengths, dtype=torch.int32)
        cu = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32)
        cu[1:] = seq_lens_t.cumsum(0)

        # Cached prefix length per request (extend attention reads these KV
        # entries from the shared prefix pages in req_to_token).
        prefix_lens = torch.tensor(
            [req.cached_len for req in reqs], dtype=torch.int32
        ).to(self.device, non_blocking=True)

        # Full page table (shared prefix pages first) for backends that
        # address the KV cache by page ID (e.g. flash_attn_with_kvcache).
        max_blocks = (self.max_seq_len + self.page_size - 1) // self.page_size
        rows = []
        for req in reqs:
            handle = req.cache_handle
            ids = list(handle.page_ids[:max_blocks]) if handle is not None else []
            rows.append(ids + [-1] * (max_blocks - len(ids)))
        block_table = torch.tensor(rows, dtype=torch.int32).to(
            self.device, non_blocking=True
        )

        batch.attn_meta = AttentionMetadata(
            forward_mode="prefill",
            write_loc=write_loc,
            cu_seqlens_q=cu.to(device=self.device, non_blocking=True),
            prefix_lens=prefix_lens,
            block_table=block_table,
            req_to_token=self._build_req_to_token(reqs),
            # Max uncached length as a Python int — backends use it for FA
            # varlen sizing without a host sync (.item()).
            max_seqlen=max(seq_lengths) if seq_lengths else 0,
        )

    def _build_req_to_token(self, reqs: list[Req]) -> torch.Tensor:
        """Build the (num_reqs, max_seq_len) page table on CPU, then upload.

        Each row is filled with vectorized torch ops instead of per-page
        slice writes on the accelerator: column c of row i holds the flat
        cache slot `page_ids[c // page_size] * page_size + c % page_size`,
        with -1 beyond the request's length or allocated pages.
        """
        page_size = self.page_size
        table = torch.full((len(reqs), self.max_seq_len), -1, dtype=torch.int32)
        for i, req in enumerate(reqs):
            handle = req.cache_handle
            if handle is None:
                continue
            n_filled = min(len(req.input_ids), len(handle.page_ids) * page_size)
            if n_filled <= 0:
                continue
            pages = torch.tensor(handle.page_ids, dtype=torch.int32)
            cols = torch.arange(n_filled, dtype=torch.int32)
            table[i, :n_filled] = (
                pages[cols // page_size] * page_size + cols % page_size
            )
        return table.to(self.device, non_blocking=True)
