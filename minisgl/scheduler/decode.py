"""Decode manager: schedules running requests for token-by-token generation."""

__all__ = ["DecodeManager"]
import torch

from minisgl.config import ServerArgs
from minisgl.engine.kvcache.pool import KVCachePool
from minisgl.models.attention.metadata import AttentionMetadata
from minisgl.scheduler.batch import Batch, Req


class DecodeManager:
    """Manages the decode queue of actively generating requests.

    Uses KV cache for incremental decoding: only the last token per request
    is processed, while all previous tokens are read from the KV cache.
    Pre-allocates tensors to minimize per-step allocation overhead.
    """

    def __init__(
        self,
        args: ServerArgs,
        pool: KVCachePool,
        device: torch.device | None = None,
    ) -> None:
        self.max_running_req = args.max_running_req
        self.max_seq_len = args.max_seq_len
        self.page_size = args.page_size
        self.pool = pool
        self.device = device or torch.device("cpu")
        self._max_blocks = (self.max_seq_len + self.page_size - 1) // self.page_size

        # Column indices for the vectorized req_to_token row fill: column c of
        # a row holds page_ids[c // page_size] * page_size + c % page_size.
        self._cols_buf = torch.arange(
            self.max_seq_len, dtype=torch.int32, device=self.device
        )

        # Device-side cache of each page table. A request's page_ids only
        # change at page boundaries / on eviction, so rebuilding the tensor
        # every decode step is wasteful; key by page_ids content (a tuple) so
        # reuse is always safe even if a handle object is recycled. Bounded to
        # a few distinct tables (active requests rarely exceed this) so it
        # cannot grow without limit over a long run.
        self._pages_cache: dict[tuple[int, ...], torch.Tensor] = {}
        self._pages_cache_max = 64

        self._input_ids_buf = torch.zeros(
            self.max_running_req, 1, dtype=torch.long, device=self.device
        )
        self._positions_buf = torch.zeros(
            self.max_running_req, 1, dtype=torch.long, device=self.device
        )
        self._write_loc_buf = torch.full(
            (self.max_running_req,), -1, dtype=torch.int32, device=self.device
        )
        self._cache_seqlens_buf = torch.zeros(
            self.max_running_req, dtype=torch.int32, device=self.device
        )
        self._block_table_buf = torch.full(
            (self.max_running_req, self._max_blocks),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self._req_to_token_buf = torch.full(
            (self.max_running_req, self.max_seq_len),
            -1,
            dtype=torch.int32,
            device=self.device,
        )

    def schedule_decode(self, running: list[Req]) -> Batch | None:
        if not running:
            return None

        num_reqs = len(running)
        batch = Batch(reqs=running, phase="decode")

        # Max total length this step (Python int — attention backends use it
        # to size gathers without a host sync).
        max_seqlen = max(len(req.input_ids) for req in running)

        # Clear only the columns the backend can read this step (with one
        # page of margin) instead of the full-width tables: backends never
        # touch req_to_token[:, max_seqlen:] or block_table columns beyond
        # ceil(max_seqlen / page_size).
        token_cols = min(
            self.max_seq_len,
            (max_seqlen + self.page_size - 1) // self.page_size * self.page_size
            + self.page_size,
        )
        block_cols = min(
            self._max_blocks,
            (max_seqlen + self.page_size - 1) // self.page_size + 1,
        )
        self._block_table_buf[:num_reqs, :block_cols].fill_(-1)
        self._req_to_token_buf[:num_reqs, :token_cols].fill_(-1)

        for i, req in enumerate(running):
            total_len = len(req.input_ids)
            self._input_ids_buf[i, 0] = req.input_ids[-1]
            self._positions_buf[i, 0] = total_len - 1
            # cache_seqlens semantics: total length INCLUDING the current
            # token (matches flash_attn_with_kvcache).
            self._cache_seqlens_buf[i] = total_len

            handle = req.cache_handle
            if handle is not None:
                pos = total_len - 1
                page_idx = pos // self.page_size
                if page_idx < len(handle.page_ids):
                    self._write_loc_buf[i] = (
                        handle.page_ids[page_idx] * self.page_size
                        + pos % self.page_size
                    )
                else:
                    self._write_loc_buf[i] = -1

                # Reuse the device-side page-table tensor when this page list
                # was seen before (page tables change only on alloc/evict).
                key = tuple(handle.page_ids)
                pages = self._pages_cache.get(key)
                if pages is None:
                    if len(self._pages_cache) >= self._pages_cache_max:
                        # Drop one arbitrary entry to stay bounded.
                        self._pages_cache.pop(next(iter(self._pages_cache)))
                    pages = torch.tensor(
                        key, dtype=torch.int32, device=self.device
                    )
                    self._pages_cache[key] = pages
                n_blocks = min(len(handle.page_ids), self._max_blocks)
                self._block_table_buf[i, :n_blocks] = pages[:n_blocks]

                # Vectorized row fill into the pre-allocated buffer (same
                # trick as BatchContext._build_req_to_token): column c holds
                # page_ids[c // page_size] * page_size + c % page_size.
                n_filled = min(total_len, len(handle.page_ids) * self.page_size)
                cols = self._cols_buf[:n_filled]
                self._req_to_token_buf[i, :n_filled] = (
                    pages[cols // self.page_size] * self.page_size
                    + cols % self.page_size
                )
            else:
                self._write_loc_buf[i] = -1

        batch.input_ids = self._input_ids_buf[:num_reqs]
        batch.positions = self._positions_buf[:num_reqs]
        batch.attn_meta = AttentionMetadata(
            forward_mode="decode",
            # Views (no clone): the metadata object never outlives this step's
            # forward/sample — the next schedule_decode() first fill_()s these
            # buffers and then builds a fresh metadata object. A backend or
            # GraphRunner.replay reads these rows during the current forward
            # only, so aliasing the pre-allocated buffers is safe and skips two
            # small per-step copies.
            write_loc=self._write_loc_buf[:num_reqs],
            block_table=self._block_table_buf[:num_reqs],
            req_to_token=self._req_to_token_buf[:num_reqs],
            cache_seqlens=self._cache_seqlens_buf[:num_reqs],
            # Python int, so attention backends can size gathers without .item().
            max_seqlen=max_seqlen,
        )
        return batch
