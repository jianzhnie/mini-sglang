"""Decode manager: schedules running requests for token-by-token generation."""

__all__ = ["DecodeManager"]
import torch

from minisgl.config import ServerArgs
from minisgl.engine.kvcache.pool import KVCachePool
from minisgl.engine.kvcache.radix import RadixCacheManager
from minisgl.scheduler.batch import Batch, Req


class DecodeManager:
    """Manages the decode queue of actively generating requests.

    Uses KV cache for incremental decoding: only the last token per request
    is processed, while all previous tokens are read from the KV cache.
    """

    def __init__(
        self,
        args: ServerArgs,
        pool: KVCachePool,
        radix_cache: RadixCacheManager,
        device: torch.device | None = None,
    ) -> None:
        self.max_running_req = args.max_running_req
        self.max_seq_len = args.max_seq_len
        self.page_size = args.page_size
        self.pool = pool
        self.radix_cache = radix_cache
        self.device = device or torch.device("cpu")

    def schedule_decode(self, running: list[Req]) -> Batch | None:
        if not running:
            return None

        all_input_ids = [req.input_ids[-1] for req in running]
        all_positions = [len(req.input_ids) - 1 for req in running]

        if not all_input_ids:
            return None

        batch = Batch(reqs=running, phase="decode")
        batch.input_ids = torch.tensor(
            all_input_ids, dtype=torch.long, device=self.device
        )
        batch.positions = torch.tensor(
            all_positions, dtype=torch.long, device=self.device
        )

        write_loc: list[int] = []
        cache_seqlens: list[int] = []
        num_reqs = len(running)
        max_blocks = (self.max_seq_len + self.page_size - 1) // self.page_size
        block_table = torch.full(
            (num_reqs, max_blocks), -1, dtype=torch.int32, device=self.device
        )

        for i, req in enumerate(running):
            total_len = len(req.input_ids)
            cache_seqlens.append(total_len)

            # Write location for the current token
            if req.cache_handle:
                pos = total_len - 1
                page_idx = pos // self.page_size
                if page_idx < len(req.cache_handle.page_ids):
                    loc = (
                        req.cache_handle.page_ids[page_idx] * self.page_size
                        + pos % self.page_size
                    )
                    write_loc.append(loc)
                else:
                    write_loc.append(-1)
            else:
                write_loc.append(-1)

            # Build block_table row: page indices for this request
            if req.cache_handle:
                for j, pid in enumerate(req.cache_handle.page_ids):
                    if j < max_blocks:
                        block_table[i, j] = pid

        batch.write_loc = torch.tensor(
            write_loc, dtype=torch.int32, device=self.device
        )
        batch.cache_seqlens = torch.tensor(
            cache_seqlens, dtype=torch.int32, device=self.device
        )
        batch.block_table = block_table
        return batch
