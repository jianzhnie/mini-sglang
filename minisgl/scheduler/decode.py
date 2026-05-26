"""Decode manager: schedules running requests for token-by-token generation."""

__all__ = ["DecodeManager"]
import torch

from minisgl.config import ServerArgs
from minisgl.engine.kvcache.pool import KVCachePool
from minisgl.engine.kvcache.radix import RadixCacheManager
from minisgl.scheduler.batch import Batch, Req


class DecodeManager:
    """Manages the decode queue of actively generating requests.


    Naive implementation: re-processes all tokens each step
    (no KV cache for simplicity and correctness).
    """

    def __init__(
        self,
        args: ServerArgs,
        pool: KVCachePool,
        radix_cache: RadixCacheManager,
    ) -> None:
        self.max_running_req = args.max_running_req
        self.page_size = args.page_size
        self.pool = pool
        self.radix_cache = radix_cache
        self.naive_mode = True  # Skip KV cache for now

    def schedule_decode(self, running: list[Req]) -> Batch | None:
        if not running:
            return None

        # In real decode mode (with KV cache), only process the last token per request.
        # In naive mode (no KV cache), re-process all tokens.
        if self.naive_mode:
            all_input_ids = []
            all_positions = []
            for req in running:
                all_input_ids.extend(req.input_ids)
                all_positions.extend(range(len(req.input_ids)))
        else:
            all_input_ids = [req.input_ids[-1] for req in running]
            all_positions = [len(req.input_ids) - 1 for req in running]

        if not all_input_ids:
            return None

        batch = Batch(reqs=running, phase="decode")
        batch.input_ids = torch.tensor(all_input_ids, dtype=torch.long, device="cpu")
        batch.positions = torch.tensor(all_positions, dtype=torch.long, device="cpu")
        return batch
