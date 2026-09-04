"""KV cache sizing / allocation / per-layer binding.

Moved out of the Engine so the engine's runtime surface stays focused on
forward + sampling. The allocator decides how many pages fit in the memory
budget, creates the paged ``KVCachePool``, and binds each attention layer to
its slice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from minisgl.config import ModelArgs, ServerArgs
    from minisgl.engine.kvcache.pool import KVCachePool

# Teaching simplification: on CPU there is no free-memory query, so the KV
# pool is sized against this fixed budget instead of the accelerator heuristic
# used on CUDA/NPU. 512 MB is enough to run the small (<=1B) models that are
# practical on CPU without letting the pool dominate host RAM.
_CPU_KV_CACHE_BYTES = 512 * 1024 * 1024  # 512 MB


class KVCacheAllocator:
    """Sizes, allocates, and binds the paged KV cache for one model."""

    def __init__(self, server_args: "ServerArgs", model_args: "ModelArgs") -> None:
        self.args = server_args
        self.model_args = model_args

    def available_memory(self, device: "torch.device") -> int:
        """Bytes available for the KV pool on ``device``."""
        if device.type in ("cuda", "npu"):
            from minisgl.utils.device import mem_get_info
            from minisgl.utils.logger import logger

            free_mem, total_mem = mem_get_info(device)
            used_mem = total_mem - free_mem
            available = int(total_mem * self.args.memory_ratio - used_mem)
            if available <= 0:
                # The model already fills the budget (memory_ratio over-used).
                # Silently allocating a 1-page pool turns every later request
                # into "needs more pages than the pool" aborts — fail loudly
                # instead so the operator can raise memory_ratio or use a
                # smaller model.
                logger.error(
                    "No memory left for KV cache: total=%d used=%d (ratio=%.2f). "
                    "Lower the model size or free GPU memory.",
                    total_mem,
                    used_mem,
                    self.args.memory_ratio,
                )
                raise RuntimeError("KV cache allocation failed: no free memory")
            return available
        return _CPU_KV_CACHE_BYTES

    def num_pages(self, available: int, num_kv_heads_per_rank: int, dtype_itemsize: int) -> int:
        """Number of pages that fit ``available`` bytes, capped by demand."""
        args, ma = self.args, self.model_args
        bytes_per_page = (
            2
            * ma.num_layers
            * args.page_size
            * num_kv_heads_per_rank
            * ma.head_dim
            * dtype_itemsize
        )
        num_pages = max(1, available // bytes_per_page)
        max_pages_needed = args.max_running_req * max(
            1,
            args.max_seq_len // args.page_size + 1,
        )
        return min(num_pages, max_pages_needed)

    def allocate(
        self,
        model,
        device: "torch.device",
        tp_size: int,
    ) -> "KVCachePool":
        """Create and bind the paged KV pool for ``model`` on ``device``."""
        from minisgl.engine.kvcache.pool import KVCachePool
        from minisgl.utils.logger import logger

        ma = self.model_args
        # Match the attention modules: replicate KV heads when num_kv_heads
        # is not divisible by tp_size (never allocate a 0-head buffer).
        num_kv_heads_per_rank = max(1, ma.num_kv_heads // tp_size)
        dtype = model.lm_head.weight.dtype

        available = self.available_memory(device)
        num_pages = self.num_pages(available, num_kv_heads_per_rank, dtype.itemsize)
        logger.info("Allocating KV cache: %d pages", num_pages)

        pool = KVCachePool(
            num_layers=ma.num_layers,
            num_pages=num_pages,
            page_size=self.args.page_size,
            num_kv_heads=num_kv_heads_per_rank,
            head_dim=ma.head_dim,
            dtype=dtype,
            device=device,
        )
        self.bind_layers(pool, model)
        return pool

    @staticmethod
    def bind_layers(pool: "KVCachePool", model) -> None:
        """Bind each attention layer to its slice of the paged KV cache pool.

        The pool buffer is (num_layers, num_pages, page_size, num_kv_heads,
        head_dim); modules() yields the layers' attention modules in layer
        order, so layer i gets slice i.
        """
        from minisgl.models.attention.layer import BaseAttention

        k_all, v_all = pool.get_all_kv_cache()
        layer_id = 0
        for module in model.modules():
            if isinstance(module, BaseAttention):
                module.set_kv_cache(k_all[layer_id], v_all[layer_id])
                layer_id += 1
