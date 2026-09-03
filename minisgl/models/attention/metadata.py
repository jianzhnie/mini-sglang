"""Typed per-batch attention metadata.

Replaces the old five-level ``**kwargs`` passthrough chain
(CausalLM -> Model -> decoder layer -> attention module -> backend) with a
single typed object, built on the scheduler side (BatchContext for prefill,
DecodeManager for decode) and consumed by the attention backends.

The KV cache tensors themselves are NOT part of this object: each attention
layer holds its own slice of the paged pool (see BaseAttention.set_kv_cache).
"""

from __future__ import annotations

__all__ = ["AttentionMetadata"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(slots=True)
class AttentionMetadata:
    """Per-batch attention inputs, built by the scheduler side, consumed by backends.

    Fields
    ------
    forward_mode:
        "prefill" or "decode".
    write_loc:
        (total_tokens,) flat KV-cache slots the current tokens are written
        to; -1 entries are skipped. Used by both phases.
    cu_seqlens_q:
        (num_reqs+1,) varlen query boundaries (prefill).
    prefix_lens:
        (num_reqs,) cached prefix length per request. Nonzero entries turn a
        prefill into extend attention: the prefix KV is read back from the
        paged cache (the suffix KV was written there just before).
    block_table:
        (num_reqs, max_blocks) page IDs per request — the page table dialect
        of the FlashAttention backend.
    req_to_token:
        (num_reqs, max_seq_len) flat cache slot per position — the page
        table dialect of the PyTorch backend.
    cache_seqlens:
        (num_reqs,) total length per request INCLUDING the current token
        (decode; also what flash_attn_with_kvcache expects).
    max_seqlen:
        Batch max sequence length as a Python int, so backends size gathers
        without a .item() host sync (illegal during CUDA graph capture).

    A None metadata object (``attn_meta=None`` on the model forward) means
    plain causal self-attention without a KV cache — used by tests and
    teaching demos. But once a metadata object exists, missing fields that
    the mode requires (e.g. decode without req_to_token) are an error, not a
    silent fallback.
    """

    forward_mode: str  # "prefill" | "decode"
    write_loc: Tensor | None  # KV write slots
    cu_seqlens_q: Tensor | None = None  # prefill varlen
    prefix_lens: Tensor | None = None  # extend: cached prefix per request
    block_table: Tensor | None = None  # paged table (FA)
    req_to_token: Tensor | None = None  # paged table (PT)
    cache_seqlens: Tensor | None = None  # decode: total len per request
    max_seqlen: int | None = None
