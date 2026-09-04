"""Sequence (request) and Batch data structures for the scheduler."""

__all__ = ["Batch", "OutputToken", "Req", "SequenceStatus"]
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal

import torch

from minisgl.config import SamplingParams
from minisgl.engine.kvcache.pool import BaseCacheHandle
from minisgl.models.attention.metadata import AttentionMetadata


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass(slots=True)
class OutputToken:
    """One generated token reported by Scheduler.step()."""

    uid: int
    token_id: int
    finished: bool
    # None while in progress; on the final token one of:
    #   "stop" / "length" / "abort" / "error"
    # ("abort" / "error" mean token_id is meaningless: the request never
    # produced a usable token — it was rejected up front or died mid-run).
    finish_reason: str | None = None


@dataclass(slots=True)
class Req:
    """A single inference request tracked by the scheduler."""

    input_ids: list[int] = field(default_factory=list)
    cached_len: int = 0
    output_len: int = 0
    uid: int = 0
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    cache_handle: BaseCacheHandle | None = None
    status: SequenceStatus = SequenceStatus.WAITING

    @property
    def uncached_len(self) -> int:
        return len(self.input_ids) - self.cached_len

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    def append_token(self, token_id: int) -> None:
        self.input_ids.append(token_id)
        self.output_len += 1


@dataclass(slots=True)
class Batch:
    """A batch of requests for one forward pass."""

    reqs: list[Req] = field(default_factory=list)
    phase: Literal["prefill", "decode"] = "prefill"

    # Derived fields filled by BatchContext (prefill) / DecodeManager (decode)
    input_ids: torch.Tensor | None = None  # (total_tokens,)
    positions: torch.Tensor | None = None  # (total_tokens,)
    # Attention inputs for this batch: KV write slots, page tables, varlen
    # boundaries, sequence lengths. See AttentionMetadata.
    attn_meta: AttentionMetadata | None = None
    # (num_reqs,) index of each request's last uncached token in the flat
    # prefill batch — prefill only runs lm_head on these positions.
    logits_indices: torch.Tensor | None = None
