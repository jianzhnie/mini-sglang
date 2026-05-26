"""Sequence (request) and Batch data structures for the scheduler."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Literal, Optional

import torch

from minisgl.config import SamplingParams
from minisgl.engine.kvcache.pool import BaseCacheHandle


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Req:
    """A single inference request tracked by the scheduler."""
    input_ids: List[int] = field(default_factory=list)
    table_idx: int = 0
    cached_len: int = 0
    output_len: int = 0
    uid: int = 0
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    cache_handle: Optional[BaseCacheHandle] = None
    status: SequenceStatus = SequenceStatus.WAITING

    @property
    def total_len(self) -> int:
        return len(self.input_ids)

    @property
    def uncached_len(self) -> int:
        return len(self.input_ids) - self.cached_len

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    def append_token(self, token_id: int) -> None:
        self.input_ids.append(token_id)
        self.output_len += 1


@dataclass
class Batch:
    """A batch of requests for one forward pass."""
    reqs: List[Req] = field(default_factory=list)
    phase: Literal["prefill", "decode"] = "prefill"

    # Derived fields filled by BatchContext
    input_ids: Optional[torch.Tensor] = None       # (total_tokens,)
    positions: Optional[torch.Tensor] = None        # (total_tokens,)
    write_loc: Optional[torch.Tensor] = None        # (total_tokens,) page table indices
    req_to_token: Optional[torch.Tensor] = None     # (num_reqs, max_seq_len) page table

    def size(self) -> int:
        return len(self.reqs)
