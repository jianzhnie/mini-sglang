"""Vocabulary parallel embedding."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from minisgl.utils.device import get_tp_rank, get_tp_size, is_distributed


class VocabParallelEmbedding(nn.Module):
    """Embedding layer with vocabulary sharded across TP ranks."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        tp_size = get_tp_size()
        tp_rank = get_tp_rank()
        self.vocab_start = tp_rank * (num_embeddings // tp_size)
        self.vocab_end = (tp_rank + 1) * (num_embeddings // tp_size)
        self.num_embeddings_per_rank = num_embeddings // tp_size
        self.original_vocab_size = num_embeddings

        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_rank, embedding_dim)
        )
        self.weight.is_vocab_parallel = True

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward embedding lookup.

        Token IDs in [vocab_start, vocab_end) are looked up locally;
        out-of-range tokens return zero.
        """
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        safe_ids = (input_ids - self.vocab_start).clamp(0, self.num_embeddings_per_rank - 1)
        out = F.embedding(safe_ids, self.weight)
        out = out * mask.unsqueeze(-1).to(out.dtype)
        if is_distributed():
            from minisgl.engine.distributed.pynccl import all_reduce
            out = all_reduce(out)
        return out
