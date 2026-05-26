"""Sampling strategies: greedy, top-k, top-p (nucleus), temperature."""

__all__ = ["Sampler"]
import torch
import torch.nn.functional as F

from minisgl.config import SamplingParams


class Sampler:
    """Token sampler supporting greedy, top-k, top-p, and temperature sampling."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def sample(
        self,
        logits: torch.Tensor,  # (batch_size, vocab_size)
        sampling_params: SamplingParams,
    ) -> torch.Tensor:
        """Sample next tokens for a batch of requests.

        Args:
            logits: Raw logits (batch_size, vocab_size).
            sampling_params: Sampling configuration.

        Returns:
            Tensor of token IDs (batch_size,).
        """
        temperature = sampling_params.temperature

        if temperature <= 0.0:
            return logits.argmax(dim=-1)

        # Apply temperature
        logits = logits / temperature

        # Apply top-k filtering
        if sampling_params.top_k > 0:
            logits = _apply_top_k(logits, sampling_params.top_k)

        # Apply top-p (nucleus) filtering
        if sampling_params.top_p < 1.0:
            logits = _apply_top_p(logits, sampling_params.top_p)

        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits below the top-k threshold."""
    top_k_values, _ = logits.topk(k, dim=-1)
    threshold = top_k_values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Zero out logits below the top-p (nucleus) threshold."""
    sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative probability above p (keep first token above threshold)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))
