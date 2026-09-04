"""Sampling strategies: greedy, top-k, top-p (nucleus), temperature."""

__all__ = ["Sampler"]
import torch
import torch.nn.functional as F

from minisgl.config import SamplingParams


class Sampler:
    """Token sampler supporting greedy, top-k, top-p, and temperature sampling."""

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

        # Apply top-k filtering (clamp k to vocab size; top_k <= 0 disables it)
        top_k = min(sampling_params.top_k, logits.shape[-1])
        if top_k > 0:
            logits = _apply_top_k(logits, top_k)

        # Apply top-p (nucleus) filtering
        if sampling_params.top_p < 1.0:
            logits = _apply_top_p(logits, sampling_params.top_p)

        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def sample_batch(
        self,
        logits: torch.Tensor,  # (num_reqs, vocab_size)
        params_list: list[SamplingParams],
    ) -> list[int]:
        """Sample one token per request, batching requests that share params.

        Requests whose sampling params are identical are sampled together (one
        kernel call); requests are otherwise independent. ``logits`` row ``i``
        belongs to ``params_list[i]``.

        Returns:
            A list of ``len(params_list)`` token IDs.
        """
        # Fast path: all greedy -> one argmax over the whole batch.
        if all(p.temperature <= 0.0 for p in params_list):
            return logits.argmax(dim=-1).tolist()

        # Group rows by identical (temperature, top_k, top_p) so each group is
        # one batched sample call.
        groups: dict[tuple, list[int]] = {}
        for i, params in enumerate(params_list):
            key = (params.temperature, params.top_k, params.top_p)
            groups.setdefault(key, []).append(i)

        token_ids = [0] * len(params_list)
        for indices in groups.values():
            sampled = self.sample(logits[indices], params_list[indices[0]]).tolist()
            for j, idx in enumerate(indices):
                token_ids[idx] = sampled[j]
        return token_ids


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
        1,
        sorted_indices,
        sorted_indices_to_remove,
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))
