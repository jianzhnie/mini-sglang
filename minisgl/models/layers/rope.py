"""Rotary Position Embedding (RoPE)."""

__all__ = ["RotaryEmbedding", "apply_rotary_emb"]
import torch


class RotaryEmbedding:
    """Manages precomputed cos/sin tables for rotary position embeddings."""

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000.0,
    ) -> None:
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        # Precompute cos/sin tables
        self._cos_table: torch.Tensor | None = None
        self._sin_table: torch.Tensor | None = None
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int) -> None:
        """Build cos/sin cache tables up to seq_len."""
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_table = (
            emb.cos().unsqueeze(0).unsqueeze(0)
        )  # (1, 1, seq_len, head_dim)
        self._sin_table = emb.sin().unsqueeze(0).unsqueeze(0)

    def _ensure_cache(self, seq_len: int) -> None:
        if self._cos_table is None or seq_len > self._cos_table.shape[2]:
            self._build_cache(max(seq_len, self.max_position_embeddings))

    def prebuild(self, seq_len: int, device: torch.device) -> None:
        """Build cos/sin tables up to seq_len and move them to device.

        Idempotent: safe to call multiple times (e.g. once per layer when
        layers hold separate RotaryEmbedding instances). Called by the Engine
        at init so __call__ never needs to grow tables or cross devices.
        """
        self._ensure_cache(seq_len)
        if self._cos_table.device != device:
            self._cos_table = self._cos_table.to(device)
            self._sin_table = self._sin_table.to(device)

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Apply RoPE to q and k in-place.

        Args:
            q: query tensor (batch, num_heads, seq_len, head_dim)
            k: key tensor (batch, num_kv_heads, seq_len, head_dim)
            positions: position indices (seq_len,), on q's device
        """
        # Tables live on the compute device (one-time move here as a fallback
        # for non-Engine callers; Engine calls prebuild() at init). Combined
        # with on-device indexing below, this eliminates the per-layer,
        # per-step host-device syncs (positions.max().item() + .cpu()) that
        # used to run here — two syncs per layer per forward step — and keeps
        # the op CUDA-graph capturable.
        if self._cos_table.device != q.device:
            self._cos_table = self._cos_table.to(q.device)
            self._sin_table = self._sin_table.to(q.device)

        # Index directly with on-device positions. Out-of-range positions
        # fail loudly via torch's index error rather than being clamped.
        pos = positions.long()
        cos = self._cos_table[:, :, pos, :].to(dtype=q.dtype)
        sin = self._sin_table[:, :, pos, :].to(dtype=q.dtype)

        if cos.dim() == 4 and cos.shape[2] > 1 and q.shape[2] == 1:
            # Decode shape: q is (batch, heads, 1, head_dim) but positions is
            # (num_reqs,) — one position per request, not per seq slot. Indexing
            # produced (1, 1, num_reqs, head_dim); move the position axis to the
            # front so cos/sin broadcast as (num_reqs, 1, 1, head_dim).
            cos = cos.permute(2, 0, 1, 3)
            sin = sin.permute(2, 0, 1, 3)

        apply_rotary_emb(q, k, cos, sin)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> None:
    """Apply rotary embedding to q and k in-place.

    Args:
        q: shape (batch, num_heads, seq_len, head_dim)
        k: shape (batch, num_kv_heads, seq_len, head_dim)
        cos: shape (1, 1, seq_len, head_dim)
        sin: shape (1, 1, seq_len, head_dim)
    """
    q_rot = q.float()
    k_rot = k.float()

    # Rotate half the dimensions
    head_dim = q.shape[-1]
    q1, q2 = q_rot[..., : head_dim // 2], q_rot[..., head_dim // 2 :]
    k1, k2 = k_rot[..., : head_dim // 2], k_rot[..., head_dim // 2 :]

    cos_half = cos[..., : head_dim // 2]
    sin_half = sin[..., : head_dim // 2]

    q_rotated_1 = q1 * cos_half - q2 * sin_half
    q_rotated_2 = q2 * cos_half + q1 * sin_half
    k_rotated_1 = k1 * cos_half - k2 * sin_half
    k_rotated_2 = k2 * cos_half + k1 * sin_half

    q.copy_(torch.cat([q_rotated_1, q_rotated_2], dim=-1).to(q.dtype))
    k.copy_(torch.cat([k_rotated_1, k_rotated_2], dim=-1).to(k.dtype))
