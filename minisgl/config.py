"""Configuration dataclasses for mini-sglang."""

__all__ = ["ModelArgs", "SamplingParams", "ServerArgs"]
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SamplingParams:
    """Sampling parameters for text generation."""

    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024


@dataclass
class ServerArgs:
    """Server and engine configuration parsed from CLI / env."""

    model_path: str = ""
    host: str = "127.0.0.1"
    port: int = 8000
    tp_size: int = 1
    memory_ratio: float = 0.9
    max_running_req: int = 256
    max_seq_len: int = 8192
    page_size: int = 16
    cuda_graph_bs: int | None = None
    attention_backend: str = "fa"
    dtype: str = "auto"
    device: str = "auto"
    trust_remote_code: bool = False
    shell: bool = False

    def __post_init__(self) -> None:
        if self.cuda_graph_bs is None:
            self.cuda_graph_bs = self.max_running_req
        if self.device != "auto":
            import torch

            from minisgl.utils.device import set_device

            set_device(torch.device(self.device))


@dataclass
class ModelArgs:
    """Model architecture parameters loaded from HuggingFace config.json."""

    hidden_size: int = 0
    num_layers: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    use_sliding_window: bool = False
    sliding_window: int | None = None
    rope_scaling: dict | None = None
    # MoE
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    decoder_sparse_step: int = 1
    # Qwen3 specific
    qk_norm: bool = False
    # Head dim
    head_dim: int = 0

    @classmethod
    def from_pretrained(cls, model_path: str) -> "ModelArgs":
        config_file = Path(model_path) / "config.json"
        with config_file.open() as f:
            cfg = json.load(f)

        num_kv_heads = cfg.get("num_key_value_heads", cfg.get("num_attention_heads", 0))
        hidden_size = cfg["hidden_size"]
        num_heads = cfg.get("num_attention_heads", 0)
        head_dim = cfg.get("head_dim", hidden_size // num_heads if num_heads else 0)

        return cls(
            hidden_size=hidden_size,
            num_layers=cfg.get("num_hidden_layers", 0),
            num_attention_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=cfg.get("intermediate_size", 0) or cfg.get("ffn_dim", 0),
            vocab_size=cfg.get("vocab_size", 0),
            max_position_embeddings=cfg.get("max_position_embeddings", 8192),
            rope_theta=cfg.get("rope_theta", 10000.0),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
            use_sliding_window=cfg.get("use_sliding_window", False),
            sliding_window=cfg.get("sliding_window", None),
            rope_scaling=cfg.get("rope_scaling", None),
            num_experts=cfg.get("num_experts", 0),
            num_experts_per_tok=cfg.get("num_experts_per_tok", 0),
            moe_intermediate_size=cfg.get("moe_intermediate_size", 0),
            decoder_sparse_step=cfg.get("decoder_sparse_step", 1),
            qk_norm=cfg.get("qk_norm", False),
            head_dim=head_dim,
        )
