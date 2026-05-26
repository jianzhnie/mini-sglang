"""Tokenizer worker process: handles token encode/decode via ZMQ."""

__all__ = ["TokenizeMsg", "DetokenizeMsg", "UserMsg", "TokenizerWorker"]
from dataclasses import dataclass

from minisgl.utils.logger import logger


@dataclass
class TokenizeMsg:
    uid: int
    text: str


@dataclass
class DetokenizeMsg:
    uid: int
    token_id: int
    finished: bool


@dataclass
class UserMsg:
    uid: int
    input_ids: list[int]
    sampling_params: dict | None = None


class TokenizerWorker:
    """Wraps a HuggingFace tokenizer for encode/decode operations.

    In a multi-process architecture, this runs as a separate process
    communicating via ZMQ sockets. For simplicity, the in-process
    version is used by default.
    """

    def __init__(self, model_path: str):
        from transformers import AutoTokenizer

        logger.info(f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.eos_token_id = self.tokenizer.eos_token_id or 151643

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode(self, token_id: int, skip_special_tokens: bool = True) -> str:
        """Decode a single token ID to text."""
        return self.tokenizer.decode(
            [token_id],
            skip_special_tokens=skip_special_tokens,
        )

    def apply_chat_template(
        self, messages: list[dict], add_generation_prompt: bool = True
    ) -> str:
        """Apply the model's chat template, with fallback for models without one."""
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        # Fallback: for base models, use raw content as prompt
        parts: list[str] = []
        for msg in messages:
            content = msg.get("content", "")
            if content:
                parts.append(content)
        return "\n\n".join(parts)
