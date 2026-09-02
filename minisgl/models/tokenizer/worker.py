"""Tokenizer worker: wraps a HuggingFace tokenizer for encode/decode.

In-process implementation (SGLang runs this as a separate ZMQ process; the
teaching version keeps everything in one process).
"""

__all__ = ["TokenizerWorker"]

from minisgl.utils.logger import logger


class TokenizerWorker:
    """Wraps a HuggingFace tokenizer for encode/decode operations."""

    def __init__(self, model_path: str, trust_remote_code: bool = False) -> None:
        from transformers import AutoTokenizer

        logger.info("Loading tokenizer from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode(
        self, token_id: int | list[int], skip_special_tokens: bool = True
    ) -> str:
        """Decode token ID(s) to text.

        Accepts a single token ID (int) for streaming or a list of token IDs
        for batch/non-streaming output.
        """
        if isinstance(token_id, list):
            return self.tokenizer.decode(
                token_id,
                skip_special_tokens=skip_special_tokens,
            )
        return self.tokenizer.decode(
            [token_id],
            skip_special_tokens=skip_special_tokens,
        )

    def apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
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
