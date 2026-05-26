"""High-level LLM API for offline (non-server) inference.

Usage:
    from minisgl import LLM
    llm = LLM(model_path="Qwen/Qwen2-0.5B-Instruct")
    output = llm.generate(["Hello, world!", "What is AI?"])
"""

__all__ = ["LLM"]
from minisgl.config import ModelArgs, SamplingParams, ServerArgs
from minisgl.engine.engine import Engine
from minisgl.models.tokenizer.worker import TokenizerWorker
from minisgl.scheduler.scheduler import Scheduler
from minisgl.utils.logger import logger


class LLM:
    """High-level API for offline LLM inference."""

    def __init__(
        self,
        model_path: str,
        tp_size: int = 1,
        attention_backend: str = "fa",
        dtype: str = "auto",
        trust_remote_code: bool = True,
        max_seq_len: int = 8192,
        memory_ratio: float = 0.9,
    ) -> None:
        self.model_path = model_path

        server_args = ServerArgs(
            model_path=model_path,
            tp_size=tp_size,
            attention_backend=attention_backend,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            max_seq_len=max_seq_len,
            memory_ratio=memory_ratio,
        )

        model_args = ModelArgs.from_pretrained(model_path)

        logger.info(f"Loading model from {model_path}")
        self.engine = Engine(server_args, model_args, tp_rank=0)
        self.scheduler = Scheduler(server_args, self.engine)
        self.tokenizer = TokenizerWorker(model_path)

        logger.info("LLM ready")

    def generate(
        self,
        prompts: str | list[str],
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = 1024,
    ) -> str | list[str]:
        """Generate text from prompts.

        Args:
            prompts: Single prompt string or list of prompt strings.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling threshold.
            top_k: Top-k filtering threshold.
            max_tokens: Maximum tokens to generate per prompt.

        Returns:
            Generated text or list of generated texts.
        """
        single_input = isinstance(prompts, str)
        prompt_list = [prompts] if single_input else prompts

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )

        results: list[str] = []
        uid_to_idx = {}

        # Submit all requests
        for i, prompt in enumerate(prompt_list):
            input_ids = self.tokenizer.encode(prompt)
            uid = self.scheduler.add_request(input_ids, sampling_params)
            uid_to_idx[uid] = i
            results.append("")

        # Run inference loop
        pending_uids = set(uid_to_idx.keys())
        while pending_uids:
            step_results = self.scheduler.step()
            for uid, token_id, finished in step_results:
                if uid in pending_uids:
                    idx = uid_to_idx[uid]
                    text = self.tokenizer.decode(token_id)
                    results[idx] += text
                    if finished:
                        pending_uids.discard(uid)

        return results[0] if single_input else results

    def chat(
        self,
        messages: list[dict] | list[list[dict]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
    ) -> str | list[str]:
        """Chat completion interface.

        Args:
            messages: A list of message dicts (role/content) or list of message lists.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            max_tokens: Maximum tokens to generate.

        Returns:
            Assistant response text or list of response texts.
        """
        single_input = (
            isinstance(messages, list) and messages and isinstance(messages[0], dict)
        )
        msg_list = [messages] if single_input else messages

        prompts = [self.tokenizer.apply_chat_template(msgs) for msgs in msg_list]

        return self.generate(
            prompts,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
