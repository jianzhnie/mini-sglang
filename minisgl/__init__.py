"""Mini-SGLang: A lightweight educational LLM inference framework.

Usage:
    python -m minisgl --model-path Qwen/Qwen2-0.5B-Instruct --port 8000
    python -m minisgl --model-path Qwen/Qwen2-0.5B-Instruct --shell
"""

__all__ = ["parse_args", "run_server", "run_shell", "main"]
import argparse

from minisgl.config import ModelArgs, ServerArgs
from minisgl.engine.engine import Engine
from minisgl.engine.llm import LLM
from minisgl.models.tokenizer.worker import TokenizerWorker
from minisgl.scheduler.scheduler import Scheduler
from minisgl.utils.logger import setup_logger


def parse_args() -> ServerArgs:
    parser = argparse.ArgumentParser(
        description="Mini-SGLang: Lightweight LLM Inference"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to HF model"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor parallelism size"
    )
    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=0.9,
        help="Ratio of GPU memory for KV cache",
    )
    parser.add_argument("--max-running-req", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument(
        "--page-size",
        type=int,
        default=16,
        help="KV cache page size in tokens",
    )
    parser.add_argument(
        "--cuda-graph-bs",
        type=int,
        default=None,
        help="Max batch size for CUDA graph capture",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="fa",
        choices=["fa", "fi", "fa,fi"],
        help="Attention backend: fa (FlashAttention), fi (FlashInfer)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code for HF models",
    )
    parser.add_argument(
        "--shell", action="store_true", help="Interactive CLI shell mode"
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return ServerArgs(**vars(parser.parse_args()))


def run_server(args: ServerArgs) -> None:
    """Launch the HTTP server with scheduler in background thread."""
    import uvicorn

    from minisgl.server.frontend import app, init_frontend

    setup_logger(level=getattr(__import__("logging"), args.log_level))

    model_args = ModelArgs.from_pretrained(args.model_path)
    tokenizer = TokenizerWorker(args.model_path)

    engine = Engine(args, model_args, tp_rank=0)
    scheduler = Scheduler(args, engine)

    init_frontend(args, scheduler, tokenizer)

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


def run_shell(args: ServerArgs) -> None:
    """Interactive CLI shell for text generation."""
    setup_logger()

    llm = LLM(
        model_path=args.model_path,
        tp_size=args.tp_size,
        attention_backend=args.attention_backend,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        max_seq_len=args.max_seq_len,
        memory_ratio=args.memory_ratio,
    )

    print("\nMini-SGLang Interactive Shell")
    print(f"Model: {args.model_path}")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        messages = [{"role": "user", "content": user_input}]
        response = llm.chat(messages, temperature=0.7, max_tokens=2048)
        print(response)
        print()


def main() -> None:
    args = parse_args()

    if args.shell:
        run_shell(args)
    else:
        run_server(args)


if __name__ == "__main__":
    main()
