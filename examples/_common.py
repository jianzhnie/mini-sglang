#!/usr/bin/env python3
"""Shared helpers for the executable example scripts.

Centralizes the boilerplate the demos used to repeat: model discovery/validation,
default ServerArgs, engine construction, the drive-until-idle loop, and banner
printing. Each example keeps its own *teaching* logic (streaming metrics, sampling
comparison, HTTP client, NPU results) and delegates only the scaffolding here.

Kept deliberately light: top-level imports are stdlib only, so ``import _common``
costs nothing and works from any directory. torch / transformers / minisgl are
imported lazily inside the functions that need them (the examples already follow
that style, and minisgl needs the repo root on sys.path, which the scripts add).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from minisgl.config import ServerArgs
    from minisgl.engine.engine import Engine

# Models each demo auto-detects when --model-path is omitted.
DEFAULT_MODEL_NAMES: tuple[str, ...] = (
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
)


def _candidate_roots(extra_roots: Sequence[str]) -> list[Path]:
    """Return the local roots searched for models, in priority order."""
    env_root = os.environ.get("MINISGL_MODELS", "")
    roots: list[Path] = []
    if env_root:  # os.pathsep-separated, like tests/test_examples.py sets it
        roots.extend(Path(p) for p in env_root.split(os.pathsep) if p)
    roots += [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / "hfhub" / "models",
    ]
    roots += [Path(r) for r in extra_roots]
    return roots


def find_model(*names: str, extra_roots: Sequence[str] = ()) -> str:
    """Locate a local HF model directory; return "" if none is found.

    ``names`` may be absolute paths, ``org/model`` identifiers (resolved under
    each root, and against the HF hub cache layout ``models--org--model``), or
    bare model names. Searching is non-fatal (used by npu's ``--models`` loop,
    which warns and skips misses).
    """
    roots = _candidate_roots(extra_roots)
    for name in names:
        candidates: list[Path] = [Path(name)]
        for root in roots:
            candidates.append(root / name)
            # HF hub cache layout: <root>/models--<org>--<model>.
            candidates.append(root / f"models--{name.replace('/', '--')}")
            # A bare model name may sit one level down, e.g. <root>/Qwen/<model>.
            candidates.append(root / name.replace("--", "/"))
        for p in candidates:
            if p.is_dir() and (p / "config.json").exists():
                return str(p)
    return ""


def resolve_model_path(
    explicit: str | None,
    *names: str,
    extra_roots: Sequence[str] = (),
) -> str:
    """Resolve the model to run: an explicit path wins, else auto-detect.

    Fails loudly (usage + exit 1) when nothing usable is found, so callers in
    ``if __name__ == "__main__"`` need no further validation.
    """
    path = explicit or find_model(*names, extra_roots=extra_roots)
    if not path or not (Path(path) / "config.json").exists():
        print(f"ERROR: Model not found at: {path!r}")
        print(f"  python {Path(sys.argv[0]).name} --model-path /path/to/hf_model")
        sys.exit(1)
    return path


def cli_main(
    description: str,
    run,
    *names: str,
    extra_roots: Sequence[str] = (),
) -> None:
    """Run an example's main from the CLI.

    Owns the argparse + --model-path resolution that every model example
    repeats, so each script's ``if __name__ == "__main__"`` is one line::

        cli_main("Mini-SGLang Offline Inference", main)
    """
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()
    names = names or DEFAULT_MODEL_NAMES
    run(resolve_model_path(args.model_path, *names, extra_roots=extra_roots))


def default_server_args(
    model_path: str,
    *,
    max_running_req: int = 8,
    max_seq_len: int = 256,
    page_size: int = 16,
    memory_ratio: float = 0.5,
    attention_backend: str = "fa",
    cuda_graph_bs: int | None = 0,
    **overrides: object,
) -> ServerArgs:
    """Build a teaching-friendly ServerArgs; keyword args override defaults."""
    from minisgl.config import ServerArgs

    kwargs: dict[str, object] = {
        "model_path": model_path,
        "tp_size": 1,
        "attention_backend": attention_backend,
        "max_running_req": max_running_req,
        "max_seq_len": max_seq_len,
        "page_size": page_size,
        "memory_ratio": memory_ratio,
        "cuda_graph_bs": cuda_graph_bs,
    }
    kwargs.update(overrides)
    return ServerArgs(**kwargs)


def build_engine(model_path: str, **overrides: object) -> tuple[ServerArgs, Engine]:
    """Create ServerArgs + a single-GPU Engine for a model path."""
    from minisgl.config import ModelArgs
    from minisgl.engine.engine import Engine

    args = default_server_args(model_path, **overrides)
    model_args = ModelArgs.from_pretrained(model_path)
    return args, Engine(args, model_args, tp_rank=0)


def load_tokenizer(model_path: str, *, trust_remote_code: bool = False):
    """Load the HF AutoTokenizer for a model (lazy import)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )


def drive(scheduler):
    """Yield every OutputToken until the scheduler is idle.

    This is the shared ``while not idle: for out in step()`` scaffold. What each
    demo does with a token (collect by uid, time to first token, filter aborts,
    print incrementally) stays in the caller — that is the teaching part.
    """
    while not scheduler.is_idle():
        for out in scheduler.step():
            yield out


def banner(title: str, *, lines: Sequence[str] = (), width: int = 60) -> None:
    """Print a centered '=' banner, optionally with extra info lines."""
    bar = "=" * width
    print(bar)
    print(f"  {title}")
    for line in lines:
        if line:  # skip blank filler lines
            print(f"  {line}")
    print(bar)


def section(title: str) -> None:
    """Print a '── ... ──' sub-heading."""
    print(f"\n  {'─' * 5} {title} {'─' * 5}")
