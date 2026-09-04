"""Shared server-launch path.

Wires an Engine + Scheduler + tokenizer + FastAPI frontend from a
:class:`~minisgl.config.ServerArgs` and runs uvicorn. Used by both the
``python -m minisgl`` CLI (``cli.run_server``) and the standalone
``examples/server_demo.py`` so the assembly is defined once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minisgl.config import ServerArgs


def serve(args: ServerArgs) -> None:
    """Start the HTTP server for ``args`` (blocking until it shuts down)."""
    import logging

    import uvicorn

    from minisgl.config import ModelArgs
    from minisgl.engine.engine import Engine
    from minisgl.scheduler.scheduler import Scheduler
    from minisgl.server.api import app, init_frontend
    from minisgl.tokenizer import TokenizerWorker
    from minisgl.utils.logger import setup_logger

    log_level = args.log_level
    setup_logger(level=getattr(logging, log_level))

    model_args = ModelArgs.from_pretrained(args.model_path)
    tokenizer = TokenizerWorker(
        args.model_path, trust_remote_code=args.trust_remote_code
    )

    engine = Engine(args, model_args, tp_rank=0)
    scheduler = Scheduler(args, engine)

    init_frontend(args, scheduler, tokenizer)

    uvicorn.run(app, host=args.host, port=args.port, log_level=log_level.lower())
