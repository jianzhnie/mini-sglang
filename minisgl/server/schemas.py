"""Pydantic request/response models for the OpenAI-compatible HTTP API."""

from __future__ import annotations

__all__ = ["ChatCompletionRequest", "ChatMessage", "CompletionRequest"]

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """A chat message with role and content."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = "default"
    messages: list[ChatMessage]
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 1024
    stream: bool = False
    ignore_eos: bool = False


class CompletionRequest(BaseModel):
    """OpenAI-compatible text completion request."""

    model: str = "default"
    prompt: str
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 1024
    stream: bool = False
    ignore_eos: bool = False
