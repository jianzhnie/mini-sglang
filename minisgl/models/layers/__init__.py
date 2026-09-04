"""Shared model layers: embedding, linear, RMSNorm, and RoPE.

Each class lives in its own module under ``minisgl/models/layers/`` and is
imported by the decoder model code (``minisgl/models/base.py``) and by tests
via the concrete path (e.g. ``from minisgl.models.layers.rms_norm import RMSNorm``).
"""
