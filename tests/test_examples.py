#!/usr/bin/env python3
"""Smoke tests for the example scripts.

The CPU demo is self-contained (random-weight toy model) and always runs.
Model-dependent examples need a local HuggingFace model: provide one or more
paths via the MINISGL_TEST_MODELS environment variable (os.pathsep-separated);
without it those tests are skipped instead of failing.

Usage:
    python tests/test_examples.py
    MINISGL_TEST_MODELS=/path/to/Qwen2.5-0.5B python tests/test_examples.py
    MINISGL_TEST_MODELS=/m/opt-125m:/m/Qwen3-0.6B python tests/test_examples.py
"""

import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable

# Examples that require a real model via --model-path.
MODEL_EXAMPLES = [
    "examples/offline_inference.py",
    "examples/benchmark.py",
    "examples/server_demo.py",
]


def model_paths() -> list[str]:
    raw = os.environ.get("MINISGL_TEST_MODELS", "")
    return [p for p in raw.split(os.pathsep) if p]


def run_example(
    script: str, model_path: str | None, timeout: int = 300
) -> tuple[bool, str]:
    cmd = [PYTHON, script]
    if model_path:
        cmd += ["--model-path", model_path]
    env = {**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=ROOT,
            env=env,
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"


def check_output_quality(output: str) -> list[str]:
    issues = []
    lines = output.lower()
    if "error" in lines and "errno" not in lines:
        issues.append("Contains 'error'")
    if "traceback" in lines:
        issues.append("Contains traceback")
    if "nan" in lines and "token" not in lines:
        issues.append("Possible NaN")
    rep_phrases = ["is is is", "the the the", "a a a a"]
    for phrase in rep_phrases:
        if phrase in lines:
            issues.append(f"Repetitive output: '{phrase}'")
    return issues


class TestExamples(unittest.TestCase):
    def test_cpu_demo(self):
        """Self-contained CPU demo (no model download) must always pass."""
        ok, output = run_example("examples/cpu_demo.py", None, timeout=120)
        self.assertTrue(ok, msg=output[-2000:])
        self.assertEqual(check_output_quality(output), [])

    def test_model_examples(self):
        """Each model-dependent example runs against every configured model."""
        models = model_paths()
        if not models:
            self.skipTest(
                "no local model available; set MINISGL_TEST_MODELS to an "
                "os.pathsep-separated list of model paths to enable"
            )
        for script in MODEL_EXAMPLES:
            for model_path in models:
                with self.subTest(script=script, model=model_path):
                    ok, output = run_example(script, model_path, timeout=300)
                    self.assertTrue(ok, msg=output[-2000:])
                    self.assertEqual(check_output_quality(output), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
