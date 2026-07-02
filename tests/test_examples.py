#!/usr/bin/env python3
"""Run all examples against multiple models and report results.

Usage:
    python tests/test_examples.py
"""

import os
import subprocess
import sys
import time

PYTHON = "/home/jianzhnie/llmtuner/software/miniconda3/envs/llm_minimind/bin/python"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    ("/home/jianzhnie/llmtuner/hfhub/models/facebook/opt-125m", "OPT-125M"),
    ("/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
    ("/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-0.6B", "Qwen3-0.6B"),
]

EXAMPLES = [
    ("examples/cpu_demo.py", False),
    ("examples.py", True),
    ("examples/batch_inference.py", True),
    ("examples/llm_generate.py", True),
]


def run_example(script: str, model_path: str | None, timeout: int = 300) -> tuple[bool, str, float]:
    cmd = [PYTHON, script]
    if model_path:
        cmd += ["--model-path", model_path]
    env = {**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=ROOT, env=env,
        )
        elapsed = time.time() - t0
        output = result.stdout + result.stderr
        ok = result.returncode == 0
        return ok, output, elapsed
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT", time.time() - t0


def check_output_quality(output: str, model_name: str) -> list[str]:
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


def main():
    print("=" * 70)
    print("  Mini-SGLang Example Test Suite — Multi-Model")
    print("=" * 70)

    results = []

    for script, needs_model in EXAMPLES:
        if not needs_model:
            print(f"\n{'─' * 70}")
            print(f"  {script} (no model)")
            ok, output, elapsed = run_example(script, None, timeout=60)
            issues = check_output_quality(output, "none") if ok else ["FAILED"]
            status = "PASS" if ok and not issues else "FAIL"
            results.append((script, "N/A", status, elapsed, issues))
            print(f"  {status} ({elapsed:.1f}s)")
            if issues:
                print(f"  Issues: {issues}")
            continue

        for model_path, model_name in MODELS:
            print(f"\n{'─' * 70}")
            print(f"  {script} + {model_name}")
            ok, output, elapsed = run_example(script, model_path, timeout=300)
            issues = check_output_quality(output, model_name) if ok else ["FAILED"]
            status = "PASS" if ok and not issues else "FAIL"
            results.append((script, model_name, status, elapsed, issues))
            print(f"  {status} ({elapsed:.1f}s)")
            if issues:
                print(f"  Issues: {issues}")
            # Print key output lines
            for line in output.split("\n"):
                line = line.strip()
                if line.startswith("Output:") or line.startswith("Prompt:") or "→" in line:
                    print(f"    {line}")

    print(f"\n{'=' * 70}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 70}")
    total = len(results)
    passed = sum(1 for r in results if r[2] == "PASS")
    for script, model, status, elapsed, issues in results:
        flag = "OK" if status == "PASS" else "!!"
        issue_str = f"  ({', '.join(issues)})" if issues else ""
        print(f"  [{flag}] {script:35s} {model:15s} {elapsed:6.1f}s{issue_str}")
    print(f"\n  {passed}/{total} passed")
    print("=" * 70)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
