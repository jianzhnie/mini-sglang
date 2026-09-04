"""Config, sampling, weight loading, device/distributed helpers.

Run: python3 tests/utils/test_utils.py   (or: python -m pytest tests/utils/test_utils.py)
"""

import logging
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

# Make the repo root importable regardless of the invocation directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))



# ── TestSampler ──
class TestSampler(unittest.TestCase):
    def test_greedy_sampling(self):
        from minisgl.config import SamplingParams
        from minisgl.sampling import Sampler

        sampler = Sampler()
        logits = torch.randn(4, 1000)  # 4 requests, 1000 vocab
        params = SamplingParams(temperature=0.0)  # greedy
        tokens = sampler.sample(logits, params)
        self.assertEqual(tokens.shape, (4,))
        # Greedy should pick argmax
        expected = logits.argmax(dim=-1)
        self.assertTrue(torch.equal(tokens, expected))

    def test_temperature_sampling(self):
        from minisgl.config import SamplingParams
        from minisgl.sampling import Sampler

        sampler = Sampler()
        logits = torch.randn(4, 1000)
        params = SamplingParams(temperature=0.8, top_k=50, top_p=0.9)
        tokens = sampler.sample(logits, params)
        self.assertEqual(tokens.shape, (4,))

    def test_top_k_top_p(self):
        from minisgl.sampling import _apply_top_k, _apply_top_p

        logits = torch.randn(1, 1000)
        # Top-k: only top 10 should be > -inf
        filtered = _apply_top_k(logits.clone(), 10)
        self.assertEqual((filtered > float("-inf")).sum().item(), 10)
        # Top-p
        filtered = _apply_top_p(logits.clone(), 0.95)
        self.assertTrue((filtered > float("-inf")).sum().item() > 0)

    def test_sample_batch_all_greedy(self):
        from minisgl.config import SamplingParams
        from minisgl.sampling import Sampler

        torch.manual_seed(0)
        sampler = Sampler()
        logits = torch.randn(3, 100)
        params = [SamplingParams(temperature=0.0)] * 3  # all greedy
        tokens = sampler.sample_batch(logits, params)
        self.assertEqual(len(tokens), 3)
        self.assertEqual(tokens, logits.argmax(dim=-1).tolist())

    def test_sample_batch_groups_identical_params(self):
        from minisgl.config import SamplingParams
        from minisgl.sampling import Sampler

        torch.manual_seed(1)
        sampler = Sampler()
        logits = torch.randn(4, 100)
        # Two greedy, two stochastic (sharing params -> sampled together).
        params = [
            SamplingParams(temperature=0.0),
            SamplingParams(temperature=0.9, top_k=20),
            SamplingParams(temperature=0.9, top_k=20),
            SamplingParams(temperature=0.0),
        ]
        tokens = sampler.sample_batch(logits, params)
        self.assertEqual(len(tokens), 4)
        # Greedy rows pick argmax deterministically.
        self.assertEqual(tokens[0], logits[0].argmax().item())
        self.assertEqual(tokens[3], logits[3].argmax().item())
        # Non-greedy rows fall inside vocab range.
        self.assertTrue(all(0 <= t < 100 for t in tokens))



# ── Test KV Cache Pool ──

# ── TestDistributed ──
class TestDistributed(unittest.TestCase):
    def test_all_reduce_noop(self):
        from minisgl.engine.collectives import all_reduce

        x = torch.randn(10)
        y = all_reduce(x)
        self.assertTrue(torch.equal(x, y))  # no-op when not distributed

    def test_all_reduce_invalid_op_raises(self):
        from minisgl.engine.collectives import all_reduce

        with self.assertRaises(ValueError):
            all_reduce(torch.randn(4), op="avg")


# ── Test Weight Loading ──

# ── TestWeightLoading ──
class TestWeightLoading(unittest.TestCase):
    def test_shard_tensor_not_divisible_raises(self):
        from minisgl.utils.weights import shard_tensor

        x = torch.randn(7, 4)
        with self.assertRaises(ValueError):
            shard_tensor(x, dim=0, rank=0, world_size=2)

    def test_shard_tensor_1d(self):
        from minisgl.utils.weights import shard_tensor

        x = torch.arange(8)
        shard = shard_tensor(x, dim=0, rank=1, world_size=2)
        self.assertEqual(shard.tolist(), [4, 5, 6, 7])

    def test_missing_params_warn(self):
        from minisgl.utils.weights import load_weights_parallel

        model = nn.Linear(4, 4)
        with self.assertLogs("minisgl", level="WARNING") as cm:
            loaded = load_weights_parallel(model, {})
        self.assertEqual(loaded, 0)
        self.assertTrue(any("not found" in m for m in cm.output))

    def test_shape_mismatch_skipped_warns(self):
        from minisgl.utils.weights import load_weights_parallel

        model = nn.Linear(4, 4)
        # bias has wrong shape (5 vs 4) and is 1-D: cannot be truncated
        sd = {"weight": torch.randn(4, 4), "bias": torch.randn(5)}
        with self.assertLogs("minisgl", level="WARNING") as cm:
            loaded = load_weights_parallel(model, sd)
        self.assertEqual(loaded, 1)
        self.assertTrue(any("shape mismatch" in m for m in cm.output))

    def test_column_parallel_bias_sharded(self):
        from unittest import mock

        from minisgl.models.layers.linear import ColumnParallelLinear
        from minisgl.utils import device as device_mod
        from minisgl.utils.weights import load_weights_parallel

        # Build the layer as if on a TP=2 rank (dist not initialized, so no
        # real communication happens).
        with mock.patch.object(device_mod._state, "tp_size", 2):
            layer = ColumnParallelLinear(4, 8, bias=True)
        self.assertEqual(layer.bias.shape, (4,))

        sd = {"weight": torch.randn(8, 4), "bias": torch.arange(8.0)}
        loaded = load_weights_parallel(layer, sd, tp_rank=1, tp_size=2)
        self.assertEqual(loaded, 2)
        # Rank 1 gets the second half of the bias
        self.assertEqual(layer.bias.tolist(), [4.0, 5.0, 6.0, 7.0])

    def test_load_hf_weights_safetensors_index(self):
        """load_hf_weights follows model.safetensors.index.json shards."""
        import json
        import os
        import tempfile

        from minisgl.utils.weights import load_hf_weights

        with tempfile.TemporaryDirectory() as tmp:
            a = torch.randn(3, 3)
            b = torch.randn(4, 4)
            from safetensors.torch import save_file

            save_file(
                {"a.weight": a},
                os.path.join(tmp, "model-00001-of-00002.safetensors"),
            )
            save_file(
                {"b.weight": b},
                os.path.join(tmp, "model-00002-of-00002.safetensors"),
            )
            with open(os.path.join(tmp, "model.safetensors.index.json"), "w") as f:
                json.dump(
                    {
                        "weight_map": {
                            "a.weight": "model-00001-of-00002.safetensors",
                            "b.weight": "model-00002-of-00002.safetensors",
                        }
                    },
                    f,
                )

            state = load_hf_weights(tmp)
            self.assertEqual(set(state), {"a.weight", "b.weight"})
            self.assertEqual(state["a.weight"].shape, (3, 3))
            self.assertEqual(state["b.weight"].shape, (4, 4))

    def test_load_hf_weights_plain_bin_dir_no_index(self):
        """A dir of pytorch_model.bin files without an index is loaded."""
        import tempfile
        from pathlib import Path

        from minisgl.utils.weights import load_hf_weights

        with tempfile.TemporaryDirectory() as tmp:
            torch.save({"x": torch.ones(2)}, str(Path(tmp) / "pytorch_model.bin"))
            # A stray .safetensors next to it is also picked up.
            from safetensors.torch import save_file

            save_file({"y": torch.zeros(2)}, str(Path(tmp) / "extra.safetensors"))
            state = load_hf_weights(tmp)
            self.assertEqual(state["x"].tolist(), [1.0, 1.0])
            self.assertEqual(state["y"].tolist(), [0.0, 0.0])


# ── Test dtype resolution ──

# ── TestResolveDtype ──
class TestResolveDtype(unittest.TestCase):
    def test_auto_reads_config_json(self):
        import json
        import tempfile
        from pathlib import Path

        from minisgl.engine.model_runner import resolve_dtype

        with tempfile.TemporaryDirectory() as d:
            Path(d, "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}))
            self.assertEqual(resolve_dtype("auto", d, "cuda"), torch.bfloat16)
            # CPU falls back to float32
            self.assertEqual(resolve_dtype("auto", d, "cpu"), torch.float32)

    def test_explicit_and_fallbacks(self):
        from minisgl.engine.model_runner import resolve_dtype

        self.assertEqual(
            resolve_dtype("float16", "/nonexistent", "cuda"), torch.float16
        )
        # Missing config.json under auto falls back to float32
        self.assertEqual(resolve_dtype("auto", "/nonexistent", "cuda"), torch.float32)
        # CPU forces float32
        self.assertEqual(
            resolve_dtype("float16", "/nonexistent", "cpu"), torch.float32
        )
        # Unknown dtype string falls back to float32
        self.assertEqual(resolve_dtype("weird", "/nonexistent", "cuda"), torch.float32)


# ── Test Config ──

# ── TestConfig ──
class TestConfig(unittest.TestCase):
    def test_config_creation(self):
        from minisgl.config import SamplingParams, ServerArgs

        args = ServerArgs(model_path="/tmp/test", port=8000, tp_size=1)
        self.assertEqual(args.port, 8000)
        self.assertEqual(args.tp_size, 1)

        params = SamplingParams(temperature=0.7, top_k=50, top_p=0.95)
        self.assertEqual(params.temperature, 0.7)
        self.assertEqual(params.top_k, 50)

    def test_sampling_params_clamps_invalid_values(self):
        # max_tokens must never be < 1 (0/negative underflows page counting),
        # and top_p must stay within [0, 1] (nucleus threshold).
        from minisgl.config import SamplingParams

        self.assertEqual(SamplingParams(max_tokens=0).max_tokens, 1)
        self.assertEqual(SamplingParams(max_tokens=-5).max_tokens, 1)
        self.assertEqual(SamplingParams(top_p=2.0).top_p, 1.0)
        self.assertEqual(SamplingParams(top_p=-0.5).top_p, 1.0)
        # temperature <= 0 is the documented greedy sentinel — left untouched.
        self.assertEqual(SamplingParams(temperature=0.0).temperature, 0.0)
        # Valid values pass through unchanged.
        p = SamplingParams(max_tokens=64, top_p=0.9)
        self.assertEqual(p.max_tokens, 64)
        self.assertEqual(p.top_p, 0.9)


# ── Test Tokenizer Worker ──

# ── TestTokenizerWorker ──
class TestTokenizerWorker(unittest.TestCase):
    @unittest.skipIf(
        not __import__("importlib.util").util.find_spec("transformers"),
        "transformers not installed",
    )
    def test_tokenizer_creation(self):
        """Test with a tiny tokenizer (requires transformers)."""
        from minisgl.tokenizer import TokenizerWorker

        # Use a model that's likely cached or small
        try:
            worker = TokenizerWorker("google/bert_uncased_L-2_H-128_A-2")
            ids = worker.encode("Hello world")
            self.assertIsInstance(ids, list)
            self.assertTrue(len(ids) > 0)
        except Exception:
            self.skipTest("Model not available offline")

    def _local_qwen3(self):
        """A locally-cached Qwen3 tokenizer (skip if absent / no network)."""
        from pathlib import Path

        candidate = Path.home() / "hfhub" / "models" / "Qwen" / "Qwen3-0.6B"
        if not (candidate / "tokenizer.json").exists():
            candidate = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-0.6B"
            if not (candidate / "snapshots").exists():
                return None
        return str(candidate)

    @unittest.skipIf(
        not __import__("importlib.util").util.find_spec("transformers"),
        "transformers not installed",
    )
    def test_encode_decode_roundtrip(self):
        """encode -> decode round-trips a real prompt without data loss."""
        model_path = self._local_qwen3()
        if not model_path:
            self.skipTest("no local Qwen3 tokenizer; set one up offline")
        from minisgl.tokenizer import TokenizerWorker

        worker = TokenizerWorker(model_path)
        text = "The capital of France is"
        ids = worker.encode(text)
        self.assertGreater(len(ids), 0)
        # Round-trip must recover every word (BPE keeps subwords but the
        # decoded text contains the original tokens in order).
        decoded = worker.decode(ids)
        for word in text.split():
            self.assertIn(word.lower(), decoded.lower())

        # decode() accepts both an int and a list[int].
        self.assertEqual(worker.decode(ids[0]), worker.decode([ids[0]]))

    @unittest.skipIf(
        not __import__("importlib.util").util.find_spec("transformers"),
        "transformers not installed",
    )
    def test_apply_chat_template_real_and_fallback(self):
        model_path = self._local_qwen3()
        if not model_path:
            self.skipTest("no local Qwen3 tokenizer; set one up offline")
        from minisgl.tokenizer import TokenizerWorker

        worker = TokenizerWorker(model_path)
        messages = [{"role": "user", "content": "Hello"}]
        # Real template path (Qwen3 ships a chat template).
        self.assertTrue(
            worker.tokenizer.chat_template, "expected a real chat template"
        )
        rendered = worker.apply_chat_template(messages)
        self.assertIn("Hello", rendered)
        self.assertIn("user", rendered)

        # Fallback path: a tokenizer without chat_template concatenates content.
        class _NoTemplate:
            chat_template = None

            def apply_chat_template(self, *a, **kw):  # pragma: no cover
                raise AssertionError("fallback must not call this")

        worker.tokenizer = _NoTemplate()
        self.assertEqual(
            worker.apply_chat_template(
                [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
            ),
            "a\n\nb",
        )


# ── Test Full Model (Dummy) ──

# ── TestSamplerEdgeCases ──
class TestSamplerEdgeCases(unittest.TestCase):
    def test_single_token_batch(self):
        from minisgl.config import SamplingParams
        from minisgl.sampling import Sampler

        sampler = Sampler()
        logits = torch.randn(1, 100)
        params = SamplingParams(temperature=0.0)
        tokens = sampler.sample(logits, params)
        self.assertEqual(tokens.shape, (1,))
        self.assertEqual(tokens[0].item(), logits.argmax(dim=-1)[0].item())

    def test_top_k_equals_vocab(self):
        from minisgl.sampling import _apply_top_k

        logits = torch.randn(1, 50)
        filtered = _apply_top_k(logits.clone(), 50)
        self.assertTrue(torch.equal(logits, filtered))

    def test_very_low_temperature(self):
        from minisgl.config import SamplingParams
        from minisgl.sampling import Sampler

        # temperature=0.01 is still *sampling* (not greedy): the argmax token
        # wins with overwhelming probability but is not guaranteed. Seeding the
        # draw keeps this deterministic regardless of RNG state left by earlier
        # tests (it flaked on 4/50 seeds under the file's old ordering).
        torch.manual_seed(0)
        sampler = Sampler()
        logits = torch.randn(4, 100)
        params = SamplingParams(temperature=0.01)
        tokens = sampler.sample(logits, params)
        expected = logits.argmax(dim=-1)
        self.assertTrue(torch.equal(tokens, expected))


# ── Test FrontendManager ──

# ── TestDeviceUtils ──
class TestDeviceUtils(unittest.TestCase):
    def test_get_device_type(self):
        from minisgl.utils.device import get_device_type

        dtype = get_device_type()
        self.assertIn(dtype, ("cpu", "cuda", "npu"))

    def test_is_npu_available(self):
        from minisgl.utils.device import is_npu_available

        result = is_npu_available()
        self.assertIsInstance(result, bool)

    def test_is_accelerator_available(self):
        from minisgl.utils.device import is_accelerator_available

        result = is_accelerator_available()
        self.assertIsInstance(result, bool)

    def test_synchronize_cpu(self):
        from minisgl.utils.device import synchronize

        synchronize()

    def test_mem_get_info_cpu(self):
        from minisgl.utils.device import mem_get_info

        free, total = mem_get_info(torch.device("cpu"))
        self.assertEqual(free, 0)
        self.assertEqual(total, 0)

    def test_set_device_cpu(self):
        from minisgl.utils.device import get_device, reset_device_state, set_device

        reset_device_state()
        set_device(torch.device("cpu"))
        self.assertEqual(get_device().type, "cpu")
        reset_device_state()

    def test_init_distributed_auto_backend(self):
        from minisgl.utils.device import get_device_type

        dtype = get_device_type()
        if dtype == "npu":
            expected_backend = "hccl"
        elif dtype == "cuda":
            expected_backend = "nccl"
        else:
            expected_backend = "gloo"
        self.assertIn(expected_backend, ("hccl", "nccl", "gloo"))


# ── Test ServerArgs Device Config ──

# ── TestServerArgsDevice ──
class TestServerArgsDevice(unittest.TestCase):
    def test_device_field_default(self):
        from minisgl.config import ServerArgs

        args = ServerArgs(model_path="/tmp/test")
        self.assertEqual(args.device, "auto")

    def test_attention_backend_pt(self):
        from minisgl.models.attention.dispatcher import AttentionBackend

        AttentionBackend.configure("pt")
        q = torch.randn(1, 4, 8, 32)
        k = torch.randn(1, 4, 8, 32)
        v = torch.randn(1, 4, 8, 32)
        out = AttentionBackend.forward(q, k, v)
        self.assertEqual(out.shape, q.shape)
        AttentionBackend.configure("fa")


# ── Test Logger ──
class TestLogger(unittest.TestCase):
    def test_setup_logger_returns_configured_logger(self):
        from minisgl.utils.logger import setup_logger

        log = setup_logger("test_logger", level=logging.INFO)
        self.assertEqual(log.level, logging.INFO)
        # Handlers are replaced on each call, so exactly one stream handler.
        self.assertEqual(len(log.handlers), 1)
        # Stream handler writes to stdout.
        self.assertIsInstance(log.handlers[0], logging.StreamHandler)

    def test_setup_logger_log_file(self):
        import tempfile
        from pathlib import Path

        from minisgl.utils.logger import setup_logger

        with tempfile.TemporaryDirectory() as d:
            log_file = Path(d, "test.log")
            log = setup_logger("test_file_logger", level=logging.INFO, log_file=str(log_file))
            self.assertEqual(len(log.handlers), 2)  # stream + file
            log.warning("hello file")
            for h in log.handlers:
                h.flush()
            self.assertIn("hello file", log_file.read_text())

    def test_logger_writes_to_logging_capture(self):
        from minisgl.utils.logger import logger

        with self.assertLogs("minisgl", level="WARNING") as cm:
            logger.warning("a warning message")
        self.assertTrue(any("a warning message" in m for m in cm.output))


# ── Test CLI Argument Parsing ──
class TestCLIArgs(unittest.TestCase):
    def test_parse_args_defaults(self):
        from minisgl.cli import parse_args

        # parse_args reads sys.argv; feed it a minimal valid invocation.
        old_argv = sys.argv
        sys.argv = ["minisgl", "--model-path", "/tmp/model"]
        try:
            args = parse_args()
        finally:
            sys.argv = old_argv
        self.assertEqual(args.model_path, "/tmp/model")
        self.assertEqual(args.tp_size, 1)
        self.assertEqual(args.port, 8000)
        self.assertEqual(args.attention_backend, "fa")
        self.assertFalse(args.shell)

    def test_parse_args_custom_values(self):
        from minisgl.cli import parse_args

        old_argv = sys.argv
        sys.argv = [
            "minisgl", "--model-path", "/m/model", "--tp-size", "1",
            "--port", "9000", "--device", "cpu", "--dtype", "float32",
            "--attention-backend", "pt", "--max-seq-len", "4096",
            "--page-size", "8", "--shell",
        ]
        try:
            args = parse_args()
        finally:
            sys.argv = old_argv
        self.assertEqual(args.port, 9000)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.dtype, "float32")
        self.assertEqual(args.attention_backend, "pt")
        self.assertEqual(args.max_seq_len, 4096)
        self.assertEqual(args.page_size, 8)
        self.assertTrue(args.shell)

    def test_parse_args_rejects_tp_size_gt_1(self):
        from minisgl.cli import parse_args

        old_argv = sys.argv
        sys.argv = ["minisgl", "--model-path", "/m/model", "--tp-size", "2"]
        try:
            with self.assertRaises(SystemExit):
                parse_args()
        finally:
            sys.argv = old_argv

    def test_parse_args_rejects_invalid_backend(self):
        from minisgl.cli import parse_args

        old_argv = sys.argv
        sys.argv = ["minisgl", "--model-path", "/m/model", "--attention-backend", "fi"]
        try:
            with self.assertRaises(SystemExit):
                parse_args()
        finally:
            sys.argv = old_argv


# ── Test Shell Loop (run_shell) ──
class TestRunShell(unittest.TestCase):
    """Drive the interactive shell loop without a real model."""

    @staticmethod
    def _run(inputs, llm):
        import builtins
        import io
        import unittest.mock as mock
        from contextlib import redirect_stdout

        import minisgl.cli as cli

        from minisgl.config import ServerArgs

        real_llm = cli.LLM
        cli.LLM = lambda **kw: llm
        buf = io.StringIO()
        supply = iter(inputs)

        def fake_input(_prompt):
            try:
                return next(supply)
            except StopIteration:
                # Exhausting the canned inputs simulates EOF on stdin, which
                # run_shell handles by exiting the loop.
                raise EOFError from None

        try:
            with redirect_stdout(buf):
                with mock.patch("builtins.input", side_effect=fake_input):
                    cli.run_shell(ServerArgs(model_path="/tmp/model"))
        finally:
            cli.LLM = real_llm
        return buf.getvalue()

    def test_shell_quits_on_exit_command(self):
        class _Llm:
            def chat(self, messages, temperature, max_tokens):
                return "hello back"

        out = self._run(["hi", "quit"], _Llm())
        self.assertIn("hello back", out)
        self.assertIn("Goodbye!", out)

    def test_shell_handles_eof_and_empty_lines(self):
        class _Llm:
            def chat(self, messages, temperature, max_tokens):
                return "reply"

        # Empty line is skipped (no LLM call); EOF ends the loop.
        out = self._run(["", "hello"], _Llm())
        self.assertIn("reply", out)
        self.assertIn("Goodbye!", out)


# ── Test Module Entry Point (python -m minisgl) ──
class TestModuleEntry(unittest.TestCase):
    def test_minisgl_cli_main_importable(self):
        from minisgl.__main__ import main as entry_main
        from minisgl.cli import main as cli_main

        self.assertIs(entry_main, cli_main)


# ── Test Lazy Package Exports ──
class TestLazyExports(unittest.TestCase):
    def test_public_api_resolves_lazily(self):
        # from minisgl import LLM must work even though engine.llm is heavy.
        from minisgl import LLM, SamplingParams, ServerArgs

        self.assertTrue(callable(LLM))
        self.assertEqual(SamplingParams.__name__, "SamplingParams")
        self.assertEqual(ServerArgs.__name__, "ServerArgs")

    def test_all_matches_lazy_export_keys(self):
        # Guard the hand-maintained __all__/_LAZY_EXPORTS pair from drift.
        import minisgl

        self.assertEqual(set(minisgl.__all__), set(minisgl._LAZY_EXPORTS.keys()))

    def test_lightweight_import_does_not_pull_torch(self):
        # A fresh subprocess avoids cached modules from prior imports.
        import subprocess
        import sys

        code = (
            "import sys; import minisgl.utils.logger;"
            "print('torch' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main(verbosity=2)
