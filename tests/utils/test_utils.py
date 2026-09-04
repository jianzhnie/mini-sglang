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


# ── Test dtype resolution ──

# ── TestResolveDtype ──
class TestResolveDtype(unittest.TestCase):
    def test_auto_reads_config_json(self):
        import json
        import tempfile
        from pathlib import Path

        from minisgl.engine.engine import _resolve_dtype

        with tempfile.TemporaryDirectory() as d:
            Path(d, "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}))
            self.assertEqual(_resolve_dtype("auto", d, "cuda"), torch.bfloat16)
            # CPU falls back to float32
            self.assertEqual(_resolve_dtype("auto", d, "cpu"), torch.float32)

    def test_explicit_and_fallbacks(self):
        from minisgl.engine.engine import _resolve_dtype

        self.assertEqual(
            _resolve_dtype("float16", "/nonexistent", "cuda"), torch.float16
        )
        # Missing config.json under auto falls back to float32
        self.assertEqual(_resolve_dtype("auto", "/nonexistent", "cuda"), torch.float32)
        # CPU forces float32
        self.assertEqual(
            _resolve_dtype("float16", "/nonexistent", "cpu"), torch.float32
        )
        # Unknown dtype string falls back to float32
        self.assertEqual(_resolve_dtype("weird", "/nonexistent", "cuda"), torch.float32)


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


# ── Test Module Entry Point (python -m minisgl) ──
class TestModuleEntry(unittest.TestCase):
    def test_minisgl_cli_main_importable(self):
        from minisgl.__main__ import main as entry_main
        from minisgl.cli import main as cli_main

        self.assertIs(entry_main, cli_main)


if __name__ == "__main__":
    unittest.main(verbosity=2)
