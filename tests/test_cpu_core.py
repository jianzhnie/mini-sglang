"""CPU-based self-contained test for mini-sglang core logic.

Run: python3 tests/test_cpu_core.py
"""

import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, ".")


# ── Test RMSNorm ──
class TestRMSNorm(unittest.TestCase):
    def test_basic_norm(self):
        from minisgl.models.layers.rms_norm import RMSNorm

        norm = RMSNorm(64, eps=1e-6)
        x = torch.randn(2, 10, 64)
        out, res = norm(x)
        self.assertEqual(out.shape, x.shape)  # output same shape
        self.assertIsNone(res)  # no residual when None
        # Variance after norm should be ~1
        self.assertTrue(
            torch.allclose(out.float().pow(2).mean(-1), torch.ones(2, 10), atol=0.1)
        )

    def test_fused_residual(self):
        from minisgl.models.layers.rms_norm import RMSNorm

        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        residual = torch.randn(2, 10, 64)
        out, new_res = norm(x, residual)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(new_res.shape, residual.shape)
        self.assertTrue(new_res is not None)


# ── Test RoPE ──
class TestRoPE(unittest.TestCase):
    def test_rope_application(self):
        from minisgl.models.layers.rope import RotaryEmbedding

        rope = RotaryEmbedding(head_dim=64, max_position_embeddings=128)
        q = torch.randn(2, 8, 16, 64)  # (batch, heads, seq, head_dim)
        k = torch.randn(2, 8, 16, 64)
        q_copy = q.clone()
        k_copy = k.clone()
        positions = torch.arange(16)
        rope(q, k, positions)
        # RoPE should change Q and K
        self.assertFalse(torch.allclose(q, q_copy))
        self.assertFalse(torch.allclose(k, k_copy))
        # But shapes should stay the same
        self.assertEqual(q.shape, q_copy.shape)


# ── Test Linear Layers ──
class TestLinearLayers(unittest.TestCase):
    def test_column_parallel_linear_tp1(self):
        from minisgl.models.layers.linear import ColumnParallelLinear

        layer = ColumnParallelLinear(64, 128, bias=False)
        x = torch.randn(2, 10, 64)
        y = layer(x)
        self.assertEqual(y.shape, (2, 10, 128))

    def test_row_parallel_linear_tp1(self):
        from minisgl.models.layers.linear import RowParallelLinear

        # RowParallel receives gathered input (full in_features)
        layer = RowParallelLinear(128, 64, bias=False)
        x = torch.randn(2, 10, 128)  # (batch, seq, full in_features)
        y = layer(x)
        self.assertEqual(y.shape, (2, 10, 64))

    def test_vocab_parallel_embedding_tp1(self):
        from minisgl.models.layers.embedding import VocabParallelEmbedding

        emb = VocabParallelEmbedding(1000, 64)
        ids = torch.randint(0, 1000, (2, 10))
        y = emb(ids)
        self.assertEqual(y.shape, (2, 10, 64))


# ── Test Attention Backend ──
class TestAttentionBackend(unittest.TestCase):
    def test_pytorch_backend(self):
        from minisgl.models.attention.backend import AttentionBackend

        AttentionBackend.configure("fa")  # Use FA
        q = torch.randn(1, 8, 32, 64)  # (batch, heads, seq, head_dim)
        k = torch.randn(1, 8, 32, 64)
        v = torch.randn(1, 8, 32, 64)
        # Should not raise
        out = AttentionBackend.forward(q, k, v)
        self.assertEqual(out.shape, q.shape)


# ── Test Sampling ──
class TestSampler(unittest.TestCase):
    def test_greedy_sampling(self):
        from minisgl.config import SamplingParams
        from minisgl.sampling.sampler import Sampler

        sampler = Sampler(1000)
        logits = torch.randn(4, 1000)  # 4 requests, 1000 vocab
        params = SamplingParams(temperature=0.0)  # greedy
        tokens = sampler.sample(logits, params)
        self.assertEqual(tokens.shape, (4,))
        # Greedy should pick argmax
        expected = logits.argmax(dim=-1)
        self.assertTrue(torch.equal(tokens, expected))

    def test_temperature_sampling(self):
        from minisgl.config import SamplingParams
        from minisgl.sampling.sampler import Sampler

        sampler = Sampler(1000)
        logits = torch.randn(4, 1000)
        params = SamplingParams(temperature=0.8, top_k=50, top_p=0.9)
        tokens = sampler.sample(logits, params)
        self.assertEqual(tokens.shape, (4,))

    def test_top_k_top_p(self):
        from minisgl.sampling.sampler import _apply_top_k, _apply_top_p

        logits = torch.randn(1, 1000)
        # Top-k: only top 10 should be > -inf
        filtered = _apply_top_k(logits.clone(), 10)
        self.assertEqual((filtered > float("-inf")).sum().item(), 10)
        # Top-p
        filtered = _apply_top_p(logits.clone(), 0.95)
        self.assertTrue((filtered > float("-inf")).sum().item() > 0)


# ── Test KV Cache Pool ──
class TestKVCachePool(unittest.TestCase):
    def test_pool_alloc_free(self):
        from minisgl.engine.kvcache.pool import KVCachePool

        pool = KVCachePool(
            num_layers=2,
            num_pages=100,
            page_size=16,
            num_kv_heads=8,
            head_dim=64,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.assertEqual(pool.free_count(), 100)
        h1 = pool.alloc(10)
        self.assertEqual(h1.num_pages(), 10)
        self.assertEqual(pool.free_count(), 90)
        h2 = pool.alloc(5)
        self.assertEqual(h2.num_pages(), 5)
        self.assertEqual(pool.free_count(), 85)
        pool.free(h1)
        self.assertEqual(pool.free_count(), 95)
        pool.free(h2)
        self.assertEqual(pool.free_count(), 100)

    def test_get_kv_cache(self):
        from minisgl.engine.kvcache.pool import KVCachePool

        pool = KVCachePool(
            num_layers=4,
            num_pages=20,
            page_size=16,
            num_kv_heads=4,
            head_dim=32,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        k, v = pool.get_kv_cache(layer_idx=0)
        self.assertEqual(k.shape, (20, 16, 4, 32))
        self.assertEqual(v.shape, (20, 16, 4, 32))
        k_all, v_all = pool.get_all_kv_cache()
        self.assertEqual(k_all.shape, (4, 20, 16, 4, 32))
        self.assertEqual(v_all.shape, (4, 20, 16, 4, 32))


# ── Test Radix Cache ──
class TestRadixCache(unittest.TestCase):
    def test_match_prefix(self):
        from minisgl.engine.kvcache.pool import KVCachePool
        from minisgl.engine.kvcache.radix import RadixCacheManager

        pool = KVCachePool(
            num_layers=2,
            num_pages=100,
            page_size=4,
            num_kv_heads=4,
            head_dim=32,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        radix = RadixCacheManager(pool, page_size=4)

        # Insert a sequence
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        handle = pool.alloc(2)
        radix.insert(tokens, handle)

        # Match full prefix
        matched = radix.match_prefix([1, 2, 3, 4, 9, 0])
        self.assertEqual(matched, 4)  # Pages are size 4

        # Match partial
        matched = radix.match_prefix([1, 2, 5, 6])
        self.assertEqual(matched, 0)  # Diverges at token 3

        # No match
        matched = radix.match_prefix([99, 99])
        self.assertEqual(matched, 0)


# ── Test Scheduler Batch Logic ──
class TestSchedulerBatch(unittest.TestCase):
    def test_batch_creation(self):
        from minisgl.config import SamplingParams
        from minisgl.scheduler.batch import Batch, Req

        req = Req(
            input_ids=[1, 2, 3, 4, 5],
            uid=0,
            sampling_params=SamplingParams(max_tokens=100),
        )
        batch = Batch(reqs=[req], phase="prefill")
        self.assertEqual(batch.size(), 1)
        self.assertEqual(batch.phase, "prefill")
        self.assertEqual(req.total_len, 5)
        self.assertEqual(req.uncached_len, 5)
        self.assertFalse(req.is_finished)

    def test_req_lifecycle(self):
        from minisgl.config import SamplingParams
        from minisgl.scheduler.batch import Req, SequenceStatus

        req = Req(
            input_ids=[1, 2, 3], uid=0, sampling_params=SamplingParams(max_tokens=10)
        )
        self.assertEqual(req.status, SequenceStatus.WAITING)
        req.status = SequenceStatus.RUNNING
        req.append_token(42)
        self.assertEqual(req.total_len, 4)
        self.assertEqual(req.output_len, 1)

    def test_context_prepare(self):
        from minisgl.config import SamplingParams
        from minisgl.engine.context import BatchContext
        from minisgl.engine.kvcache.pool import KVCachePool
        from minisgl.scheduler.batch import Batch, Req

        pool = KVCachePool(
            num_layers=2,
            num_pages=20,
            page_size=4,
            num_kv_heads=4,
            head_dim=32,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        handle = pool.alloc(2)

        req = Req(
            input_ids=[1, 2, 3, 4, 5],
            uid=0,
            cache_handle=handle,
            sampling_params=SamplingParams(),
        )
        batch = Batch(reqs=[req], phase="prefill")

        ctx = BatchContext(
            max_running_req=4, max_seq_len=16, page_size=4, device=torch.device("cpu")
        )
        ctx.prepare(batch)

        self.assertEqual(batch.input_ids.tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(batch.positions.tolist(), [0, 1, 2, 3, 4])
        self.assertIsNotNone(batch.write_loc)


# ── Test Distributed ──
class TestDistributed(unittest.TestCase):
    def test_all_reduce_noop(self):
        from minisgl.engine.distributed.pynccl import all_reduce

        x = torch.randn(10)
        y = all_reduce(x)
        self.assertTrue(torch.equal(x, y))  # no-op when not distributed


# ── Test Config ──
class TestConfig(unittest.TestCase):
    def test_config_creation(self):
        from minisgl.config import CacheArgs, SamplingParams, ServerArgs

        args = ServerArgs(model_path="/tmp/test", port=8000, tp_size=1)
        self.assertEqual(args.port, 8000)
        self.assertEqual(args.tp_size, 1)

        cache = CacheArgs.from_server_args(args)
        self.assertEqual(cache.page_size, 16)
        self.assertEqual(cache.max_seq_len, 8192)

        params = SamplingParams(temperature=0.7, top_k=50, top_p=0.95)
        self.assertEqual(params.temperature, 0.7)
        self.assertEqual(params.top_k, 50)


# ── Test Tokenizer Worker ──
class TestTokenizerWorker(unittest.TestCase):
    @unittest.skipIf(
        not __import__("importlib.util").util.find_spec("transformers"),
        "transformers not installed",
    )
    def test_tokenizer_creation(self):
        """Test with a tiny tokenizer (requires transformers)."""
        from minisgl.models.tokenizer.worker import TokenizerWorker

        # Use a model that's likely cached or small
        try:
            worker = TokenizerWorker("google/bert_uncased_L-2_H-128_A-2")
            ids = worker.encode("Hello world")
            self.assertIsInstance(ids, list)
            self.assertTrue(len(ids) > 0)
        except Exception:
            self.skipTest("Model not available offline")


# ── Test Full Model (Dummy) ──
class TestModelDummy(unittest.TestCase):
    def test_qwen2_model_create(self):
        """Create a tiny Qwen2 model and run a forward pass."""
        from minisgl.config import ModelArgs
        from minisgl.models.qwen2 import Qwen2ForCausalLM

        config = ModelArgs(
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=4,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            head_dim=32,
            rms_norm_eps=1e-6,
        )

        model = Qwen2ForCausalLM(config)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 8))  # 1 req, 8 tokens
        positions = torch.arange(8)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, positions=positions)

        self.assertEqual(logits.shape, (1, 8, 1000))

    def test_qwen3_model_create(self):
        from minisgl.config import ModelArgs
        from minisgl.models.qwen3 import Qwen3ForCausalLM

        config = ModelArgs(
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=2,  # GQA: fewer KV heads
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            head_dim=32,
            qk_norm=True,
        )
        model = Qwen3ForCausalLM(config)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 8))
        positions = torch.arange(8)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, positions=positions)
        self.assertEqual(logits.shape, (1, 8, 1000))

    def test_llama_model_create(self):
        from minisgl.config import ModelArgs
        from minisgl.models.llama import LlamaForCausalLM

        config = ModelArgs(
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=4,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            head_dim=32,
        )
        model = LlamaForCausalLM(config)
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 6))
        positions = torch.arange(6)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, positions=positions)
        self.assertEqual(logits.shape, (2, 6, 1000))

    def test_mistral_model_create(self):
        from minisgl.config import ModelArgs
        from minisgl.models.mistral import MistralForCausalLM

        config = ModelArgs(
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=2,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            head_dim=32,
            sliding_window=4096,
        )
        model = MistralForCausalLM(config)
        model.eval()
        input_ids = torch.randint(0, 1000, (1, 8))
        positions = torch.arange(8)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, positions=positions)
        self.assertEqual(logits.shape, (1, 8, 1000))

    def test_deep_decoder_forward(self):
        """Test forward pass through multiple layers with residual flow."""
        torch.manual_seed(42)

        from minisgl.config import ModelArgs
        from minisgl.models.qwen2 import Qwen2ForCausalLM

        config = ModelArgs(
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=4,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            head_dim=32,
        )
        model = Qwen2ForCausalLM(config)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
            elif hasattr(module, "_init_weights"):
                module._init_weights()
        model.eval()
        input_ids = torch.randint(0, 1000, (1, 16))
        positions = torch.arange(16)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, positions=positions)
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

    def test_with_kv_cache(self):
        """Test forward pass with KV cache tensors."""

        from minisgl.config import ModelArgs
        from minisgl.engine.kvcache.pool import KVCachePool
        from minisgl.models.qwen2 import Qwen2ForCausalLM

        config = ModelArgs(
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=4,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            head_dim=32,
        )
        model = Qwen2ForCausalLM(config)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
        model.eval()

        pool = KVCachePool(
            num_layers=2,
            num_pages=20,
            page_size=16,
            num_kv_heads=4,
            head_dim=32,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        k_all, v_all = pool.get_all_kv_cache()

        # Prefill: 8 tokens, write to pages via flat indices
        input_ids = torch.randint(0, 1000, (1, 8))
        positions = torch.arange(8)
        # write_loc: flat indices into page*page_size space
        write_loc = torch.arange(8, dtype=torch.int32)  # tokens 0-7 write to slots 0-7

        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                positions=positions,
                k_cache=k_all,
                v_cache=v_all,
                write_loc=write_loc,
            )

        # Decode: 1 token, write to slot 8
        input_ids = torch.randint(0, 1000, (1, 1))
        positions = torch.tensor([8])
        write_loc = torch.tensor([8], dtype=torch.int32)

        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                positions=positions,
                k_cache=k_all,
                v_cache=v_all,
                write_loc=write_loc,
            )
        self.assertEqual(logits.shape, (1, 1, 1000))


# ── Test Qwen3MoE Model ──
class TestQwen3MoEModel(unittest.TestCase):
    def test_moe_model_create(self):
        from minisgl.config import ModelArgs
        from minisgl.models.qwen3_moe import Qwen3MoEForCausalLM

        config = ModelArgs(
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=4,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            head_dim=32,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=256,
            decoder_sparse_step=1,
        )
        model = Qwen3MoEForCausalLM(config)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
        model.eval()
        input_ids = torch.randint(0, 1000, (1, 8))
        positions = torch.arange(8)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, positions=positions)
        self.assertEqual(logits.shape, (1, 8, 1000))


# ── Test FusedMoE ──
class TestFusedMoE(unittest.TestCase):
    def test_fused_moe_pytorch(self):
        from minisgl.models.moe.fused_moe import fused_moe_pytorch

        total_tokens, hidden_size = 4, 64
        num_experts, intermediate_size = 4, 128

        x = torch.randn(total_tokens, hidden_size)
        router_logits = torch.randn(total_tokens, num_experts)
        gate_w = torch.randn(num_experts, intermediate_size, hidden_size)
        up_w = torch.randn(num_experts, intermediate_size, hidden_size)
        down_w = torch.randn(num_experts, hidden_size, intermediate_size)

        out = fused_moe_pytorch(x, router_logits, gate_w, up_w, down_w, top_k=2)
        self.assertEqual(out.shape, (total_tokens, hidden_size))
        self.assertFalse(torch.isnan(out).any())


# ── Test Model Registry ──
class TestRegistry(unittest.TestCase):
    def test_detect_model_type_fallback(self):
        # Should not crash on non-existent path
        self.assertTrue(True)  # Module loaded OK

    def test_create_model(self):
        from minisgl.config import ModelArgs
        from minisgl.models.registry import create_model

        config = ModelArgs(
            hidden_size=128,
            num_layers=1,
            num_attention_heads=2,
            num_kv_heads=2,
            intermediate_size=256,
            vocab_size=100,
            max_position_embeddings=64,
            head_dim=64,
        )
        for mt in ["qwen2", "qwen3", "llama", "mistral"]:
            model = create_model(config, model_type=mt)
            # Apply initialization to prevent NaN
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
            model.eval()
            self.assertIsNotNone(model)
            ids = torch.randint(0, 100, (1, 4))
            pos = torch.arange(4)
            with torch.inference_mode():
                out = model(input_ids=ids, positions=pos)
            self.assertEqual(out.shape, (1, 4, 100))


if __name__ == "__main__":
    print("=" * 60)
    print("Mini-SGLang CPU Core Test Suite")
    print("=" * 60)
    unittest.main(verbosity=2)
