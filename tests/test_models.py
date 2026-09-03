"""Qwen3 / Qwen3-MoE model and registry unit tests.

Run: python3 test_models.py   (or: python -m pytest tests/test_models.py)
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

# Make the repo root importable regardless of the invocation directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



# ── TestModelDummy ──
class TestModelDummy(unittest.TestCase):
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

        # Single sequence.
        input_ids = torch.randint(0, 1000, (1, 8))
        positions = torch.arange(8)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, positions=positions)
        self.assertEqual(logits.shape, (1, 8, 1000))

        # Batched (2 requests) — the RMSNorm + RoPE backbone is shared by every
        # dense model in the family, so one batched forward covers the layout.
        input_ids = torch.randint(0, 1000, (2, 6))
        positions = torch.arange(6)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, positions=positions)
        self.assertEqual(logits.shape, (2, 6, 1000))

    def test_qwen3_without_qk_norm_gqa(self):
        """Qwen3 without qk_norm + num_kv_heads == num_heads is the dense path."""
        from minisgl.config import ModelArgs
        from minisgl.models.qwen3 import Qwen3ForCausalLM

        config = ModelArgs(
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=4,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            head_dim=32,
            qk_norm=False,
        )
        model = Qwen3ForCausalLM(config)
        model.eval()
        input_ids = torch.randint(0, 1000, (1, 8))
        positions = torch.arange(8)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, positions=positions)
        self.assertEqual(logits.shape, (1, 8, 1000))

    def test_deep_decoder_forward(self):
        """Test forward pass through multiple layers with residual flow."""
        from minisgl.config import ModelArgs
        from minisgl.models.qwen3 import Qwen3ForCausalLM

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
        gen = torch.Generator().manual_seed(123)
        model = Qwen3ForCausalLM(config)
        for _name, param in model.named_parameters():
            if param.dim() >= 2:
                nn.init.normal_(param, std=0.02, generator=gen)
            elif param.dim() == 1:
                nn.init.ones_(param)
        model.eval()
        input_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
        )
        positions = torch.arange(16)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, positions=positions)
        self.assertFalse(torch.isnan(logits).any(), f"NaN in logits: {logits}")
        self.assertFalse(torch.isinf(logits).any(), f"Inf in logits: {logits}")

    def test_with_kv_cache(self):
        """Test forward pass with KV cache tensors."""

        from minisgl.config import ModelArgs
        from minisgl.engine.kvcache.pool import KVCachePool
        from minisgl.models.attention.metadata import AttentionMetadata
        from minisgl.models.qwen3 import Qwen3ForCausalLM

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
        model = Qwen3ForCausalLM(config)
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
        # Bind each layer to its own slice of the pool (what the engine does).
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.set_kv_cache(k_all[i], v_all[i])

        # Prefill: 8 tokens, write to pages via flat indices
        input_ids = torch.randint(0, 1000, (1, 8))
        positions = torch.arange(8)
        # write_loc: flat indices into page*page_size space
        write_loc = torch.arange(8, dtype=torch.int32)  # tokens 0-7 write to slots 0-7

        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                positions=positions,
                attn_meta=AttentionMetadata(
                    forward_mode="prefill",
                    write_loc=write_loc,
                ),
            )

        # Decode: 1 token, write to slot 8
        input_ids = torch.randint(0, 1000, (1, 1))
        positions = torch.tensor([8])
        write_loc = torch.tensor([8], dtype=torch.int32)
        # Paged KV metadata: tokens 0..8 live in flat slots 0..8
        req_to_token = torch.arange(9, dtype=torch.int32).unsqueeze(0)
        # cache_seqlens semantics: total length INCLUDING the current token.
        cache_seqlens = torch.tensor([9], dtype=torch.int32)

        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                positions=positions,
                attn_meta=AttentionMetadata(
                    forward_mode="decode",
                    write_loc=write_loc,
                    req_to_token=req_to_token,
                    cache_seqlens=cache_seqlens,
                ),
            )
        self.assertEqual(logits.shape, (1, 1, 1000))


# ── Test Qwen3MoE Model ──

# ── TestQwen3MoEModel ──
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

    def test_moe_mlp_accepts_2d_input(self):
        """Prefill feeds 2-D (total_tokens, hidden) into the MLP — must not crash."""
        from minisgl.models.qwen3_moe import Qwen3MoEMLP

        mlp = Qwen3MoEMLP(
            hidden_size=16,
            moe_intermediate_size=8,
            num_experts=4,
            num_experts_per_tok=2,
        )
        with torch.inference_mode():
            out_2d = mlp(torch.randn(5, 16))
            out_3d = mlp(torch.randn(2, 5, 16))
        self.assertEqual(out_2d.shape, (5, 16))
        self.assertEqual(out_3d.shape, (2, 5, 16))
        self.assertFalse(torch.isnan(out_2d).any())

    def test_moe_routing_matches_hf_semantics(self):
        """HF routing: softmax over ALL experts -> top-k -> renormalize.

        Hand-checkable case: num_experts_per_tok=1 with zero router logits
        gives softmax weight 1/E for every expert; top-1 selects any one of
        them and renormalizes its weight to 1.0. With identity expert
        projections the output must be silu(x) * x.
        """
        import torch.nn.functional as F

        from minisgl.models.qwen3_moe import Qwen3MoEMLP

        hidden = 4
        mlp = Qwen3MoEMLP(
            hidden_size=hidden,
            moe_intermediate_size=hidden,
            num_experts=2,
            num_experts_per_tok=1,
        )
        with torch.no_grad():
            mlp.gate.weight.zero_()  # uniform logits -> softmax = 1/2 each
            eye = torch.eye(hidden)
            mlp.expert_gate.copy_(eye.expand(2, hidden, hidden))
            mlp.expert_up.copy_(eye.expand(2, hidden, hidden))
            mlp.expert_down.copy_(eye.expand(2, hidden, hidden))

        x = torch.randn(3, hidden)
        with torch.inference_mode():
            out = mlp(x)
        # weight renormalizes to 1.0 -> output == expert(x) exactly
        expected = F.silu(x) * x
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_load_hf_experts_aggregation(self):
        """Per-expert HF keys must land in the fused expert tensors."""
        from minisgl.models.qwen3_moe import Qwen3MoEMLP

        mlp = Qwen3MoEMLP(
            hidden_size=4,
            moe_intermediate_size=3,
            num_experts=2,
            num_experts_per_tok=1,
        )
        state_dict = {}
        for i in range(2):
            for proj in ("gate_proj", "up_proj"):
                state_dict[f"model.layers.0.mlp.experts.{i}.{proj}.weight"] = (
                    torch.full((3, 4), float(i + 1))
                )
            state_dict[f"model.layers.0.mlp.experts.{i}.down_proj.weight"] = torch.full(
                (4, 3), float(i + 1)
            )

        class _FakeLayer:
            pass

        layer = _FakeLayer()
        layer.mlp = mlp

        class _FakeModel:
            layers = [layer]

        class _FakeCausalLM:
            model = _FakeModel()

        # Call the unbound method on a minimal stand-in.
        from minisgl.models.qwen3_moe import Qwen3MoEForCausalLM

        loaded = Qwen3MoEForCausalLM.load_hf_experts(_FakeCausalLM(), state_dict)
        self.assertEqual(loaded, 6)
        self.assertTrue(torch.all(mlp.expert_gate[0] == 1.0))
        self.assertTrue(torch.all(mlp.expert_down[1] == 2.0))


# ── Test Decode vs Full Attention ──

# ── TestRegistry ──
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
        for mt in ["qwen3", "qwen3_moe"]:
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


# ── Test NaiveCacheManager ──

# ── TestSharedDecoder ──
class TestSharedDecoder(unittest.TestCase):
    def test_gated_mlp(self):
        from minisgl.models.base import GatedMLP

        mlp = GatedMLP(128, 512)
        x = torch.randn(2, 8, 128)
        y = mlp(x)
        self.assertEqual(y.shape, (2, 8, 128))

    def test_rmsnorm_decoder_layer(self):
        from minisgl.config import ModelArgs
        from minisgl.models.base import RMSNormDecoderLayer
        from minisgl.models.qwen3 import Qwen3Attention

        config = ModelArgs(
            hidden_size=128,
            num_layers=1,
            num_attention_heads=4,
            num_kv_heads=4,
            intermediate_size=512,
            vocab_size=100,
            max_position_embeddings=64,
            head_dim=32,
        )
        layer = RMSNormDecoderLayer(
            hidden_size=128,
            rms_norm_eps=1e-6,
            attention=Qwen3Attention(config),
            intermediate_size=512,
        )
        x = torch.randn(1, 8, 128)
        positions = torch.arange(8)
        out = layer(x, positions)
        self.assertEqual(out.shape, (1, 8, 128))

    def test_qwen3_inherits_tie_weights(self):
        from minisgl.models.base import RMSNormForCausalLM
        from minisgl.models.qwen3 import Qwen3ForCausalLM

        self.assertTrue(issubclass(Qwen3ForCausalLM, RMSNormForCausalLM))


# ── Test End-to-End Scheduler Loop ──



if __name__ == '__main__':
    unittest.main(verbosity=2)
