"""Model layer & attention-backend unit tests.

Run: python3 tests/models/test_layers.py   (or: python -m pytest tests/models/test_layers.py)
"""

import sys
import unittest
from pathlib import Path

import torch

# Make the repo root importable regardless of the invocation directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))



# ── TestRMSNorm ──
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
            torch.allclose(out.float().pow(2).mean(-1), torch.ones(2, 10), atol=0.1),
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

# ── TestRoPE ──
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

# ── TestLinearLayers ──
class TestLinearLayers(unittest.TestCase):
    def test_column_parallel_linear_tp1(self):
        from minisgl.models.layers.linear import ColumnParallelLinear

        layer = ColumnParallelLinear(64, 128, bias=False)
        x = torch.randn(2, 10, 64)
        y = layer(x)
        self.assertEqual(y.shape, (2, 10, 128))

    def test_column_parallel_defaults(self):
        from minisgl.models.layers.linear import ColumnParallelLinear

        layer = ColumnParallelLinear(64, 128, bias=True)
        # Default: no all-gather (hidden-dim shards stay sharded under TP)
        self.assertFalse(layer.gather_output)
        # lm_head opts into gathering for full-vocab logits
        lm_head = ColumnParallelLinear(64, 128, bias=False, gather_output=True)
        self.assertTrue(lm_head.gather_output)
        # 1-D bias is marked for column-parallel sharding like the weight
        self.assertTrue(getattr(layer.bias, "is_column_parallel", False))

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

# ── TestAttentionBackend ──
class TestAttentionBackend(unittest.TestCase):
    def test_pytorch_backend(self):
        from minisgl.models.attention.dispatcher import AttentionBackend

        AttentionBackend.configure("fa")  # Use FA
        q = torch.randn(1, 8, 32, 64)  # (batch, heads, seq, head_dim)
        k = torch.randn(1, 8, 32, 64)
        v = torch.randn(1, 8, 32, 64)
        # attn_meta=None: plain causal self-attention without KV cache.
        out = AttentionBackend.forward(q, k, v)
        self.assertEqual(out.shape, q.shape)

    def test_package_reexports(self):
        """minisgl.models.attention re-exports must alias the submodule classes."""
        import minisgl.models.attention as attn_pkg
        from minisgl.models.attention.dispatcher import AttentionBackend
        from minisgl.models.attention.fa_backend import FlashAttentionBackend
        from minisgl.models.attention.pt_backend import PyTorchBackend

        self.assertIs(attn_pkg.AttentionBackend, AttentionBackend)
        self.assertIs(attn_pkg.FlashAttentionBackend, FlashAttentionBackend)
        self.assertIs(attn_pkg.PyTorchBackend, PyTorchBackend)


# ── Test Sampling ──

# ── TestDecodeMatchesFullAttention ──
class TestDecodeMatchesFullAttention(unittest.TestCase):
    def test_decode_with_cache_matches_full(self):
        """PT decode with cache_seqlens=total_len must match full attention
        over the whole sequence (query = last position only)."""
        import torch.nn.functional as F

        from minisgl.models.attention.metadata import AttentionMetadata
        from minisgl.models.attention.pt_backend import PyTorchBackend

        torch.manual_seed(0)
        num_heads, head_dim, total = 4, 16, 7
        k_cache = torch.zeros(2, 8, num_heads, head_dim)
        v_cache = torch.zeros(2, 8, num_heads, head_dim)
        k_full = torch.randn(1, num_heads, total, head_dim)
        v_full = torch.randn(1, num_heads, total, head_dim)
        # tokens 0..total-1 (incl. current) already written to flat slots
        k_cache.view(-1, num_heads, head_dim)[:total] = k_full[0].transpose(0, 1)
        v_cache.view(-1, num_heads, head_dim)[:total] = v_full[0].transpose(0, 1)

        q = torch.randn(1, num_heads, 1, head_dim)
        req_to_token = torch.arange(total, dtype=torch.int32).unsqueeze(0)
        cache_seqlens = torch.tensor([total], dtype=torch.int32)

        out = PyTorchBackend.forward(
            q,
            q,
            q,
            k_cache=k_cache,
            v_cache=v_cache,
            attn_meta=AttentionMetadata(
                forward_mode="decode",
                write_loc=None,
                req_to_token=req_to_token,
                cache_seqlens=cache_seqlens,
                max_seqlen=total,
            ),
        )
        ref = F.scaled_dot_product_attention(
            q, k_full, v_full, is_causal=False
        )  # single query at the last position sees all keys
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))


# ── Test Model Registry ──

# ── TestPyTorchBackendDecode ──
class TestPyTorchBackendDecode(unittest.TestCase):
    def test_decode_with_valid_mask(self):
        from minisgl.models.attention.metadata import AttentionMetadata
        from minisgl.models.attention.pt_backend import PyTorchBackend

        q = torch.randn(2, 4, 1, 32)
        k_cache = torch.randn(10, 16, 4, 32)
        v_cache = torch.randn(10, 16, 4, 32)

        req_to_token = torch.full((2, 8), -1, dtype=torch.int32)
        req_to_token[0, :3] = torch.tensor([0, 1, 2])
        req_to_token[1, :5] = torch.tensor([16, 17, 18, 19, 20])
        cache_seqlens = torch.tensor([3, 5], dtype=torch.int32)

        out = PyTorchBackend.forward(
            q,
            q[:, :, :, :],
            q[:, :, :, :],
            k_cache=k_cache,
            v_cache=v_cache,
            attn_meta=AttentionMetadata(
                forward_mode="decode",
                write_loc=None,
                req_to_token=req_to_token,
                cache_seqlens=cache_seqlens,
            ),
        )
        self.assertEqual(out.shape, (2, 4, 1, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_decode_missing_paged_metadata_raises(self):
        """attn_meta exists but lacks the decode fields: error, not fallback."""
        from minisgl.models.attention.metadata import AttentionMetadata
        from minisgl.models.attention.pt_backend import PyTorchBackend

        q = torch.randn(2, 4, 1, 32)
        k_cache = torch.randn(10, 16, 4, 32)
        v_cache = torch.randn(10, 16, 4, 32)
        meta = AttentionMetadata(forward_mode="decode", write_loc=None)
        with self.assertRaises(RuntimeError):
            PyTorchBackend.forward(q, q, q, k_cache, v_cache, meta)

    def test_none_attn_meta_is_plain_causal(self):
        """attn_meta=None must equal plain causal SDPA without a KV cache."""
        import torch.nn.functional as F

        from minisgl.models.attention.pt_backend import PyTorchBackend

        torch.manual_seed(0)
        q = torch.randn(1, 4, 8, 32)
        k = torch.randn(1, 4, 8, 32)
        v = torch.randn(1, 4, 8, 32)
        out = PyTorchBackend.forward(q, k, v)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        self.assertTrue(torch.allclose(out, ref, atol=1e-6))

    def test_prefill_varlen(self):
        from minisgl.models.attention.metadata import AttentionMetadata
        from minisgl.models.attention.pt_backend import PyTorchBackend

        q = torch.randn(1, 4, 8, 32)
        k = torch.randn(1, 4, 8, 32)
        v = torch.randn(1, 4, 8, 32)
        cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)

        out = PyTorchBackend.forward(
            q,
            k,
            v,
            attn_meta=AttentionMetadata(
                forward_mode="prefill",
                write_loc=None,
                cu_seqlens_q=cu_seqlens,
            ),
        )
        self.assertEqual(out.shape, (1, 4, 8, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_prefill_extend_matches_full_prefill(self):
        # Extend attention (cached prefix + uncached suffix) must match a full
        # causal prefill over the whole sequence.
        import torch.nn.functional as F

        from minisgl.models.attention.metadata import AttentionMetadata
        from minisgl.models.attention.pt_backend import PyTorchBackend

        torch.manual_seed(0)
        num_heads, head_dim, page_size = 4, 32, 4
        total, cached = 6, 4
        suffix = total - cached

        k_cache = torch.zeros(2, page_size, num_heads, head_dim)
        v_cache = torch.zeros(2, page_size, num_heads, head_dim)
        q_full = torch.randn(1, num_heads, total, head_dim)
        k_full = torch.randn(1, num_heads, total, head_dim)
        v_full = torch.randn(1, num_heads, total, head_dim)
        # KV cache flat layout: page_id * page_size + offset
        k_cache.view(-1, num_heads, head_dim)[:total] = k_full[0].transpose(0, 1)
        v_cache.view(-1, num_heads, head_dim)[:total] = v_full[0].transpose(0, 1)

        req_to_token = torch.full((1, 8), -1, dtype=torch.int32)
        req_to_token[0, :total] = torch.arange(total, dtype=torch.int32)
        prefix_lens = torch.tensor([cached], dtype=torch.int32)
        cu_seqlens = torch.tensor([0, suffix], dtype=torch.int32)

        out = PyTorchBackend.forward(
            q_full[:, :, cached:, :],
            k_full[:, :, cached:, :],
            v_full[:, :, cached:, :],
            k_cache=k_cache,
            v_cache=v_cache,
            attn_meta=AttentionMetadata(
                forward_mode="prefill",
                write_loc=None,
                cu_seqlens_q=cu_seqlens,
                prefix_lens=prefix_lens,
                req_to_token=req_to_token,
            ),
        )
        ref = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=True)[
            :, :, cached:, :
        ]
        self.assertEqual(out.shape, ref.shape)
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))


# ── Test BaseAttention helpers ──
class TestBaseAttentionHelpers(unittest.TestCase):
    """Exercises the shared BaseAttention machinery used by Qwen3/Qwen3-MoE."""

    def _make_attn(self):
        from minisgl.config import ModelArgs
        from minisgl.models.qwen3 import Qwen3Attention

        config = ModelArgs(
            hidden_size=128,
            num_layers=1,
            num_attention_heads=4,
            num_kv_heads=2,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=128,
            head_dim=32,
            qk_norm=True,
        )
        return Qwen3Attention(config)

    def test_reshape_for_attention(self):
        attn = self._make_attn()
        batch, seq = 2, 6
        q = torch.randn(batch, seq, attn.num_local_heads * attn.head_dim)
        k = torch.randn(batch, seq, attn.num_local_kv_heads * attn.head_dim)
        v = torch.randn(batch, seq, attn.num_local_kv_heads * attn.head_dim)
        qr, kr, vr = attn._reshape_for_attention(q, k, v, batch, seq)
        self.assertEqual(qr.shape, (batch, attn.num_local_heads, seq, attn.head_dim))
        self.assertEqual(kr.shape, (batch, attn.num_local_kv_heads, seq, attn.head_dim))
        self.assertEqual(vr.shape, (batch, attn.num_local_kv_heads, seq, attn.head_dim))

    def test_reshape_output_roundtrip(self):
        attn = self._make_attn()
        batch, seq = 1, 8
        # forward-shaped output: (batch, heads, seq, head_dim) -> (batch, seq, hidden)
        out = torch.randn(batch, attn.num_local_heads, seq, attn.head_dim)
        reshaped = attn._reshape_output(out, batch, seq)
        self.assertEqual(reshaped.shape, (batch, seq, attn.num_local_heads * attn.head_dim))
        # contiguous and independent of the input strides
        self.assertTrue(reshaped.is_contiguous())

    def test_pre_rope_hook_applies_qk_norm(self):
        attn = self._make_attn()
        q = torch.randn(1, attn.num_local_heads, 4, attn.head_dim)
        k = torch.randn(1, attn.num_local_kv_heads, 4, attn.head_dim)
        q_before = q.clone()
        k_before = k.clone()
        qn, kn = attn._pre_rope_hook(q, k)
        # QK-norm normalizes along the head dim, so values must change and
        # per-head RMS should be ~1.
        self.assertFalse(torch.allclose(qn, q_before, atol=1e-6))
        self.assertFalse(torch.allclose(kn, k_before, atol=1e-6))
        self.assertTrue(
            torch.allclose(
                qn.pow(2).mean(-1, keepdim=True).sqrt(), torch.ones_like(qn.pow(2).mean(-1, keepdim=True).sqrt()), atol=1e-3
            )
        )

    def test_write_kv_cache_scatter(self):
        """_write_kv_cache scatters K/V into the paged cache at write_loc."""
        attn = self._make_attn()
        num_kv_heads, head_dim = 2, 32
        num_pages, page_size = 2, 8
        k_cache = torch.zeros(num_pages, page_size, num_kv_heads, head_dim)
        v_cache = torch.zeros(num_pages, page_size, num_kv_heads, head_dim)
        attn.set_kv_cache(k_cache, v_cache)

        k = torch.randn(1, num_kv_heads, 3, head_dim)
        v = torch.randn(1, num_kv_heads, 3, head_dim)
        write_loc = torch.tensor([0, 8, -1], dtype=torch.int32)  # slots 0, page1-slot0; -1 skipped
        attn._write_kv_cache(k, v, write_loc)

        flat_k = k_cache.view(-1, num_kv_heads, head_dim)
        flat_v = v_cache.view(-1, num_kv_heads, head_dim)
        self.assertTrue(torch.equal(flat_k[0], k[0, :, 0, :]))
        self.assertTrue(torch.equal(flat_v[0], v[0, :, 0, :]))
        self.assertTrue(torch.equal(flat_k[8], k[0, :, 1, :]))
        self.assertTrue(torch.equal(flat_v[8], v[0, :, 1, :]))
        # The -1 slot must not be written (remains zero).
        self.assertTrue(torch.all(flat_k[5] == 0))
        self.assertTrue(torch.all(flat_v[5] == 0))

    def test_set_kv_cache_binds_slices(self):
        attn = self._make_attn()
        kc = torch.zeros(1, 1, attn.num_local_kv_heads, attn.head_dim)
        vc = torch.zeros(1, 1, attn.num_local_kv_heads, attn.head_dim)
        attn.set_kv_cache(kc, vc)
        self.assertIs(attn.k_cache, kc)
        self.assertIs(attn.v_cache, vc)
        # BaseAttention without a bound cache skips writes silently.
        attn.k_cache = None
        attn.v_cache = None
        k = torch.randn(1, attn.num_local_kv_heads, 1, attn.head_dim)
        v = torch.randn(1, attn.num_local_kv_heads, 1, attn.head_dim)
        attn._write_kv_cache(
            k, v, torch.tensor([0], dtype=torch.int32)
        )  # must not raise


if __name__ == '__main__':
    unittest.main(verbosity=2)
