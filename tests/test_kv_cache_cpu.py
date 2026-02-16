import unittest

import torch

from yoctoGPT.advanced_model import AdvancedGPT, AdvancedGPTConfig
from yoctoGPT.model import GPT, GPTConfig
from yoctoGPT.performance_model import PerformanceGPT, PerformanceGPTConfig


class TestKVCacheCPU(unittest.TestCase):
    def _assert_cache_equivalence(self, model: torch.nn.Module, x: torch.Tensor) -> None:
        model.eval()
        with torch.no_grad():
            # Reference logits from a single full forward pass.
            full_logits = model(x)
            if isinstance(full_logits, tuple):
                full_logits = full_logits[0]

            # Incremental logits with cache.
            past_kv = None
            parts = []
            for t in range(x.size(1)):
                xt = x[:, t : t + 1]
                out = model(xt, past_kv=past_kv, use_cache=True)
                self.assertIsInstance(out, tuple)
                logits_t, past_kv = out
                parts.append(logits_t)
            cached_logits = torch.cat(parts, dim=1)

            self.assertEqual(tuple(cached_logits.shape), tuple(full_logits.shape))
            self.assertTrue(torch.allclose(cached_logits, full_logits, atol=1e-5, rtol=1e-4))

    def test_cache_equivalence_all_models(self) -> None:
        torch.manual_seed(11)
        vocab = 64
        x = torch.randint(0, vocab, (2, 8), dtype=torch.long)

        models = [
            GPT(GPTConfig(vocab_size=vocab, block_size=16, n_layer=2, n_head=2, n_embd=16)),
            AdvancedGPT(AdvancedGPTConfig(vocab_size=vocab, block_size=16, n_layer=2, n_head=2, n_embd=16)),
            PerformanceGPT(PerformanceGPTConfig(vocab_size=vocab, block_size=16, n_layer=2, n_head=2, n_embd=16)),
        ]
        for m in models:
            with self.subTest(model=m.__class__.__name__):
                self._assert_cache_equivalence(m, x)


if __name__ == "__main__":
    unittest.main()
