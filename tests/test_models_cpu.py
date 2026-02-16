import unittest

import torch

from yoctoGPT.advanced_model import AdvancedGPT, AdvancedGPTConfig
from yoctoGPT.model import GPT, GPTConfig
from yoctoGPT.performance_model import PerformanceGPT, PerformanceGPTConfig


class TestModelsCPU(unittest.TestCase):
    def test_forward_backward_and_generate_cpu(self) -> None:
        torch.manual_seed(7)
        batch = 2
        seq = 8
        vocab = 64
        x = torch.randint(0, vocab, (batch, seq), dtype=torch.long)

        model_specs = [
            (GPT, GPTConfig(vocab_size=vocab, block_size=16, n_layer=2, n_head=2, n_embd=16)),
            (AdvancedGPT, AdvancedGPTConfig(vocab_size=vocab, block_size=16, n_layer=2, n_head=2, n_embd=16)),
            (PerformanceGPT, PerformanceGPTConfig(vocab_size=vocab, block_size=16, n_layer=2, n_head=2, n_embd=16)),
        ]

        for model_cls, cfg in model_specs:
            with self.subTest(model=model_cls.__name__):
                model = model_cls(cfg)
                model.train()
                logits = model(x)
                self.assertEqual(tuple(logits.shape), (batch, seq, vocab))

                loss = logits.mean()
                loss.backward()
                grad_count = sum(1 for p in model.parameters() if p.grad is not None)
                self.assertGreater(grad_count, 0)

                model.eval()
                out = model.generate(x[:1], max_new_tokens=4)
                self.assertEqual(tuple(out.shape), (1, seq + 4))


if __name__ == "__main__":
    unittest.main()
