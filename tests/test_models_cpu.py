import unittest
import types

import torch

from yoctoGPT.advanced_model import AdvancedGPT, AdvancedGPTConfig
from yoctoGPT.config import ModelConfigBase
from yoctoGPT.model import GPT, GPTConfig
from yoctoGPT.performance_model import PerformanceGPT, PerformanceGPTConfig


class TestModelsCPU(unittest.TestCase):
    def test_model_configs_share_base(self) -> None:
        self.assertTrue(issubclass(GPTConfig, ModelConfigBase))
        self.assertTrue(issubclass(AdvancedGPTConfig, ModelConfigBase))
        self.assertTrue(issubclass(PerformanceGPTConfig, ModelConfigBase))

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
                if isinstance(logits, tuple):
                    logits = logits[0]
                self.assertEqual(tuple(logits.shape), (batch, seq, vocab))

                # Unified API: all models should support labels and return (logits, loss).
                out = model(x, labels=x)
                self.assertIsInstance(out, tuple)
                logits_labeled, loss_labeled = out
                self.assertEqual(tuple(logits_labeled.shape), (batch, seq, vocab))
                self.assertEqual(loss_labeled.ndim, 0)

                loss = logits.mean() + loss_labeled
                loss.backward()
                grad_count = sum(1 for p in model.parameters() if p.grad is not None)
                self.assertGreater(grad_count, 0)

                model.eval()
                out = model.generate(x[:1], max_new_tokens=4)
                self.assertEqual(tuple(out.shape), (1, seq + 4))

    def test_generate_per_sequence_eos_masking(self) -> None:
        torch.manual_seed(13)
        batch_input = torch.tensor([[3], [4]], dtype=torch.long)
        eos_token = 0
        vocab = 8

        model_specs = [
            (GPT, GPTConfig(vocab_size=vocab, block_size=16, n_layer=2, n_head=2, n_embd=16)),
            (AdvancedGPT, AdvancedGPTConfig(vocab_size=vocab, block_size=16, n_layer=2, n_head=2, n_embd=16)),
            (PerformanceGPT, PerformanceGPTConfig(vocab_size=vocab, block_size=16, n_layer=2, n_head=2, n_embd=16)),
        ]

        for model_cls, cfg in model_specs:
            with self.subTest(model=model_cls.__name__):
                model = model_cls(cfg)
                model.eval()
                model._step = 0

                def fake_forward(self, idx, *args, **kwargs):
                    step = self._step
                    self._step += 1
                    bsz, tlen = idx.shape
                    logits = torch.full((bsz, tlen, self.config.vocab_size), -100.0, device=idx.device)
                    if step == 0:
                        logits[0, -1, eos_token] = 10.0
                        logits[1, -1, 1] = 10.0
                    elif step == 1:
                        logits[0, -1, 2] = 10.0
                        logits[1, -1, 1] = 10.0
                    else:
                        logits[0, -1, 2] = 10.0
                        logits[1, -1, eos_token] = 10.0
                    return logits, []

                model.forward = types.MethodType(fake_forward, model)  # type: ignore[method-assign]
                out = model.generate(batch_input.clone(), max_new_tokens=4, top_k=1, eos_token=eos_token)
                new_tokens = out[:, batch_input.size(1) :]

                for row in range(new_tokens.size(0)):
                    row_tokens = new_tokens[row]
                    eos_positions = (row_tokens == eos_token).nonzero(as_tuple=True)[0]
                    if eos_positions.numel() == 0:
                        continue
                    first = int(eos_positions[0].item())
                    self.assertTrue(bool((row_tokens[first:] == eos_token).all()))


if __name__ == "__main__":
    unittest.main()
