import unittest

import torch

from yoctoGPT.model import GPT, GPTConfig


def mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


@unittest.skipUnless(mps_available(), "MPS is not available on this machine")
class TestMpsSmoke(unittest.TestCase):
    def test_tiny_forward_backward(self) -> None:
        device = "mps"
        model = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=16)).to(device)
        x = torch.randint(0, 64, (2, 8), dtype=torch.long, device=device)
        logits = model(x)
        loss = logits.mean()
        loss.backward()

        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        self.assertGreater(grad_count, 0)


if __name__ == "__main__":
    unittest.main()
