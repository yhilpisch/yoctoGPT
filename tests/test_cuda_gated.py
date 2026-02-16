import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from yoctoGPT.data import CharVocab, save_ids_bin
from yoctoGPT.model import GPT, GPTConfig


def cuda_available() -> bool:
    return torch.cuda.is_available()


@unittest.skipUnless(cuda_available(), "CUDA is not available on this machine")
class TestCudaGated(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]

    def _make_char_dataset(self, root: Path) -> Path:
        data_dir = root / "data_char"
        data_dir.mkdir(parents=True, exist_ok=True)
        text = ("cuda training smoke data.\n" * 80).strip()
        vocab = CharVocab.from_text(text)
        ids = vocab.encode(text)
        split = int(len(ids) * 0.9)
        save_ids_bin(ids[:split], data_dir / "train.bin")
        save_ids_bin(ids[split:], data_dir / "val.bin")
        vocab.save(data_dir / "vocab.json")
        return data_dir

    def test_forward_backward_cuda(self) -> None:
        device = "cuda"
        model = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=16)).to(device)
        x = torch.randint(0, 64, (2, 8), dtype=torch.long, device=device)
        logits = model(x)
        loss = logits.mean()
        loss.backward()
        grads = sum(1 for p in model.parameters() if p.grad is not None)
        self.assertGreater(grads, 0)

    def test_amp_autocast_cuda(self) -> None:
        device = "cuda"
        model = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=16)).to(device)
        x = torch.randint(0, 64, (2, 8), dtype=torch.long, device=device)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(x)
            loss = logits.mean()
        loss.backward()
        grads = sum(1 for p in model.parameters() if p.grad is not None)
        self.assertGreater(grads, 0)

    def test_train_cli_cuda_tiny(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            data_dir = self._make_char_dataset(td_path)
            ckpt_dir = td_path / "ckpt_cuda"

            amp_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
            cmd = [
                sys.executable,
                "-m",
                "yoctoGPT.train",
                "--mode",
                "char",
                "--device",
                "cuda",
                "--data_dir",
                str(data_dir),
                "--ckpt_dir",
                str(ckpt_dir),
                "--batch_size",
                "2",
                "--block_size",
                "16",
                "--n_layer",
                "2",
                "--n_head",
                "2",
                "--n_embd",
                "32",
                "--max_iters",
                "1",
                "--eval_interval",
                "1",
                "--eval_iters",
                "1",
                "--amp",
                "--amp_dtype",
                amp_dtype,
                "--grad_accum_steps",
                "2",
                "--auto_microbatch",
                "--save_strategy",
                "none",
            ]
            res = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res.returncode, 0, msg=res.stderr)


if __name__ == "__main__":
    unittest.main()
