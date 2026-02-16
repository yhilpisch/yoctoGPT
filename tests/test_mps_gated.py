import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from yoctoGPT.data import CharVocab, save_ids_bin
from yoctoGPT.model import GPT, GPTConfig


def mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


@unittest.skipUnless(mps_available(), "MPS is not available on this machine")
class TestMpsGated(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]

    def _make_char_dataset(self, root: Path) -> Path:
        data_dir = root / "data_char"
        data_dir.mkdir(parents=True, exist_ok=True)
        text = ("mps training smoke data.\n" * 80).strip()
        vocab = CharVocab.from_text(text)
        ids = vocab.encode(text)
        split = int(len(ids) * 0.9)
        save_ids_bin(ids[:split], data_dir / "train.bin")
        save_ids_bin(ids[split:], data_dir / "val.bin")
        vocab.save(data_dir / "vocab.json")
        return data_dir

    def test_generate_on_mps(self) -> None:
        device = "mps"
        model = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=16)).to(device)
        x = torch.randint(0, 64, (1, 8), dtype=torch.long, device=device)
        out = model.generate(x, max_new_tokens=4)
        self.assertEqual(tuple(out.shape), (1, 12))

    def test_train_cli_mps_tiny(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            data_dir = self._make_char_dataset(td_path)
            ckpt_dir = td_path / "ckpt_mps"

            cmd = [
                sys.executable,
                "-m",
                "yoctoGPT.train",
                "--mode",
                "char",
                "--device",
                "mps",
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
                "--grad_accum_steps",
                "2",
                "--activation_checkpointing",
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
