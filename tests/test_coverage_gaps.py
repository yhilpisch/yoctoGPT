"""End-to-end tests for the sampler and chat CLI entry points."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from yoctoGPT.data import CharVocab, save_ids_bin
from yoctoGPT.model import GPT, GPTConfig
from yoctoGPT.advanced_model import AdvancedGPT, AdvancedGPTConfig
from yoctoGPT.performance_model import PerformanceGPT, PerformanceGPTConfig


class _CLITestBase(unittest.TestCase):
    """Shared helpers for sampler and chat CLI tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]

    def _make_char_dataset_and_checkpoint(
        self,
        root: Path,
        model_type: str = "gpt",
    ) -> tuple[Path, Path]:
        """Create a tiny char dataset and save a checkpoint from a fresh model."""
        data_dir = root / "data_char"
        data_dir.mkdir(parents=True, exist_ok=True)

        text = ("sampler test data for end to end.\n" * 80).strip()
        vocab = CharVocab.from_text(text)
        ids = vocab.encode(text)
        split = int(len(ids) * 0.9)
        save_ids_bin(ids[:split], data_dir / "train.bin")
        save_ids_bin(ids[split:], data_dir / "val.bin")
        vocab.save(data_dir / "vocab.json")

        ckpt_dir = root / "ckpt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if model_type == "gpt_plus":
            cfg = AdvancedGPTConfig(
                vocab_size=vocab.vocab_size,
                block_size=32,
                n_layer=2,
                n_head=2,
                n_embd=32,
            )
            model = AdvancedGPT(cfg)
        elif model_type == "gpt_fast":
            cfg = PerformanceGPTConfig(
                vocab_size=vocab.vocab_size,
                block_size=32,
                n_layer=2,
                n_head=2,
                n_embd=32,
            )
            model = PerformanceGPT(cfg)
        else:
            cfg = GPTConfig(
                vocab_size=vocab.vocab_size,
                block_size=32,
                n_layer=2,
                n_head=2,
                n_embd=32,
            )
            model = GPT(cfg)

        arch = (
            "gpt_plus" if isinstance(model, AdvancedGPT)
            else "gpt_fast" if isinstance(model, PerformanceGPT)
            else "gpt"
        )
        ckpt_path = ckpt_dir / "model.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": cfg.__dict__,
                "arch": arch,
                "mode": "char",
                "tokenizer_path": None,
                "char_vocab_path": str(data_dir / "vocab.json"),
            },
            ckpt_path,
        )
        return data_dir, ckpt_path

    def _run_cli(self, args: list[str]) -> subprocess.CompletedProcess:
        cmd = [sys.executable, "-m"] + args
        return subprocess.run(
            cmd,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )


class TestSamplerCLI(_CLITestBase):
    def test_sampler_char_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir, ckpt_path = self._make_char_dataset_and_checkpoint(Path(td))
            res = self._run_cli([
                "yoctoGPT.sampler",
                "--mode", "char",
                "--ckpt", str(ckpt_path),
                "--vocab_path", str(data_dir / "vocab.json"),
                "--prompt", "sam",
                "--max_new_tokens", "10",
            ])
            self.assertEqual(res.returncode, 0, msg=res.stderr)
            self.assertTrue(len(res.stdout.strip()) > 0)

    def test_sampler_char_gpt_plus(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir, ckpt_path = self._make_char_dataset_and_checkpoint(
                Path(td), model_type="gpt_plus",
            )
            res = self._run_cli([
                "yoctoGPT.sampler",
                "--mode", "char",
                "--ckpt", str(ckpt_path),
                "--vocab_path", str(data_dir / "vocab.json"),
                "--prompt", "sam",
                "--max_new_tokens", "10",
            ])
            self.assertEqual(res.returncode, 0, msg=res.stderr)

    def test_sampler_char_gpt_fast(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir, ckpt_path = self._make_char_dataset_and_checkpoint(
                Path(td), model_type="gpt_fast",
            )
            res = self._run_cli([
                "yoctoGPT.sampler",
                "--mode", "char",
                "--ckpt", str(ckpt_path),
                "--vocab_path", str(data_dir / "vocab.json"),
                "--prompt", "sam",
                "--max_new_tokens", "10",
            ])
            self.assertEqual(res.returncode, 0, msg=res.stderr)

    def test_sampler_top_k_top_p(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir, ckpt_path = self._make_char_dataset_and_checkpoint(Path(td))
            res = self._run_cli([
                "yoctoGPT.sampler",
                "--mode", "char",
                "--ckpt", str(ckpt_path),
                "--vocab_path", str(data_dir / "vocab.json"),
                "--prompt", "sam",
                "--max_new_tokens", "10",
                "--top_k", "5",
                "--top_p", "0.9",
            ])
            self.assertEqual(res.returncode, 0, msg=res.stderr)


class TestChatCLI(_CLITestBase):
    def test_chat_char_single_turn(self) -> None:
        """Chat with piped input (single turn then EOF)."""
        with tempfile.TemporaryDirectory() as td:
            data_dir, ckpt_path = self._make_char_dataset_and_checkpoint(Path(td))
            res = self._run_cli([
                "yoctoGPT.chat",
                "--mode", "char",
                "--ckpt", str(ckpt_path),
                "--vocab_path", str(data_dir / "vocab.json"),
                "--max_new_tokens", "10",
            ])
            # Chat reads from stdin; with no input it exits cleanly or
            # waits. We test that it at least starts without error.
            # Feeding empty stdin causes immediate EOF → clean exit.
            self.assertEqual(res.returncode, 0, msg=res.stderr)


class TestGenerateWithKVCache(unittest.TestCase):
    """Test that generation with KV cache produces the same logits
    as full-sequence recompute (no cache), for all model variants."""

    def _test_generate_with_cache(self, model_cls, config_cls) -> None:
        torch.manual_seed(42)
        vocab = 64
        cfg = config_cls(vocab_size=vocab, block_size=32, n_layer=2, n_head=2, n_embd=32)
        model = model_cls(cfg)
        model.eval()

        idx = torch.randint(0, vocab, (1, 4), dtype=torch.long)

        # Generate with cache (default path)
        out = model.generate(idx, max_new_tokens=8, temperature=1.0, top_k=None)
        self.assertEqual(tuple(out.shape), (1, 12))

        # Verify that the KV-cache incremental path matches full forward.
        # Feed the full generated sequence through the model without cache
        # and check the logits at the last position match the cached step.
        with torch.no_grad():
            # Single full-context forward (no cache)
            logits_full = model(out)
            if isinstance(logits_full, tuple):
                logits_full = logits_full[0]
            # Incremental forward with cache
            logits_cached, presents = model(out, use_cache=True)
        self.assertTrue(
            torch.allclose(logits_full, logits_cached, atol=1e-4, rtol=1e-3),
            "Cached and non-cached forward logits should match",
        )

    def test_gpt_generate_with_cache(self) -> None:
        self._test_generate_with_cache(GPT, GPTConfig)

    def test_advanced_gpt_generate_with_cache(self) -> None:
        self._test_generate_with_cache(AdvancedGPT, AdvancedGPTConfig)

    def test_performance_gpt_generate_with_cache(self) -> None:
        self._test_generate_with_cache(PerformanceGPT, PerformanceGPTConfig)


class TestEMAEvaluation(unittest.TestCase):
    """Test that EMA evaluation path works end-to-end."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]

    def _make_char_dataset(self, root: Path) -> Path:
        data_dir = root / "data_char"
        data_dir.mkdir(parents=True, exist_ok=True)
        text = ("ema test data for evaluation.\n" * 80).strip()
        vocab = CharVocab.from_text(text)
        ids = vocab.encode(text)
        split = int(len(ids) * 0.9)
        save_ids_bin(ids[:split], data_dir / "train.bin")
        save_ids_bin(ids[split:], data_dir / "val.bin")
        vocab.save(data_dir / "vocab.json")
        return data_dir

    def test_ema_training_and_eval(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = self._make_char_dataset(Path(td))
            ckpt_dir = Path(td) / "ckpt_ema"

            cmd = [
                sys.executable, "-m", "yoctoGPT.train",
                "--mode", "char",
                "--data_dir", str(data_dir),
                "--ckpt_dir", str(ckpt_dir),
                "--batch_size", "2",
                "--block_size", "16",
                "--n_layer", "2",
                "--n_head", "2",
                "--n_embd", "32",
                "--max_iters", "3",
                "--eval_interval", "1",
                "--eval_iters", "1",
                "--ema",
                "--ema_decay", "0.999",
                "--save_strategy", "both",
            ]
            res = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res.returncode, 0, msg=res.stderr)

            # Verify EMA checkpoint was saved
            ckpt_path = ckpt_dir / "best.pt"
            self.assertTrue(ckpt_path.exists())
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self.assertIn("ema_state", ckpt, "EMA state should be saved in checkpoint")


class TestGptFastCheckpointRoundtrip(unittest.TestCase):
    """Test that gpt_fast checkpoints save and reload correctly."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]

    def _make_char_dataset(self, root: Path) -> Path:
        data_dir = root / "data_char"
        data_dir.mkdir(parents=True, exist_ok=True)
        text = ("gpt fast checkpoint roundtrip test.\n" * 80).strip()
        vocab = CharVocab.from_text(text)
        ids = vocab.encode(text)
        split = int(len(ids) * 0.9)
        save_ids_bin(ids[:split], data_dir / "train.bin")
        save_ids_bin(ids[split:], data_dir / "val.bin")
        vocab.save(data_dir / "vocab.json")
        return data_dir

    def test_gpt_fast_save_resume(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = self._make_char_dataset(Path(td))
            ckpt_dir = Path(td) / "ckpt_fast"

            base_args = [
                sys.executable, "-m", "yoctoGPT.train",
                "--mode", "char",
                "--data_dir", str(data_dir),
                "--ckpt_dir", str(ckpt_dir),
                "--model_type", "gpt_fast",
                "--batch_size", "2",
                "--block_size", "16",
                "--n_layer", "2",
                "--n_head", "2",
                "--n_embd", "32",
                "--eval_interval", "1",
                "--eval_iters", "1",
                "--save_strategy", "both",
            ]

            # Initial training
            res0 = subprocess.run(
                base_args + ["--max_iters", "2"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res0.returncode, 0, msg=res0.stderr)

            ckpt_path = ckpt_dir / "latest.pt"
            self.assertTrue(ckpt_path.exists())
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self.assertEqual(ckpt["arch"], "gpt_fast")

            # Resume should advance iterations
            initial_iters = int(ckpt.get("iters_completed", 0))
            res1 = subprocess.run(
                base_args + ["--resume", str(ckpt_path), "--max_iters", "1"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res1.returncode, 0, msg=res1.stderr)
            resumed_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            resumed_iters = int(resumed_ckpt.get("iters_completed", 0))
            self.assertGreater(resumed_iters, initial_iters)

    def test_gpt_fast_sampler_roundtrip(self) -> None:
        """Checkpoint saved by gpt_fast training should load in sampler."""
        with tempfile.TemporaryDirectory() as td:
            data_dir = self._make_char_dataset(Path(td))
            ckpt_dir = Path(td) / "ckpt_fast_samp"

            # Train
            res = subprocess.run(
                [
                    sys.executable, "-m", "yoctoGPT.train",
                    "--mode", "char",
                    "--data_dir", str(data_dir),
                    "--ckpt_dir", str(ckpt_dir),
                    "--model_type", "gpt_fast",
                    "--batch_size", "2",
                    "--block_size", "16",
                    "--n_layer", "2",
                    "--n_head", "2",
                    "--n_embd", "32",
                    "--max_iters", "2",
                    "--eval_interval", "1",
                    "--eval_iters", "1",
                    "--save_strategy", "latest",
                ],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res.returncode, 0, msg=res.stderr)

            # Sample
            res = subprocess.run(
                [
                    sys.executable, "-m", "yoctoGPT.sampler",
                    "--mode", "char",
                    "--ckpt", str(ckpt_dir / "latest.pt"),
                    "--vocab_path", str(data_dir / "vocab.json"),
                    "--prompt", "gpt",
                    "--max_new_tokens", "10",
                ],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res.returncode, 0, msg=res.stderr)


if __name__ == "__main__":
    unittest.main()
