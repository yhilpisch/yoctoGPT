import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from yoctoGPT.data import CharVocab, save_ids_bin


class TestTrainIntegrationCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]

    def _make_char_dataset(self, root: Path) -> Path:
        data_dir = root / "data_char"
        data_dir.mkdir(parents=True, exist_ok=True)

        text = ("the quick brown fox jumps over the lazy dog.\n" * 80).strip()
        vocab = CharVocab.from_text(text)
        ids = vocab.encode(text)
        split = int(len(ids) * 0.9)
        train_ids = ids[:split]
        val_ids = ids[split:]

        save_ids_bin(train_ids, data_dir / "train.bin")
        save_ids_bin(val_ids, data_dir / "val.bin")
        vocab.save(data_dir / "vocab.json")
        return data_dir

    def _run_train(self, args: list[str]) -> subprocess.CompletedProcess:
        cmd = [sys.executable, "-m", "yoctoGPT.train"] + args
        return subprocess.run(
            cmd,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_save_strategy_and_metrics_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            data_dir = self._make_char_dataset(td_path)

            common = [
                "--mode",
                "char",
                "--data_dir",
                str(data_dir),
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
                "2",
                "--eval_interval",
                "1",
                "--eval_iters",
                "1",
            ]

            # save_strategy=none
            ckpt_none = td_path / "ckpt_none"
            res_none = self._run_train(common + ["--ckpt_dir", str(ckpt_none), "--save_strategy", "none"])
            self.assertEqual(res_none.returncode, 0, msg=res_none.stderr)
            self.assertFalse((ckpt_none / "best.pt").exists())
            self.assertFalse((ckpt_none / "latest.pt").exists())

            # save_strategy=best
            ckpt_best = td_path / "ckpt_best"
            res_best = self._run_train(common + ["--ckpt_dir", str(ckpt_best), "--save_strategy", "best"])
            self.assertEqual(res_best.returncode, 0, msg=res_best.stderr)
            self.assertTrue((ckpt_best / "best.pt").exists())
            self.assertFalse((ckpt_best / "latest.pt").exists())

            # save_strategy=latest
            ckpt_latest = td_path / "ckpt_latest"
            res_latest = self._run_train(common + ["--ckpt_dir", str(ckpt_latest), "--save_strategy", "latest"])
            self.assertEqual(res_latest.returncode, 0, msg=res_latest.stderr)
            self.assertFalse((ckpt_latest / "best.pt").exists())
            self.assertTrue((ckpt_latest / "latest.pt").exists())

            metrics_path = ckpt_latest / "metrics.csv"
            self.assertTrue(metrics_path.exists())
            with metrics_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                rows = list(reader)

            expected_cols = {
                "val_ppl",
                "val_ppl_raw",
                "val_ppl_ema",
                "best_val_ppl",
                "best_val_ppl_raw",
                "best_val_ppl_ema",
                "micro_batch_size",
                "grad_accum_steps",
            }
            for col in expected_cols:
                self.assertIn(col, fieldnames)
            self.assertGreaterEqual(len(rows), 2)

    def test_resume_and_warm_start(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            data_dir = self._make_char_dataset(td_path)
            ckpt_dir = td_path / "ckpt_resume"

            base_args = [
                "--mode",
                "char",
                "--data_dir",
                str(data_dir),
                "--ckpt_dir",
                str(ckpt_dir),
                "--save_strategy",
                "both",
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
                "--eval_interval",
                "1",
                "--eval_iters",
                "1",
            ]

            # Initial run to produce checkpoints.
            res0 = self._run_train(base_args + ["--max_iters", "2"])
            self.assertEqual(res0.returncode, 0, msg=res0.stderr)
            latest = ckpt_dir / "latest.pt"
            best = ckpt_dir / "best.pt"
            self.assertTrue(latest.exists())
            self.assertTrue(best.exists())
            initial_iters = int(torch.load(latest, map_location="cpu").get("iters_completed", 0))
            self.assertGreater(initial_iters, 0)

            # Resume should advance completed iterations.
            res1 = self._run_train(base_args + ["--resume", str(latest), "--max_iters", "1"])
            self.assertEqual(res1.returncode, 0, msg=res1.stderr)
            resumed_iters = int(torch.load(latest, map_location="cpu").get("iters_completed", 0))
            self.assertGreater(resumed_iters, initial_iters)

            # Warm start from best weights should run successfully into a new dir.
            warm_dir = td_path / "ckpt_warm"
            res2 = self._run_train(
                base_args
                + [
                    "--ckpt_dir",
                    str(warm_dir),
                    "--init_from",
                    str(best),
                    "--max_iters",
                    "1",
                ]
            )
            self.assertEqual(res2.returncode, 0, msg=res2.stderr)
            self.assertTrue((warm_dir / "latest.pt").exists() or (warm_dir / "best.pt").exists())

    def test_early_stopping_triggers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            data_dir = self._make_char_dataset(td_path)
            ckpt_dir = td_path / "ckpt_early"

            args = [
                "--mode",
                "char",
                "--data_dir",
                str(data_dir),
                "--ckpt_dir",
                str(ckpt_dir),
                "--save_strategy",
                "latest",
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
                "5",
                "--eval_interval",
                "1",
                "--eval_iters",
                "1",
                "--early_stopping_patience",
                "1",
                "--early_stopping_min_delta",
                "10.0",
            ]
            res = self._run_train(args)
            self.assertEqual(res.returncode, 0, msg=res.stderr)
            self.assertIn("Early stopping triggered", res.stdout)

            metrics_path = ckpt_dir / "metrics.csv"
            with metrics_path.open("r", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(rows), 2)
            final_iter = int(rows[-1]["iter"])
            self.assertLess(final_iter, 5)


if __name__ == "__main__":
    unittest.main()
