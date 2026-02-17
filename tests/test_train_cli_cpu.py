import sys
import unittest
from unittest.mock import patch

from yoctoGPT.train import parse_args


class TestTrainCliCPU(unittest.TestCase):
    def test_defaults_parse(self) -> None:
        with patch.object(sys, "argv", ["train.py"]):
            cfg = parse_args()
        self.assertEqual(cfg.mode, "char")
        self.assertEqual(cfg.save_strategy, "both")
        self.assertEqual(cfg.early_stopping_patience, 0)
        self.assertEqual(cfg.grad_accum_steps, 1)
        self.assertEqual(cfg.memmap_threshold_mb, 128)
        self.assertFalse(cfg.always_memmap)
        self.assertFalse(cfg.ddp)

    def test_new_flags_parse(self) -> None:
        argv = [
            "train.py",
            "--mode",
            "token",
            "--save_strategy",
            "best",
            "--early_stopping_patience",
            "7",
            "--early_stopping_min_delta",
            "0.01",
            "--grad_accum_steps",
            "3",
            "--activation_checkpointing",
            "--auto_microbatch",
            "--amp",
            "--amp_dtype",
            "fp16",
            "--memmap_threshold_mb",
            "42",
            "--always_memmap",
        ]
        with patch.object(sys, "argv", argv):
            cfg = parse_args()
        self.assertEqual(cfg.mode, "token")
        self.assertEqual(cfg.save_strategy, "best")
        self.assertEqual(cfg.early_stopping_patience, 7)
        self.assertAlmostEqual(cfg.early_stopping_min_delta, 0.01, places=7)
        self.assertEqual(cfg.grad_accum_steps, 3)
        self.assertTrue(cfg.activation_checkpointing)
        self.assertTrue(cfg.auto_microbatch)
        self.assertTrue(cfg.amp)
        self.assertEqual(cfg.amp_dtype, "fp16")
        self.assertEqual(cfg.memmap_threshold_mb, 42)
        self.assertTrue(cfg.always_memmap)


if __name__ == "__main__":
    unittest.main()
