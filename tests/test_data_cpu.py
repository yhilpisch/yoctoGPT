import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from yoctoGPT.data import CharVocab, load_ids_adaptive, make_windows, sanitize_text_for_char_corpus, save_ids_bin


class TestDataCPU(unittest.TestCase):
    def test_char_vocab_roundtrip_and_persistence(self) -> None:
        text = "abca\n"
        vocab = CharVocab.from_text(text)
        ids = vocab.encode("abca")
        self.assertEqual(vocab.decode(ids), "abca")

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "vocab.json"
            vocab.save(p)
            loaded = CharVocab.load(p)
            self.assertEqual(loaded.stoi, vocab.stoi)
            self.assertEqual(loaded.itos, vocab.itos)
            self.assertEqual(loaded.decode(ids), "abca")

    def test_make_windows_shapes_and_shift(self) -> None:
        data_ids = torch.arange(0, 20, dtype=torch.long)
        ixs = torch.tensor([0, 3, 5], dtype=torch.long)
        x, y = make_windows(data_ids, block_size=4, ixs=ixs)

        self.assertEqual(tuple(x.shape), (3, 4))
        self.assertEqual(tuple(y.shape), (3, 4))
        # Targets should be inputs shifted by one in the original sequence.
        self.assertTrue(torch.equal(y[:, :-1], x[:, 1:]))

    def test_adaptive_load_threshold(self) -> None:
        ids = np.arange(2000, dtype=np.int32)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ids.bin"
            save_ids_bin(ids, p)

            in_memory = load_ids_adaptive(p, memmap_threshold_mb=1024, prefer_memmap=False)
            self.assertIsInstance(in_memory, torch.Tensor)

            mmap_data = load_ids_adaptive(p, memmap_threshold_mb=0, prefer_memmap=False)
            self.assertIsInstance(mmap_data, np.memmap)

            ixs = torch.tensor([0, 10, 100], dtype=torch.long)
            x, y = make_windows(mmap_data, block_size=8, ixs=ixs)
            self.assertEqual(tuple(x.shape), (3, 8))
            self.assertEqual(tuple(y.shape), (3, 8))

    def test_char_corpus_sanitization_modes(self) -> None:
        src = "Héllo — world!\nTabs\tand\tspaces.\nΩ-value #42"
        keep = sanitize_text_for_char_corpus(src, mode="none")
        ascii_only = sanitize_text_for_char_corpus(src, mode="ascii")
        basic = sanitize_text_for_char_corpus(src, mode="basic", lowercase=True, collapse_whitespace=True)

        self.assertIn("é", keep)
        self.assertNotIn("é", ascii_only)
        self.assertNotIn("Ω", ascii_only)
        self.assertEqual(basic, basic.lower())
        self.assertNotIn("#", basic)
        self.assertNotIn("\t", basic)

    def test_char_corpus_remove_punctuation_flags(self) -> None:
        src = "Hello, world! Is this working? Yes: maybe."
        no_punct = sanitize_text_for_char_corpus(
            src,
            mode="ascii",
            lowercase=True,
            collapse_whitespace=True,
            remove_punctuation=True,
            keep_period=False,
        )
        self.assertEqual(no_punct, "hello world is this working yes maybe")

        keep_period = sanitize_text_for_char_corpus(
            src,
            mode="ascii",
            lowercase=True,
            collapse_whitespace=True,
            remove_punctuation=True,
            keep_period=True,
        )
        self.assertEqual(keep_period, "hello world . is this working . yes maybe .")


if __name__ == "__main__":
    unittest.main()
