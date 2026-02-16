import tempfile
import unittest
from pathlib import Path

import torch

from yoctoGPT.data import CharVocab, make_windows


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


if __name__ == "__main__":
    unittest.main()
