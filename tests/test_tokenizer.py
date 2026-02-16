import unittest

from yoctoGPT.tokenizer import TOKEN_BOS, TOKEN_EOS, TOKEN_UNK, BPETokenizer, WordLevelTokenizer, _hf_available


class TestTokenizer(unittest.TestCase):
    def test_word_level_bos_eos_roundtrip(self) -> None:
        text = "Hello world. Hello test."
        tok = WordLevelTokenizer.train(text, vocab_size=128)

        self.assertIsNotNone(tok.bos_id)
        self.assertIsNotNone(tok.eos_id)
        self.assertIn(TOKEN_UNK, tok.stoi)
        self.assertIn(TOKEN_BOS, tok.stoi)
        self.assertIn(TOKEN_EOS, tok.stoi)

        ids = tok.encode("Hello world", add_bos=True, add_eos=True)
        self.assertEqual(ids[0], tok.bos_id)
        self.assertEqual(ids[-1], tok.eos_id)

        decoded = tok.decode(ids)
        self.assertIn("hello", decoded)
        self.assertNotIn(TOKEN_BOS, decoded)
        self.assertNotIn(TOKEN_EOS, decoded)

    @unittest.skipUnless(_hf_available(), "tokenizers backend not installed")
    def test_bpe_bos_eos_roundtrip(self) -> None:
        text = "Hello world. Another line."
        tok = BPETokenizer.train(text, vocab_size=128)

        self.assertIsNotNone(tok.bos_id)
        self.assertIsNotNone(tok.eos_id)

        ids = tok.encode("Hello world", add_bos=True, add_eos=True)
        self.assertEqual(ids[0], tok.bos_id)
        self.assertEqual(ids[-1], tok.eos_id)

        decoded = tok.decode(ids)
        self.assertIn("Hello", decoded)


if __name__ == "__main__":
    unittest.main()
