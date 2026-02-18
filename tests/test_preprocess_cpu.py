import unittest

from yoctoGPT.preprocess import clean_book_text


class TestPreprocessCPU(unittest.TestCase):
    def test_drop_back_matter_and_boilerplate(self) -> None:
        txt = (
            "Chapter 1 Introduction\n"
            "No arbitrage links prices and payoffs.\n"
            "O'Reilly Media, Inc.\n"
            "ISBN 978-1-2345-6789-0\n"
            "References\n"
            "Some citation 2017.\n"
        )
        cleaned = clean_book_text(txt, drop_back_matter=True, lowercase=False, collapse_whitespace=True)
        self.assertIn("No arbitrage links prices and payoffs.", cleaned)
        self.assertNotIn("O'Reilly", cleaned)
        self.assertNotIn("ISBN", cleaned)
        self.assertNotIn("Some citation", cleaned)

    def test_preserve_case_option(self) -> None:
        txt = "Theory OF Pricing\n"
        low = clean_book_text(txt, lowercase=True, collapse_whitespace=True)
        keep = clean_book_text(txt, lowercase=False, collapse_whitespace=True)
        self.assertEqual(low, "theory of pricing")
        self.assertEqual(keep, "Theory OF Pricing")


if __name__ == "__main__":
    unittest.main()

