import pathlib
import sys
import unittest

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from denovo_utils import (  # noqa: E402
    BeamSearchDecoder,
    END_TOKEN,
    PaperMetricAccumulator,
    START_TOKEN,
    TOKEN_TO_ID,
    VOCABULARY,
    count_mass_matches,
    normalize_peptide,
)


class DecoderTests(unittest.TestCase):
    def test_special_token_roles(self):
        self.assertEqual(START_TOKEN, "[")
        self.assertEqual(END_TOKEN, "]")

    def test_decoder_returns_expected_sequence(self):
        probabilities = np.full((32, len(VOCABULARY)), 1e-6, dtype=np.float64)
        probabilities[0, TOKEN_TO_ID[START_TOKEN]] = 1.0
        probabilities[1, TOKEN_TO_ID["A"]] = 1.0
        probabilities[2, TOKEN_TO_ID["C"]] = 1.0
        probabilities[3, TOKEN_TO_ID[END_TOKEN]] = 1.0
        probabilities[4:, TOKEN_TO_ID["*"]] = 1.0
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        peptide, score = BeamSearchDecoder(beam_width=3).decode(probabilities)
        self.assertEqual(peptide, "AC")
        self.assertTrue(np.isfinite(score))


class MetricTests(unittest.TestCase):
    def test_isoleucine_is_normalized_to_leucine(self):
        self.assertEqual(normalize_peptide("IIL"), "LLL")
        self.assertEqual(count_mass_matches("IIL", "LLL"), 3)

    def test_exact_and_incorrect_peptides(self):
        metrics = PaperMetricAccumulator()
        metrics.update("ACD", "ACD")
        metrics.update("ACE", "ACD")
        summary = metrics.summary()

        self.assertEqual(summary["matched_peptides"], 1)
        self.assertEqual(summary["target_peptides"], 2)
        self.assertEqual(summary["error_counts"]["1"], 1)
        self.assertAlmostEqual(summary["peptide_recall"], 0.5)


class SourceTests(unittest.TestCase):
    def test_python_sources_do_not_contain_cjk_characters(self):
        for path in PROJECT_ROOT.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            cjk = [character for character in source if "\u4e00" <= character <= "\u9fff"]
            self.assertFalse(cjk, f"CJK characters found in {path.name}")


if __name__ == "__main__":
    unittest.main()
