"""Decoding and paper-aligned evaluation utilities for DenNovo."""

from dataclasses import dataclass, field

import numpy as np


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYZ"
PAD_TOKEN = "*"
END_TOKEN = "]"
START_TOKEN = "["
VOCABULARY = [PAD_TOKEN] + list(AMINO_ACIDS) + [END_TOKEN, START_TOKEN]
ID_TO_TOKEN = {index: token for index, token in enumerate(VOCABULARY)}
TOKEN_TO_ID = {token: index for index, token in ID_TO_TOKEN.items()}

RESIDUE_MASS = {
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032029,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.047680,
    "C": 160.030190,
    "L": 113.084060,
    "I": 113.084060,
    "D": 115.026940,
    "Q": 128.058580,
    "K": 128.094960,
    "E": 129.042590,
    "M": 131.040480,
    "Z": 147.035400,
    "H": 137.058910,
    "F": 147.064410,
    "R": 156.101110,
    "Y": 163.063330,
    "N": 114.042930,
    "W": 186.079310,
}


def normalize_peptide(peptide):
    return "".join(token for token in peptide.upper().replace("I", "L") if token in RESIDUE_MASS)


class BeamSearchDecoder:
    """Decode the final DenNovo probability matrix without inference-time mass constraints."""

    def __init__(self, beam_width=5, topk_per_step=8, min_length=1):
        self.beam_width = max(1, int(beam_width))
        self.topk_per_step = max(1, int(topk_per_step))
        self.min_length = max(0, int(min_length))

    def decode(self, probabilities):
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(VOCABULARY):
            raise ValueError(
                f"Expected a [sequence_length, {len(VOCABULARY)}] probability matrix, "
                f"received {probabilities.shape}."
            )

        log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))
        beams = [(0.0, [])]
        completed = []

        for position in range(1, log_probabilities.shape[0]):
            next_beams = []
            token_ids = np.argsort(log_probabilities[position])[::-1][: self.topk_per_step]
            for score, path in beams:
                for token_id in token_ids:
                    token = ID_TO_TOKEN[int(token_id)]
                    if token in (PAD_TOKEN, START_TOKEN):
                        continue
                    next_score = score + float(log_probabilities[position, token_id])
                    if token == END_TOKEN:
                        if len(path) >= self.min_length:
                            completed.append((next_score, path))
                        continue
                    next_beams.append((next_score, path + [token]))

            if not next_beams:
                break
            next_beams.sort(key=lambda item: item[0], reverse=True)
            beams = next_beams[: self.beam_width]

        candidates = completed or beams
        if not candidates:
            return "", float("-inf")
        candidates.sort(key=lambda item: item[0], reverse=True)
        score, path = candidates[0]
        return "".join(path), score


def count_mass_matches(predicted, target, aa_tolerance=0.1, terminal_tolerance=0.5):
    """Return the maximum residue matches under the paper's mass criteria."""

    predicted = normalize_peptide(predicted)
    target = normalize_peptide(target)
    pred_masses = np.asarray([RESIDUE_MASS[token] for token in predicted], dtype=np.float64)
    target_masses = np.asarray([RESIDUE_MASS[token] for token in target], dtype=np.float64)

    pred_prefix = np.concatenate(([0.0], np.cumsum(pred_masses)))
    target_prefix = np.concatenate(([0.0], np.cumsum(target_masses)))
    pred_total = float(pred_prefix[-1])
    target_total = float(target_prefix[-1])

    rows = len(predicted) + 1
    columns = len(target) + 1
    matches = np.zeros((rows, columns), dtype=np.int32)

    for pred_index in range(1, rows):
        for target_index in range(1, columns):
            best = max(matches[pred_index - 1, target_index], matches[pred_index, target_index - 1])
            aa_close = abs(pred_masses[pred_index - 1] - target_masses[target_index - 1]) < aa_tolerance
            prefix_close = abs(pred_prefix[pred_index - 1] - target_prefix[target_index - 1]) <= terminal_tolerance
            pred_suffix = pred_total - pred_prefix[pred_index]
            target_suffix = target_total - target_prefix[target_index]
            suffix_close = abs(pred_suffix - target_suffix) <= terminal_tolerance
            if aa_close and (prefix_close or suffix_close):
                best = max(best, matches[pred_index - 1, target_index - 1] + 1)
            matches[pred_index, target_index] = best

    return int(matches[-1, -1])


@dataclass
class PaperMetricAccumulator:
    matched_amino_acids: int = 0
    predicted_amino_acids: int = 0
    target_amino_acids: int = 0
    matched_peptides: int = 0
    predicted_peptides: int = 0
    target_peptides: int = 0
    incorrect_peptides: int = 0
    error_bins: dict = field(default_factory=lambda: {"1": 0, "2": 0, ">=3": 0})

    def update(self, predicted, target):
        predicted = normalize_peptide(predicted)
        target = normalize_peptide(target)
        matched = count_mass_matches(predicted, target)

        self.matched_amino_acids += matched
        self.predicted_amino_acids += len(predicted)
        self.target_amino_acids += len(target)
        self.predicted_peptides += int(bool(predicted))
        self.target_peptides += int(bool(target))

        peptide_match = bool(target) and len(predicted) == len(target) and matched == len(target)
        self.matched_peptides += int(peptide_match)
        if target and not peptide_match:
            self.incorrect_peptides += 1
            error_count = max(len(predicted), len(target)) - matched
            if error_count <= 1:
                self.error_bins["1"] += 1
            elif error_count == 2:
                self.error_bins["2"] += 1
            else:
                self.error_bins[">=3"] += 1

    @staticmethod
    def _divide(numerator, denominator):
        return float(numerator) / float(denominator) if denominator else 0.0

    def summary(self):
        error_distribution = {
            name: self._divide(count, self.incorrect_peptides)
            for name, count in self.error_bins.items()
        }
        return {
            "aa_precision": self._divide(self.matched_amino_acids, self.predicted_amino_acids),
            "aa_recall": self._divide(self.matched_amino_acids, self.target_amino_acids),
            "peptide_precision": self._divide(self.matched_peptides, self.predicted_peptides),
            "peptide_recall": self._divide(self.matched_peptides, self.target_peptides),
            "matched_amino_acids": self.matched_amino_acids,
            "predicted_amino_acids": self.predicted_amino_acids,
            "target_amino_acids": self.target_amino_acids,
            "matched_peptides": self.matched_peptides,
            "predicted_peptides": self.predicted_peptides,
            "target_peptides": self.target_peptides,
            "incorrect_peptides": self.incorrect_peptides,
            "error_counts": dict(self.error_bins),
            "error_distribution": error_distribution,
        }
