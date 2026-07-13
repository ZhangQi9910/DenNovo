"""Train DenNovo with a peptide-level 8:1:1 split.

This entry point reuses the canonical architecture, preprocessing pipeline,
losses, metrics, and optimizer configuration from train.py. Keeping one model
implementation prevents training, evaluation, and inference from drifting.
"""

import argparse
import json
import os
from collections import defaultdict

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from denovo_utils import BeamSearchDecoder, PaperMetricAccumulator
from train import (
    Adam,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    data_seq,
    filter_spectra,
    hyper,
    i2l,
    model_builder,
    processor,
    readmgf,
    types,
)


def peptide_key(spectrum):
    peptide = spectrum.get("pep", "")
    if peptide and peptide.isalpha():
        return peptide.upper().replace("I", "L")
    return round(spectrum["mass"], 3), spectrum["charge"]


def get_stratify_tag(spectrum):
    charge_bin = f"C{spectrum['charge']}"

    peptide = spectrum.get("pep", "")
    if peptide and peptide.isalpha():
        peptide_length = len(peptide)
        if peptide_length <= 8:
            length_bin = "L1"
        elif peptide_length <= 12:
            length_bin = "L2"
        elif peptide_length <= 16:
            length_bin = "L3"
        else:
            length_bin = "L4"
    else:
        length_bin = "LNA"

    nce = spectrum.get("nce", -1)
    if nce < 0:
        nce_bin = "NNA"
    elif nce < 20:
        nce_bin = "N1"
    elif nce < 30:
        nce_bin = "N2"
    elif nce < 40:
        nce_bin = "N3"
    else:
        nce_bin = "N4"

    title = spectrum.get("title", "").lower()
    if any(tag in title for tag in ("orbitrap", "q-exactive", "qe", "lumos", "fusion")):
        instrument_bin = "orbitrap"
    elif "tof" in title or "tripletof" in title:
        instrument_bin = "tof"
    else:
        instrument_bin = "INA"

    return f"{charge_bin}_{length_bin}_{nce_bin}_{instrument_bin}"


def stratified_split_peptide_level(
    spectra,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) >= 1e-6:
        raise ValueError("Split ratios must add up to 1.0.")

    rng = np.random.RandomState(seed)
    peptide_to_indices = defaultdict(list)
    for index, spectrum in enumerate(spectra):
        peptide_to_indices[peptide_key(spectrum)].append(index)

    strata = defaultdict(list)
    for key, indices in peptide_to_indices.items():
        strata[get_stratify_tag(spectra[indices[0]])].append(key)

    train_keys = []
    val_keys = []
    test_keys = []
    for keys in strata.values():
        shuffled = np.arange(len(keys))
        rng.shuffle(shuffled)
        train_end = int(len(keys) * train_ratio)
        val_end = train_end + int(len(keys) * val_ratio)
        train_keys.extend(keys[index] for index in shuffled[:train_end])
        val_keys.extend(keys[index] for index in shuffled[train_end:val_end])
        test_keys.extend(keys[index] for index in shuffled[val_end:])

    def collect(keys):
        indices = []
        for key in keys:
            indices.extend(peptide_to_indices[key])
        return [spectra[index] for index in indices]

    train_set = collect(train_keys)
    val_set = collect(val_keys)
    test_set = collect(test_keys)

    train_peptides = set(train_keys)
    val_peptides = set(val_keys)
    test_peptides = set(test_keys)
    if train_peptides & val_peptides or train_peptides & test_peptides or val_peptides & test_peptides:
        raise RuntimeError("Peptide leakage was detected across split subsets.")

    print(
        "[Split] "
        f"spectra={len(spectra)}, unique_peptides={len(peptide_to_indices)}, "
        f"train={len(train_set)}, validation={len(val_set)}, test={len(test_set)}"
    )
    return train_set, val_set, test_set


class SplitTrainManager:
    def __init__(self, output_dir):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def data_generator(self, spectra, **kwargs):
        return data_seq(spectra, processor.process, hyper.dynamic.bsz, xonly=0, **kwargs)

    def prepare_data(self, mgf_path, frag_type="hcd"):
        spectra = i2l(filter_spectra(readmgf(mgf_path, frag_type)))
        if not spectra:
            raise ValueError("No valid spectra remained after preprocessing.")
        self.train_set, self.val_set, self.test_set = stratified_split_peptide_level(spectra, seed=42)
        if not self.train_set or not self.val_set or not self.test_set:
            raise ValueError("At least one split subset is empty. Use a larger or less fragmented dataset.")

    def setup(self, summary=True):
        checkpoint_path = os.path.join(self.output_dir, "denovo_best.weights.h5")
        log_path = os.path.join(self.output_dir, "training_log.csv")
        self.callbacks = [
            ModelCheckpoint(
                checkpoint_path,
                save_best_only=True,
                save_weights_only=True,
                monitor="val_peps_mask_acc",
                mode="max",
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_peps_mask_acc",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                mode="max",
                verbose=1,
            ),
            EarlyStopping(
                monitor="val_peps_mask_acc",
                mode="max",
                patience=6,
                min_delta=1e-4,
                restore_best_weights=True,
                verbose=1,
            ),
            CSVLogger(log_path, separator=",", append=False),
        ]
        self.builder = model_builder(hyper)
        self.model, self.inference_model = self.builder.build(summary=summary)

    def compile(self, batch_size=64, epochs=50, learning_rate=None):
        hyper.dynamic.bsz = int(batch_size)
        hyper.dynamic.eps = int(epochs)
        if learning_rate is None:
            learning_rate = (hyper.dynamic.bsz / 1024.0) * 0.0009 * 16.0 * 6.0
        hyper.dynamic.lr = float(learning_rate)
        hyper.dynamic.opt = Adam(learning_rate=hyper.dynamic.lr)
        self.builder.compile(model=self.model, opt=hyper.dynamic.opt)

    def run(self):
        train_data = self.data_generator(self.train_set, shuffle=1, training=1)
        val_data = self.data_generator(self.val_set, shuffle=0, training=0)
        test_data = self.data_generator(self.test_set, shuffle=0, training=0)

        self.model.fit(
            train_data,
            epochs=hyper.dynamic.eps,
            validation_data=val_data,
            callbacks=self.callbacks,
            verbose=1,
        )

        test_metrics = self.model.evaluate(test_data, verbose=1, return_dict=True)
        serializable_metrics = {name: float(value) for name, value in test_metrics.items()}
        metrics_path = os.path.join(self.output_dir, "test_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(serializable_metrics, handle, indent=2, sort_keys=True)
        print(f"[Test] Metrics written to {metrics_path}")

        prediction_data = data_seq(
            self.test_set,
            processor.process,
            hyper.dynamic.bsz,
            shuffle=0,
            xonly=1,
            training=0,
        )
        probabilities = self.inference_model.predict(prediction_data, verbose=1)
        decoder = BeamSearchDecoder(beam_width=5)
        paper_metrics = PaperMetricAccumulator()
        for spectrum, probability in zip(self.test_set, probabilities):
            predicted, _ = decoder.decode(probability)
            paper_metrics.update(predicted, spectrum["pep"])
        paper_metrics_path = os.path.join(self.output_dir, "test_paper_metrics.json")
        with open(paper_metrics_path, "w", encoding="utf-8") as handle:
            json.dump(paper_metrics.summary(), handle, indent=2, sort_keys=True)
        print(f"[Test] Paper-aligned metrics written to {paper_metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Train DenNovo with a peptide-level 8:1:1 split.")
    parser.add_argument("--mgf", required=True, help="Input MGF containing peptide labels")
    parser.add_argument("--frag", default="hcd", choices=list(types.keys()), help="Fragmentation type")
    parser.add_argument("--batch-size", "--bsz", dest="batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=None)
    parser.add_argument("--output-dir", default="output_split811")
    args = parser.parse_args()

    manager = SplitTrainManager(args.output_dir)
    manager.prepare_data(args.mgf, args.frag)

    gpu_count = len(tf.config.list_physical_devices("GPU"))
    strategy = tf.distribute.MirroredStrategy() if gpu_count > 1 else tf.distribute.get_strategy()
    print(f"[Runtime] replicas={strategy.num_replicas_in_sync}")
    with strategy.scope():
        manager.setup(summary=True)
        manager.compile(args.batch_size, args.epochs, args.learning_rate)
    manager.run()


if __name__ == "__main__":
    main()
