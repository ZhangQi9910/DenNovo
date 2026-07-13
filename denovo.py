"""Run deterministic de novo peptide sequencing with a trained DenNovo model."""

import argparse
import csv
import os

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from denovo_utils import BeamSearchDecoder
from train import data_seq, hyper, i2l, model_builder, processor, readmgf, types


def configure_runtime():
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def prepare_unlabeled_spectra(mgf_path, frag_type):
    spectra = readmgf(mgf_path, frag_type)
    for spectrum in spectra:
        spectrum["pep"] = "A"
        spectrum["mod"] = np.zeros(1, dtype=np.int32)
        spectrum["nmod"] = 0
    return i2l(spectra)


def run_denovo(mgf_path, model_path, output_path, frag_type="hcd", batch_size=128, beam_width=5):
    configure_runtime()
    spectra = prepare_unlabeled_spectra(mgf_path, frag_type)
    if not spectra:
        raise ValueError("The input MGF did not contain any readable spectra.")

    hyper.dynamic.bsz = int(batch_size)
    builder = model_builder(hyper)
    full_model, inference_model = builder.build(summary=False)
    full_model.load_weights(model_path)

    generator = data_seq(
        spectra,
        processor.process,
        batch_size,
        shuffle=0,
        xonly=1,
        training=0,
    )
    probabilities = inference_model.predict(generator, verbose=1)
    decoder = BeamSearchDecoder(beam_width=beam_width)

    output_path = os.path.abspath(output_path)
    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "title", "precursor_mz", "charge", "peptide", "score"])
        for index in tqdm(range(len(spectra)), desc="Decoding"):
            peptide, score = decoder.decode(probabilities[index])
            spectrum = spectra[index]
            writer.writerow(
                [
                    index,
                    spectrum.get("title", ""),
                    f"{float(spectrum['mass']):.6f}",
                    int(spectrum["charge"]),
                    peptide,
                    f"{score:.6f}",
                ]
            )

    print(f"Predictions written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run DenNovo peptide sequencing.")
    parser.add_argument("--mgf", required=True, help="Input MGF")
    parser.add_argument("--model", required=True, help="DenNovo .weights.h5 checkpoint")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--frag", default="hcd", choices=list(types.keys()), help="Fragmentation type")
    parser.add_argument("--batch-size", "--bsz", dest="batch_size", type=int, default=128)
    parser.add_argument("--width", "--beam", dest="beam_width", type=int, default=5)
    args = parser.parse_args()

    run_denovo(
        args.mgf,
        args.model,
        args.output,
        frag_type=args.frag,
        batch_size=args.batch_size,
        beam_width=args.beam_width,
    )


if __name__ == "__main__":
    main()
