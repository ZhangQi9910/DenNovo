"""Evaluate DenNovo with the mass-based criteria defined in the manuscript."""

import argparse
import csv
import json
import os

from tqdm import tqdm

from denovo_utils import BeamSearchDecoder, PaperMetricAccumulator, normalize_peptide
from train import data_seq, filter_spectra, hyper, i2l, model_builder, processor, readmgf, types


def run_evaluation(
    mgf_path,
    model_path,
    frag_type="hcd",
    batch_size=128,
    beam_width=5,
    metrics_output=None,
    predictions_output=None,
):
    spectra = i2l(filter_spectra(readmgf(mgf_path, frag_type)))
    if not spectra:
        raise ValueError("No labeled spectra remained after preprocessing and the 10 ppm filter.")

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
    metrics = PaperMetricAccumulator()
    prediction_rows = []

    for index in tqdm(range(len(spectra)), desc="Evaluating"):
        predicted, score = decoder.decode(probabilities[index])
        target = normalize_peptide(spectra[index]["pep"])
        metrics.update(predicted, target)
        prediction_rows.append(
            {
                "index": index,
                "title": spectra[index].get("title", ""),
                "target": target,
                "predicted": predicted,
                "score": score,
            }
        )

    summary = metrics.summary()
    print(json.dumps(summary, indent=2, sort_keys=True))

    if metrics_output:
        metrics_output = os.path.abspath(metrics_output)
        os.makedirs(os.path.dirname(metrics_output) or ".", exist_ok=True)
        with open(metrics_output, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(f"Metrics written to {metrics_output}")

    if predictions_output:
        predictions_output = os.path.abspath(predictions_output)
        os.makedirs(os.path.dirname(predictions_output) or ".", exist_ok=True)
        with open(predictions_output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["index", "title", "target", "predicted", "score"],
            )
            writer.writeheader()
            writer.writerows(prediction_rows)
        print(f"Predictions written to {predictions_output}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate DenNovo on a labeled MGF file.")
    parser.add_argument("--mgf", required=True, help="Labeled validation or test MGF")
    parser.add_argument("--model", required=True, help="DenNovo .weights.h5 checkpoint")
    parser.add_argument("--frag", default="hcd", choices=list(types.keys()), help="Fragmentation type")
    parser.add_argument("--batch-size", "--bsz", dest="batch_size", type=int, default=128)
    parser.add_argument("--width", "--beam", dest="beam_width", type=int, default=5)
    parser.add_argument("--metrics-output", default=None, help="Optional JSON metrics path")
    parser.add_argument("--predictions-output", default=None, help="Optional CSV prediction path")
    args = parser.parse_args()

    run_evaluation(
        args.mgf,
        args.model,
        frag_type=args.frag,
        batch_size=args.batch_size,
        beam_width=args.beam_width,
        metrics_output=args.metrics_output,
        predictions_output=args.predictions_output,
    )


if __name__ == "__main__":
    main()
