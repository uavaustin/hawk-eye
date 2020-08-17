#!/usr/bin/env python3
""" A script to benchmark inference for both detector and classification models.
This is useful for seeing how a new model performs on a device. For example, this
can be run on the Jetson to see how the models perform. """

import argparse
import pathlib
import time

import torch
import yaml

from core import classifier
from core import detector
from core import pull_assets


@torch.no_grad()
def benchmark(
    timestamp: str, model_type: str, batch_size: int, rum_time: float = 30.0
) -> None:

    # Construct the model.
    if model_type == "classifier":
        model = classifier.Classifier(timestamp=timestamp, half_precision=True)
    elif model_type == "detector":
        model = detector.Detector(timestamp=timestamp, half_precision=True)

    batch = torch.randn((batch_size, 3, model.image_size, model.image_size))

    if torch.cuda.is_available():
        model.cuda()
        model.half()
        batch = batch.cuda().half()

    print("Starting inference.")
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        model(batch)
        times.append(time.perf_counter() - start)

    latency = sum(times) / len(times)

    print(
        f"Total time: {sum(times):.4f}.\n"
        f"Average batch inference time: {latency:.4f}s. FPS: {batch_size / latency:.2f}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model benchmark script.")
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()

    benchmark(args.timestamp, args.model_type, args.batch_size)
