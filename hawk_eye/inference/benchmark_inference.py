#!/usr/bin/env python3
""" A script to benchmark inference for both detector and classification models.
This is useful for seeing how a new model performs on a device. For example, this
can be run on the Jetson to see how the models perform. """

import argparse
import time

import torch

from hawk_eye.core import classifier
from hawk_eye.core import detector


@torch.no_grad()
def benchmark(
    timestamp: str, model_type: str, batch_size: int, run_time: float
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
    start_loop = time.perf_counter()
    times = []
    while time.perf_counter() - start_loop < run_time:
        start = time.perf_counter()
        model(batch)
        times.append(time.perf_counter() - start)

    latency = sum(times) / len(times)

    print(
        f"Total time: {sum(times):.4f}.\n"
        f"Average batch inference time: {latency:.4f}s. FPS: {batch_size / latency:.2f}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model benchmark script using FP16 if CUDA is available."
    )
    parser.add_argument(
        "--timestamp", type=str, required=True, help="Model timestamp to test."
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="Why model type to test."
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="How many images per batch."
    )
    parser.add_argument(
        "--run_time", type=float, default=30.0, help="The duration of the inferencing."
    )
    args = parser.parse_args()

    benchmark(args.timestamp, args.model_type, args.batch_size, args.run_time)
