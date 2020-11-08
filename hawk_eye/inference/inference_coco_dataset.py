#!/usr/bin/env python3
import argparse
import pathlib
import json

import torch

"""

    * Metrics generated will differ for detection vs classification.
    * Focus on classification (it's easier)

    * What are scoring the models on?
        - classification: accuracy (TP / (TP + FN))

    1 How to load COCO dataset.
        - read *.json file and process the images/labels
        - you can associate labels with images (target vs background)

    2 Loading the models
        - take in user timestamp and load the model (on gpu? cpu?)

    3 Combine the loaded dataset with model to get predictions

    4 Do something with the predictions. Generate the accuracy

"""


def inference_dataset(model_timestamp, model_type, dataset):
    ...


def prepare_dataset(dataset: pathlib.Path):
    coco_json_data = json.loads((dataset / "val_coco.json").read_text())
    all_images = coco_json_data.get("images", [])
    print(all_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference COCO dataset.")

    # create arg parser that can take in a variable w/model type
    # build argument parser and try to take in the argument for the model timestamp

    parser.add_argument(
        "--model_type", type=str, help="Path to an image to inference.",
    )

    parser.add_argument("--model_timestamp", type=str, help="Timestamp of model used.")

    parser.add_argument(
        "--dataset", type=pathlib.Path, help="Path to dataset to perform inference on."
    )

    args = parser.parse_args()

    prepare_dataset(args.dataset.expanduser())

    # inference_dataset(args.model_timestamp, args.model_type, args.dataset)
