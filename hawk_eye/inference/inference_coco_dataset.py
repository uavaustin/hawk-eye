#!/usr/bin/env python3
import argparse
import dataclasses
import pathlib
import json

import cv2
import torch

from hawk_eye.core import classifier
from hawk_eye.core import detector
from hawk_eye.train import augmentations

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


@dataclasses.dataclass
class ClassificationObject:
    image_path: pathlib.Path
    image_class: int
    image_id: int


def load_model(model_timestamp, model_type):

    if model_type == "classifier":
        model = classifier.Classifier(
            timestamp=model_timestamp, half_precision=torch.cuda.is_available()
        )
        model.eval()
    elif model_type == "detector":
        # TODO(alex): Pass in the confidence for the detector.
        model = detector.Detector(
            timestamp=model_timestamp,
            confidence=0.2,
            half_precision=torch.cuda.is_available(),
        )
        model.eval()

    return model


@torch.no_grad()
def inference_dataset(model_timestamp, model_type, dataset):
    labels = prepare_dataset(dataset)

    model = load_model(model_timestamp, model_type)
    augs = augmentations.clf_eval_augs(model.image_size, model.image_size)

    num_correct = 0
    for label in labels:
        image = cv2.imread(str(label.image_path))
        image = torch.Tensor(augs(image=image)["image"])
        image = image.unsqueeze(0).permute(0, 3, 1, 2)

        results = model.classify(image, probability=True)
        _, predicted = torch.max(results.data, 1)
        num_correct += (predicted == label.image_class).sum().item()

    return num_correct / len(labels)


def prepare_dataset(dataset: pathlib.Path):
    coco_json_data = json.loads((dataset / "val_coco.json").read_text())
    all_images = coco_json_data.get("images", [])
    annotations = coco_json_data.get("annotations", [])
    images_with_objects = set([annotation["image_id"] for annotation in annotations])

    labels = []
    for image in all_images:
        image_path = dataset / "images" / image["file_name"]

        # If it is a target, class_id = 1
        if image["id"] in images_with_objects:
            labels.append(ClassificationObject(image_path, 1, image["id"]))
        else:
            # If not a target, background
            labels.append(ClassificationObject(image_path, 0, image["id"]))

    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference COCO dataset.")
    parser.add_argument(
        "--model_type", type=str, help="Path to an image to inference.",
    )
    parser.add_argument("--model_timestamp", type=str, help="Timestamp of model used.")
    parser.add_argument(
        "--dataset", type=pathlib.Path, help="Path to dataset to perform inference on."
    )
    args = parser.parse_args()

    metrics = inference_dataset(args.model_timestamp, args.model_type, args.dataset)
    print(metrics)
