#!/usr/bin/env python3
import argparse
import dataclasses
import pathlib
import json
import tempfile
from typing import List

import cv2
import torch

from hawk_eye.core import classifier
from hawk_eye.core import detector
from hawk_eye.train import augmentations
from hawk_eye.train import collate
from hawk_eye.train import datasets
from hawk_eye.train import train_det
from third_party import coco_eval


@dataclasses.dataclass
class ClassificationObject:
    image_path: pathlib.Path
    image_class: int
    image_id: int


def load_model(model_timestamp: str, model_type: str) -> torch.nn.Module:

    if model_type == "classifier":
        model = classifier.Classifier(
            timestamp=model_timestamp, half_precision=torch.cuda.is_available()
        )
        model.eval()
    elif model_type == "detector":
        # TODO(alex): Pass in the confidence for the detector.
        model = detector.Detector(
            timestamp=model_timestamp,
            confidence=0.0,
            half_precision=torch.cuda.is_available(),
        )
        model.eval()

    return model


@torch.no_grad()
def inference_clf_dataset(
    model: classifier.Classifier, labels: List[ClassificationObject]
) -> float:

    augs = augmentations.clf_eval_augs(model.image_size, model.image_size)
    num_correct = 0
    for label in labels:
        image = cv2.imread(str(label.image_path))
        image = torch.Tensor(augs(image=image)["image"])
        # HWC -> BHWC -> BCHW
        image = image.unsqueeze(0).permute(0, 3, 1, 2)

        results = model.classify(image, probability=True)
        _, predicted = torch.max(results.data, 1)
        num_correct += (predicted == label.image_class).item()

    return num_correct / len(labels)


@torch.no_grad()
def inference_det_dataset(
    model: detector.Detector,
    eval_loader: torch.utils.data.DataLoader,
    coco_json: pathlib.Path,
):
    detections_dict: List[dict] = []
    for images, image_ids in eval_loader:
        if torch.cuda.is_available():
            images = images.cuda()
        detections = model(images)

        detections_dict.extend(
            train_det.detections_to_dict(detections, image_ids, model.image_size)
        )
    results = {}
    if detections_dict:

        with tempfile.TemporaryDirectory() as d:
            tmp_json = pathlib.Path(d) / "det.json"
            tmp_json.write_text(json.dumps(detections_dict))
            results = coco_eval.get_metrics(coco_json, tmp_json)

    return results


def inference_dataset(model_timestamp: str, model_type: str, dataset: pathlib.Path):

    model = load_model(model_timestamp, model_type)

    if model_type == "classifier":
        labels = prepare_clf_dataset(dataset)
        metrics = inference_clf_dataset(model, labels)
    elif model_type == "detector":
        loader = prepare_det_dataset(dataset)
        metrics = inference_det_dataset(model, loader, dataset / "val_coco.json")

    return metrics


def prepare_clf_dataset(dataset: pathlib.Path):
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


def prepare_det_dataset(dataset: pathlib.Path):
    dataset = datasets.DetDataset(
        dataset / "images", dataset / "val_coco.json", validation=True, img_ext=".JPG"
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate.CollateVal(),
        num_workers=max(torch.multiprocessing.cpu_count(), 4),
    )
    return loader


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
