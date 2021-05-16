"""

PYTHONPATH=. python3 hawk_eye/train/calculate_f1_score.py \
    --model_type classifier \
    --timestamp 2020-09-05T15.51.57 \
    --datasets "competition_2019_20210318,competition-targets-2018,test_flight_20210319"

PYTHONPATH=. python3 hawk_eye/train/calculate_f1_score.py \
    --model_type detector \
    --timestamp 2020-10-10T14.02.09 \
    --datasets "competition_2019_20210318,competition-targets-2018,test_flight_20210319"

"""

import argparse
import pathlib
from typing import List
from hawk_eye.core import classifier
from hawk_eye.core import detector

import numpy as np
import torch
from torch import cuda
from torch.utils import data

from hawk_eye.core import classifier
from hawk_eye.core import detector
from hawk_eye.data_generation import generate_config
from hawk_eye.train.detection import dataset as det_dataset
from hawk_eye.train.detection import collate
from third_party.detectron2 import pascal_voc


def calculate_f1_score(precision, recall, beta: float = 1.2) -> int:

    """ This function finds the F1 score for a model.

    Args:
        precision: True Positives / (True Positives + False Positives) of a model
        recall: True Positives / (True Positives + False Negatives) of a model
        beta: A value such that recall is beta times more important as precision

    Returns:
        f1_score: A measure of a model's accuracy

    Examples::

        >>> calculate_f1_score(.8, .5, 2)
        .54054054
        >>> calculate_f1_score(.3, .4, 1)
        .34285714

    """
    if recall == 0:
        recall = 1.0e-8
    f1_score = (1 + (beta ** 2)) * (
        (precision * recall) / (((beta ** 2) * precision) + recall)
    )

    return f1_score


def load_model(model_type: str, timestamp: str):
    if model_type == "classifier":
        model = load_classifier(timestamp)
    elif model_type == "detector":
        model = load_detector(timestamp)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model


def main(model_type, timestamp: str, datasets: List[str]):

    model = load_model(model_type, timestamp)
    print(model)

    return model


def load_classifier(timestamp: str):
    model = classifier.Classifier(
        timestamp=timestamp, half_precision=cuda.is_available()
    )
    model.eval()
    return model


def load_detector(timestamp: str):
    model = detector.Detector(
        timestamp=timestamp, confidence=0.05, half_precision=cuda.is_available(),
    )
    model.eval()
    return model


def load_data(model_type: str, datasets: List[pathlib.Path]):

    if model_type == "detector":
        # Force datasets to use validation.
        image_dirs = [data_dir / "val" for data_dir in datasets]
        metadata_paths = [data_dir / "val_coco.json" for data_dir in datasets]
        dataset = det_dataset.DetDataset(image_dirs, metadata_paths, validation=True)

    return dataset


def run_model(model, data, model_type):
    if model_type == "classifier":
        output = run_classifier(model, data)
    elif model_type == "detector":
        output = run_detector(model, data)

    return output


@torch.no_grad()
def run_detector(model: detector.Detector, dataset: det_dataset.DetDataset):
    loader = data.DataLoader(
        dataset,
        batch_size=4,
        pin_memory=True,
        collate_fn=collate.CollateVal(),
        num_workers=0,
        drop_last=False,
        shuffle=False,
    )
    detections = []
    labels = []
    for images_batch, category_ids_batch, boxes_batch in loader:

        # Send ground truth to BoundingBox
        for boxes, categories in zip(boxes_batch, category_ids_batch):
            image_boxes = []
            for box, category in zip(boxes, categories.squeeze(0)):
                image_boxes.append(
                    pascal_voc.BoundingBox(box, 1.0, category.int().item())
                )

            labels.append(image_boxes)

        if cuda.is_available():
            images_batch = images_batch.cuda()

        detections.extend(model.get_boxes(images_batch))

    return detections, labels


def run_classifier(model, data):
    ...


def calculate_metrics(output, labels, model_type, model):
    if model_type == "classifier":
        ...
    elif model_type == "detector":

        metrics = pascal_voc.compute_metrics(
            output, labels, class_ids=list(range(model.num_classes))
        )
        precision = metrics["ap30"]
        recall = metrics["rec30"]

    return precision, recall


def filter_results(output, confidence: float, model_type: str):
    if model_type == "detector":
        filtered_detections = []
        for batch in output:
            new_batch = []
            for box in batch:
                if box.confidence > confidence:
                    new_batch.append(box)

            filtered_detections.append(new_batch)

    return filtered_detections


def main(model_type: str, timestamp: str, datasets: List[str]):

    # Load the model
    model = load_model(model_type, timestamp)

    # Load the data
    data = load_data(model_type, datasets)

    # Get the model's output
    detections, labels = run_model(model, data, model_type)

    f1_scores = {}
    precision_recalls = {}
    for confidence in range(1, 100):

        new_output = filter_results(detections, confidence / 100, model_type)
        precision, recall = calculate_metrics(new_output, labels, model_type, model)
        f1_score = calculate_f1_score(precision, recall)
        f1_scores[f1_score] = confidence
        precision_recalls[f1_score] = {"precision": precision, "recall": recall}

    max_f1_score = max(f1_scores)
    print(f"Recommended confidence: {f1_scores[max_f1_score]}.")
    print(precision_recalls[max_f1_score])


if __name__ == "__main__":
    torch.cuda.random.manual_seed(42)
    torch.random.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(description="Find a model's F1 score.")
    parser.add_argument(
        "--model_type", choices=["classifier", "detector"], required=True, type=str
    )
    parser.add_argument(
        "--timestamp", required=True, type=str,
    )
    parser.add_argument(
        "--datasets",
        required=True,
        type=str,
        help="A comma separated string of datasets.",
    )
    args = parser.parse_args()

    datasets = [
        generate_config.DATA_DIR / dataset for dataset in args.datasets.split(",")
    ]
    main(args.model_type, args.timestamp, datasets)
