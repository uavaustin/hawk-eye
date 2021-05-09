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
import torch

from typing import List
from hawk_eye.core import classifier
from hawk_eye.core import detector


def calculate_f1_score(precision, recall, beta: float) -> int:

    """ This function finds the F1 score for a model.

    Examples::

        >>> calculate_f1_score(.8, .5, 2)
        .54054054
        >>> calculate_f1_score(.3, .4, 1)
        .34285714

    """

    f1_score = (1 + (beta ** 2)) * (
        (precision * recall) / (((beta ** 2) * precision) + recall)
    )

    return f1_score


def load_model(entered_model: str):

    if entered_model == "classifier":
        model_type = "classifier"
    elif entered_model == "detector":
        model_type = "detector"
    else:
        raise ValueError(f"Unsupported model type: {entered_model}")

    return model_type


def load_classifier(model_type, timpestamp: str, datasets: List[str]):
    if model_type == "classifier":
        clf_model = classifier.Classifier(
            timestamp=timestamp, half_precision=torch.cuda.is_available()
        )
        clf_model.eval()
        print(clf_model)


def load_detector(model_type, timestamp: str, datasets: List[str]):
    if model_type == "detector":
        det_model = detector.Detector(
            timestamp=timestamp,
            confidence=0.05,
            half_precision=torch.cuda.is_available(),
        )
        det_model.eval()
        print(det_model)


if __name__ == "__main__":
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

    datasets = [dataset for dataset in args.datasets.split(",")]
    main(args.model_type, args.timestamp, datasets)
