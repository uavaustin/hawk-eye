#!/usr/bin/env python3
"""Packaging around the production models. This needs to be updated with any new models
for release. This file is also called by some bazel rules."""

import argparse

_CLASSIFIER = {
    "timestamp": "2020-09-05T15.51.57",
    "sha256": "b4f5ddb23475ce9662dae6e710bfab6bf8edf658a8b9b5c74386dfcef29d21f2",
}
_DETECTOR = {
    "timestamp": "2020-10-10T14.02.09",
    "sha256": "12e880eae372fb7fa7f3b42362c9d17c6b356456a13e41579410eb3086c9ae6b",
}
PROD_MODELS = {
    "classifier": _CLASSIFIER,
    "detector": _DETECTOR,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple model selection file.")
    parser.add_argument(
        "--model_type", type=str, choices=["classifier", "detector"], required=True,
    )
    args = parser.parse_args()

    for key, val in PROD_MODELS[args.model_type].items():
        print(f"{key}: {val}")
