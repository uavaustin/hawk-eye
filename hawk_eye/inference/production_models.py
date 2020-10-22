#!/usr/bin/env python3

import argparse

_CLASSIFIER = {
    "timestamp": "2020-09-05T15.51.57",
    "sha256": "a67421a89f06619d187dec2c821e219a0c6614203fd745802081481c8ad9e656",
}
_DETECTOR = {
    "timestamp": "2020-10-10T14.02.09",
    "sha256": "4443ff284576d6c4dabc3c6d12cb8724c8ca49322e26e5de6ccb5a9751bd2819",
}
_PROD_MODELS = {
    "classifier": _CLASSIFIER,
    "detector": _DETECTOR,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple model selection file.")
    parser.add_argument(
        "--model_type", type=str, choices=["classifier", "detector"], required=True,
    )
    args = parser.parse_args()

    for key, val in _PROD_MODELS[args.model_type].items():
        print(f"{key}: {val}")
