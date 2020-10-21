#!/usr/bin/env python3

_PROD_MODELS = {
    "classifier": "2020-09-05T15.51.57",
    "detector": "2020-10-10T14.02.09",
}

if __name__ == "__main__":
    for key, val in _PROD_MODELS.items():
        print(f"{key}: {val}")
