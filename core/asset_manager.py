#!/usr/bin/env python3
""" Functions to pull in data from the cloud.
This relies on Google Cloud Storage python API. Please see the Image Recognition lead
to recieve the proper credentials for access to the bucket. """

import tarfile
import pathlib
import tempfile
import subprocess
from typing import List, Union

from data_generation import generate_config as config

_BUCKET = "gs://uav-austin-test"


def pull_all() -> None:
    """ Pull all assets. """
    pull_backgrounds()
    pull_base_shapes()
    pull_fonts()


def pull_backgrounds() -> None:
    """Pull the shape generation backgrounds."""
    download_file(config.BACKGROUNDS_URLS, config.ASSETS_DIR)


def pull_base_shapes() -> None:
    """Pull the base shape images."""
    download_file(config.BASE_SHAPES_URL, config.ASSETS_DIR)


def pull_fonts() -> None:
    """Pull the fonts."""
    download_file(config.FONTS_URL, config.ASSETS_DIR)


# Download a file to the assets folder and return the filename.
def download_file(filenames: Union[str, List[str]], destination: pathlib.Path) -> None:

    if isinstance(filenames, str):
        filenames = [filenames]

    for filename in filenames:
        filename = pathlib.Path(filename)
        folder_name = filename.stem.split(".", 1)[0]
        if not (destination / folder_name).is_dir():
            print(f"Fetching {filename}...", end="", flush=True)

            with tempfile.TemporaryDirectory() as d:
                tmp_file = pathlib.Path(d) / "file.tar.gz"
                subprocess.check_call(
                    ["gsutil", "cp", f"{_BUCKET}/{filename}", str(tmp_file)]
                )

                untar_and_move(tmp_file, destination)

            print(" done.")


# Untar a file, unless the directory already exists.
def untar_and_move(filename: pathlib.Path, destination: pathlib.Path) -> None:

    print(f"Extracting to {destination}...", end="", flush=True)
    with tarfile.open(filename, "r") as tar:
        tar.extractall(destination)

    # Remove hidden files that might have been left behind by
    # the untarring.
    for filename in destination.rglob("._*"):
        filename.unlink()


def download_model(model_type: str, timestamp: str) -> pathlib.Path:
    assert model_type in ["classifier", "detector"], f"Unsupported model {model_type}."
    filename = f"{model_type}/{timestamp}.tar.gz"
    destination = pathlib.Path(f"~/runs/uav-{model_type}").expanduser() / timestamp

    if not destination.is_dir():
        download_file(filename, destination)

    return destination


def upload_model(model_type: str, path: pathlib.Path) -> None:
    subprocess.check_call(
        ["gsutil", "cp", str(path), f"{_BUCKET}/{model_type}/{path.name}"]
    )
