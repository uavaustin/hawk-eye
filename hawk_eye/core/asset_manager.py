"""Functions to pull in data from the cloud.

This relies on Google Cloud Storage python API. Please see the Image Recognition lead
to recieve the proper credentials for access to the bucket.

If you do not have Google Cloud APIs installed please run:

.. code-block::
    hawk_eye/setup/install_google_cloud.sh

"""

import tarfile
import pathlib
import tempfile
import subprocess
from typing import List, Union

from google.cloud import storage

from hawk_eye.data_generation import generate_config as config

BUCKET = "uav_austin"


def pull_all() -> None:
    """Pull thee backgrounds, base shapes and font files as specified in
    `hawk_eye.data_generation.generate_config.py`."""
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


def upload_file(source_path: pathlib.Path, destination: str):
    """A generic function for uploading a file from local storage to the cloud.

    Args:
        source_path: a path to the local file
        desination: where to upload the file
    """

    if not source_path.is_file():
        raise FileNotFoundError(f"Can't find {source_path}")

    bucket = _get_client_bucket()
    blob = bucket.blob(destination)
    blob.upload_from_filename(str(source_path))


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
                client = storage.Client()
                bucket = client.get_bucket(BUCKET)
                bucket.get_blob(str(filename)).download_to_filename(tmp_file)
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
    upload_file(path, f"{model_type}/{path.name}")


def _get_client_bucket():
    client = storage.Client()
    return client.get_bucket(BUCKET)
