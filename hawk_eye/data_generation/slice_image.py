#!/usr/bin/env python3
""" Contains code to slice up an image into smaller tiles. """

import argparse
import pathlib
import pdb
import sys
from typing import Tuple

import numpy as np
from PIL import Image

from hawk_eye.data_generation import generate_config as config


def slice_image(
    img_dir: str, tile_size: Tuple[int, int], overlap: int, save_dir: str
) -> None:
    """ Take in an image and slice it into smaller images.
    Args:
        img_dir: The path to the directory with images to be sliced.
        tile_size: The (width, height) of the tiles.
        overlap: The overlap between adjacent tiles.
        save_dir: The path to the directory within which to save slices.
    Returns:
        None.
    Usage:
        >>> slice_image('hawk-eye/hawk_eye/data_generation/data/test_flight_targets_20190215', (512, 512), 50, 'hawk_eye/data_generation/data/slices')
    """
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    for filename in pathlib.Path.iterdir(pathlib.Path(img_dir)):
        if filename.suffix.upper().endswith(".JPG"):
            image = Image.open(filename)
            width, height = image.size

            # Cropping logic repurposed from hawk_eye.inference.find_targets.tile_image()
            for x in range(0, width, tile_size[0] - overlap):

                # Shift back to extract tiles on the image
                if x + tile_size[0] >= width and x != 0:
                    x = width - tile_size[0]

                for y in range(0, height, tile_size[1] - overlap):

                    if y + tile_size[1] >= height and y != 0:
                        y = height - tile_size[1]

                    tile = image.crop((x, y, x + tile_size[0], y + tile_size[1]))

                    tile.save(
                        f"{pathlib.Path(save_dir).joinpath(filename.stem)}-{x}-{y}.JPG"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice an image into smaller images.")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to directory with images to be sliced.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to directory in which to store sliced images.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        required=True,
        default=config.CROP_OVERLAP,
        help="Number of pixels of overlap between adjacent slices.",
    )
    args = parser.parse_args()

    slice_image(
        img_dir=args.image_dir,
        tile_size=config.CROP_SIZE,
        overlap=args.overlap,
        save_dir=args.save_dir,
    )
