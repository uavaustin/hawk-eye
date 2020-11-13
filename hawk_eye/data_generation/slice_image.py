#!/usr/bin/env python3
""" Contains code to slice up an image into smaller tiles. """

import argparse
import pathlib
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from hawk_eye.data_generation import generate_config as config


def slice_image(
    images: List, tile_size: Tuple[int, int], overlap: int, save_dir: pathlib.Path
) -> None:
    """ Take in an image and slice it into smaller images.
    Args:
        images: A list of paths to images to be sliced.
        tile_size: The (width, height) of the tiles.
        overlap: The overlap between adjacent tiles.
        save_dir: The path to the directory within which to save slices.
    Returns:
        None.
    """
    for filename in tqdm(images, desc="Slicing images", total=len(images)):
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

                tile.save(save_dir / f"{filename.stem}-{x}-{y}.JPG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice an image into smaller images.")
    parser.add_argument(
        "--image_dir",
        type=pathlib.Path,
        required=False,
        help="Path to directory of images to be sliced.",
    )
    parser.add_argument(
        "--image_path",
        type=pathlib.Path,
        required=False,
        help="Path to an image to be sliced.",
    )
    parser.add_argument(
        "--image_extension",
        type=str,
        required=False,
        default=".JPG",
        help="Extension of images to slice.",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
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

    # Raises an error if none of image arguments are provided
    if args.image_path is None and args.image_dir is None:
        raise ValueError("Please supply either an image or directory of images.")

    # If the input image extension doesn't start with '.', a '.' is added
    if args.image_extension[0] != ".":
        image_ext = "." + args.image_extension
    else:
        image_ext = args.image_extension

    if args.image_path is not None:
        if args.image_path.suffix.upper() == image_ext.upper():
            images = [args.image_path.expanduser()]
    elif args.image_dir is not None:
        # If image_dir points to a directory, paths to all the images with the correct
        # extension are put in a list
        if args.image_dir.is_dir():
            images = list(
                args.image_dir.expanduser().glob(f"*{image_ext.lower()}")
            ) + list(args.image_dir.expanduser().glob(f"*{image_ext.upper()}"))
        else:
            raise ValueError("Please supply a valid path to a directory.")

    # Makes a directory to save slices in
    save_dir = args.save_dir.expanduser()
    save_dir.mkdir(exist_ok=True, parents=True)

    slice_image(
        images=images,
        tile_size=config.CROP_SIZE,
        overlap=args.overlap,
        save_dir=save_dir,
    )
