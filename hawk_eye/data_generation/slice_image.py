#!/usr/bin/env python3
"""Contains code to slice up an image into smaller tiles.

PYTHONPATH=. hawk_eye/data_generation/slice_image.py \
    --image_dir hawk_eye/data_generation/assets/competition-2019 \
    --image_extensions .jpg \
    --save_dir hawk_eye/data_generation/assets/competition-2019-tiles


PYTHONPATH=. hawk_eye/data_generation/slice_image.py \
    --image_path hawk_eye/data_generation/assets/competition-2019/image-001642.jpg \
    --image_extensions .jpg \
    --save_dir hawk_eye/data_generation/assets/competition-2019-tiles

"""

__author__ = "Kadhir Umasankar"

import argparse
import pathlib
import time
from typing import List, Tuple

import numpy as np
from PIL import Image
import tqdm

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
    tile_times = []
    # TODO(anya): replace with perf_counter()
    start_time = time.perf_counter()
    tile_count = 0
    tile_names = []
    for filename in tqdm.tqdm(images, desc="Slicing images", total=len(images)):

        image = Image.open(filename)
        width, height = image.size
        print(width, height)
        # Cropping logic repurposed from hawk_eye.inference.find_targets.tile_image()
        for x in range(0, width - overlap, tile_size[0] - overlap):
            print(x)
            # Shift back to extract tiles on the image
            if x + tile_size[0] > width and x != 0:
                x = width - tile_size[0]
                print(x, "offset")

            for y in range(0, height - overlap, tile_size[1] - overlap):

                if y + tile_size[1] > height and y != 0:
                    y = height - tile_size[1]

                tile = image.crop((x, y, x + tile_size[0], y + tile_size[1]))

                tile.save(save_dir / f"{filename.stem}-{x}-{y}{filename.suffix}")
                tile_names.append(
                    save_dir / f"{filename.stem}-{x}-{y}{filename.suffix}"
                )
                tile_count += 1
                end_time = time.perf_counter()
                total_time = end_time - start_time
                tile_times.append(total_time)

    print(len(tile_names))
    print(len(set(tile_names)))
    for tile_name in tile_names:
        if tile_names.count(tile_name) > 1:
            print(tile_name)

    print("Average Time to Slice an Image = ", ((sum(tile_times) / len(tile_times))))
    print(f"Total Time to Slice All Images = {sum(tile_times)}")
    print(f"Number of Original Images = {len(images)}")
    tile_count2 = list(save_dir.glob("*"))
    print(
        f"Total Number of Tiles After Slicing All Images = {tile_count}, or {len(tile_count2)}"
    )

    # print(f"Average Time to Slice an Image = {sum(tile_times) / len(tile_times):.4f}")
    # Print total time, total number of original images, how many tiles

    (save_dir / "labels.txt").write_text("\n".join(config.SHAPE_TYPES))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--image_extensions",
        type=str,
        required=False,
        default=".JPG,.jpg",
        help="A comma-separated list of extensions of images to slice.",
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
        default=50,
        required=False,
        help="Number of pixels of overlap between adjacent slices.",
    )
    args = parser.parse_args()

    # Raises an error if none of image arguments are provided
    if args.image_path is None and args.image_dir is None:
        raise ValueError("Please supply either an image or directory of images.")

    # Extracts extensions from the input argument
    exts = [
        f".{ext}" if "." not in ext else ext for ext in args.image_extensions.split(",")
    ]

    if args.image_path is not None:
        if args.image_path.suffix in exts:
            images = [args.image_path.expanduser()]

    elif args.image_dir is not None:

        # If image_dir points to a directory, paths to all the images with the correct
        # extension are put in a list
        image_dir = args.image_dir.expanduser()

        if image_dir.is_dir():
            images = []

            for ext in exts:
                images.extend(list(image_dir.glob(f"*{ext}")))
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
