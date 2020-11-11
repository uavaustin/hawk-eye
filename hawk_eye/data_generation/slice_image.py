#!/usr/bin/env python3
""" Contains code to slice up an image into smaller tiles. """

# COMBAK: check import order
from PIL import Image
import sys
import pdb
import numpy as np
from typing import Tuple


# COMBAK: how do I typecheck lists?
def slice_image(
    img_path: str, tile_size: Tuple[int, int], overlap: int  # (H, W)
) -> Image.Image:
    """ Take in an image and tile it into smaller tiles for inference.
    Args:
        image: The input image to tile.
        tile_size: The (width, height) of the tiles.
        overlap: The overlap between adjacent tiles.
    Returns:
        A list of PIL tiles.
    Usage:
        >>> tiles = slice_image(Image.new("RGB", (1000, 1000)), (512, 512), 50)
        >>> len(tiles)
        9
        >>> tiles[0].show()
        <Displays image> # COMBAK: is this the right way to say this?
    """
    # COMBAK: change the docstring
    pdb.set_trace()
    image = Image.open(img_path)
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
            # TODO: save as originalimagename-x-y
            img_name = img_path.split("/")[-1][:-4]
            tile.save(f"{img_name}-{x}-{y}.JPG")


if __name__ == "__main__":
    slice_image(
        img_path="hawk_eye/data_generation/data/test_flight_targets_20190215/EYED6009.JPG",
        tile_size=(512, 512),
        overlap=50,
    )
