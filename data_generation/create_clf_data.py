#!/usr/bin/env python3
"""  This script creates classification data for the pre-detection classification step;
however, this script relies upon detection already existing. Also, we need to ensure our
output dataset does not have repeats of image slices between the train and validation
sets -- this would mess up our training metrics. Since the detector data can contain
empty tiles, we will not be copying those here. Instead, we prefer to generate our own
empty slices so we have precise cnotrol over which exist. We will first generate all the
background crops and copy the target crops into one folder, then we'll shuffle them all
and split the data into 80% training and 20% validation. """

import json
import math
import pathlib
import random
import shutil
import tempfile
from typing import Tuple

import numpy as np
from PIL import Image
import tqdm

from data_generation import generate_config as config
from data_generation import create_detection_data

# Get constants from config
CLF_WIDTH, CLF_HEIGHT = config.PRECLF_SIZE
CROP_WIDTH, CROP_HEIGHT = config.CROP_SIZE


def create_clf_images(num_gen: int) -> None:
    """ Generate data for the classifier model. """

    # Do the initial processing in a temporary directory so we don't pollute the
    # workspace unncessarily.
    with tempfile.TemporaryDirectory() as d:
        tmp_dir = pathlib.Path(d)
        idx = 0

        print("Copying target tiles.")
        imgs = []
        imgs.extend(
            list((config.DATA_DIR / "detector_train").rglob(f"*{config.IMAGE_EXT}"))
        )
        imgs.extend(
            list((config.DATA_DIR / "detector_val").rglob(f"*{config.IMAGE_EXT}"))
        )

        for image_path in tqdm.tqdm(imgs):
            if json.loads(image_path.with_suffix(".json").read_text())["bboxes"]:
                # If there are labels, copy it to the save folder with the proper filename.
                # Load the image, resize, and save to new folder.
                image = Image.open(image_path).resize(config.PRECLF_SIZE)
                image.save(tmp_dir / f"target_{idx}{image_path.suffix}")
                idx += 1
                if idx > num_gen:
                    break

        # Collect all the backgrounds and slice them up.
        backgrounds = create_detection_data.get_backgrounds()
        num_tiles = 0
        for idx, img in enumerate(backgrounds):
            num_tiles = single_clf_image(img, idx, num_gen, tmp_dir, num_tiles)

        # Make output dir to save data after we do all the processing.
        train_dir = config.DATA_DIR / "clf_train"
        train_dir.mkdir(parents=True, exist_ok=True)

        val_dir = config.DATA_DIR / "clf_val"
        val_dir.mkdir(parents=True, exist_ok=True)

        for img in tmp_dir.glob("*"):
            if random.randint(0, 100) < 20:
                img.rename(val_dir / img.name)
            else:
                img.rename(train_dir / img.name)


def single_clf_image(
    image: Image.Image,
    number: int,
    num_gen: int,
    save_dir: pathlib.Path,
    num_tiles: int,
) -> None:
    """ Slice out crops from the original background image and save to disk. NOTE: we do
    not have any overlap between adjacent tiles because we want to avoid having any
    leakage between images. With data leakage, we might end up with two adjacent tiles in
    both the train and eval set. """
    image = Image.open(image)
    tile_num = 0
    for x in range(0, image.size[0] - config.CROP_SIZE[1], config.CROP_SIZE[0]):
        for y in range(0, image.size[1] - config.CROP_SIZE[1], config.CROP_SIZE[1]):
            if num_tiles > num_gen:
                break
            crop = image.crop((x, y, x + config.CROP_SIZE[0], y + config.CROP_SIZE[1]))
            crop = crop.resize(config.PRECLF_SIZE)
            save_path = (
                save_dir / f"background_{number}_{tile_num}_{x}_{y}{config.IMAGE_EXT}"
            )
            crop.save(save_path)
            num_tiles += 1
            tile_num += 1

    return num_tiles


if __name__ == "__main__":
    random.seed(42)

    if config.NUM_IMAGES != 0:
        create_clf_images(config.NUM_IMAGES)
