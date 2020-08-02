#!/usr/bin/env python3
"""  This script creates classification data for the pre-detection classification step;
however, this script relies upon detection already existing. Also, we need to ensure our
output dataset does not have repeats of image slices between the train and validation
sets -- this would mess up our training metrics. Since the detector data can contain
empty tiles, we will not be copying those here. Instead we prefer to generate our own
empty slices so we have precise cnotrol over which exist. """

import json
import math
import random
import shutil
from typing import Tuple

import numpy as np
from PIL import Image
import tqdm

from data_generation import generate_config as config

# Get constants from config
CLF_WIDTH, CLF_HEIGHT = config.PRECLF_SIZE
CROP_WIDTH, CROP_HEIGHT = config.CROP_SIZE


def create_clf_images(gen_type: str, num_gen: int) -> None:
    """ Generate data for the classifier model. """

    # Make output dir to save data.
    save_dir = config.DATA_DIR / gen_type
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get the already generated detection images. We specifically only care about the
    # images _with shapes_. Check that the json associated with the image does have
    # labels, forget it if not.
    data_folder = f"detector_{gen_type.split('_')[1]}"
    images_dir = config.DATA_DIR / data_folder / "images"

    idx = 0
    for image_path in tqdm.tqdm(
        list(images_dir.glob(f"*{config.IMAGE_EXT}")), total=num_gen
    ):
        if json.loads(image_path.with_suffix(".json").read_text())["bboxes"]:
            # If there are labels, copy it to the save folder with the proper filename.
            shutil.copy2(image_path, save_dir / f"target_{idx}{image_path.suffix}")
            idx += 1
            if idx > num_gen:
                break

    # Collect all the backgrounds.
    backgrounds = []
    for bkg_dir in config.BACKGROUNDS_DIRS:
        backgrounds.extend(list(bkg_dir.glob("*")))
    random.shuffle(backgrounds)

    # Calculate the 80/20 split.
    val_idx = math.ceil(0.2 * len(backgrounds))
    backgrounds = (
        backgrounds[:val_idx] if "train" in gen_type else backgrounds[-val_idx:]
    )

    for idx, img in enumerate(tqdm.tqdm(backgrounds, total=num_gen)):
        single_clf_image(img, gen_type, idx, num_gen)


def single_clf_image(
    image: Image.Image, gen_type: str, number: int, num_gen: int
) -> None:
    """ Slice out crops from the original background image and save to disk. NOTE: we do
    not have any overlap between adjacent tiles because we want to avoid having any
    leakage between images. With data leakage, we might end up with two adjacent tiles in
    both the train and eval set. """
    image = Image.open(image)
    tile_num = 0
    for x in range(0, image.size[0] - config.CROP_SIZE[1], config.CROP_SIZE[0]):
        for y in range(0, image.size[1] - config.CROP_SIZE[1], config.CROP_SIZE[1]):
            crop = image.crop((x, y, x + config.CROP_SIZE[0], y + config.CROP_SIZE[1]))
            crop = crop.resize(config.PRECLF_SIZE)
            data_path = config.DATA_DIR / gen_type
            save_path = (
                data_path / f"background_{number}_{tile_num}_{x}_{y}{config.IMAGE_EXT}"
            )
            crop.save(save_path)
            tile_num += 0
            if tile_num > num_gen:
                break


if __name__ == "__main__":
    random.seed(42)

    if config.NUM_IMAGES != 0:
        create_clf_images("clf_train", config.NUM_IMAGES)

    if config.NUM_VAL_IMAGES != 0:
        create_clf_images("clf_val", config.NUM_VAL_IMAGES)
