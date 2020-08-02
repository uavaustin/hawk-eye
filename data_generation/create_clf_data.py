#!/usr/bin/env python3
"""  This script creates classification data for the pre-detection classification step;
however, this script relies upon detection already existing. Also, we need to ensure our
output dataset does not have repeats of image slices between the train and validation
sets -- this would mess up our training metrics. Since the detector data can contain
empty tiles, we will not be copying those here. Instead we prefer to generate our own 
empty slices so we have precise cnotrol over which exist. """

import json
import shutil

from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import multiprocessing
import numpy as np

import generate_config as config
from create_detection_data import random_list, get_backgrounds

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
    for image_path in images_dir.glob(f"*{config.IMAGE_EXT}"):
        if json.loads(image_path.with_suffix(".json").read_text())["bboxes"]:
            # If there are labels, copy it to the save folder with the proper filename.
            shutil.copy2(image_path, save_dir / f"target_{idx}{image_path.suffix}")
            idx += 1

    # Get random crops and augmentations for background
    backgrounds = get_backgrounds()
    gen_types = [gen_type] * num_gen

    numbers = list(range(len(backgrounds)))
    data = zip(get_backgrounds(), [gen_type] * num_gen, numbers)

    with multiprocessing.Pool(None) as pool:
        processes = pool.imap_unordered(_single_clf_image, data)
        for _ in tqdm(processes, total=num_gen):
            pass


def _single_clf_image(data) -> None:
    """Crop detection image and augment clf image and save"""
    image, gen_type, number = data
    tile_num = 0
    for x in range(0, image.size[0], config.CROP_SIZE[0] - config.CROP_OVERLAP):
        for y in range(0, image.size[1], config.CROP_SIZE[1] - config.CROP_OVERLAP):
            crop = image.crop(
                (x, y, x + config.CROP_SIZE[0], y + config.CROP_SIZE[1])
            )
            crop = crop.resize(config.PRECLF_SIZE)
            data_path = config.DATA_DIR / gen_type
            save_path = (
                data_path / f"background_{number}_{tile_num}{x}{y}{config.IMAGE_EXT}"
            )
            crop.save(save_path)
            tile_num += 0


if __name__ == "__main__":

    if config.NUM_IMAGES != 0:
        create_clf_images("clf_train", config.NUM_IMAGES)

    if config.NUM_VAL_IMAGES != 0:
        create_clf_images("clf_val", config.NUM_VAL_IMAGES)
