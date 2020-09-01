#!/usr/bin/env python3
""" Contains logic for finding targets in images. """

import argparse
import pathlib
import time
import json
from typing import List, Tuple, Generator

import cv2
from PIL import Image
import numpy as np
import torch

from core import classifier, detector
from inference import types
from data_generation import generate_config as config
from third_party.models import postprocess

_PROD_MODELS = {"clf": "2020-08-20T18.11.42", "det": "2020-08-21T00.46.40"}


# Taken directly from albumentation src: augmentations/functional.py#L131.
# This is the only function we really need from albumentations for inference,
# so it is not worth requiring that as a dependency for distribution.
def normalize(
    img: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    max_pixel_value: float = 255.0,
) -> np.ndarray:
    """ Normalize images based on ImageNet values. This is identical to
    albumentations normalization used for training.

    Args:
        img: The image to normalize.
        mean: The mean of each channel.
        std: The standard deviation of each channel.
        max_pixel_value: The max value a pixel can have.

    Returns:
        A normalized image as a numpy array.
    """

    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator

    return img


def tile_image(
    image: np.ndarray, tile_size: Tuple[int, int], overlap: int  # (H, W)
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """ Take in an image and tile it into smaller tiles for inference.

    Args:
        image: The input image to tile.
        tile_size: The (width, height) of the tiles.
        overlap: The overlap between adjacent tiles.

    Returns:
        A tensor of the tiles and a list of the (x, y) offset for the tiles.
            The offets are needed to keep track of which tiles have targets.

    Usage: TODO(alex): fix this test
        >>> tile_image(np.zeros(1000, 1000, 3), (512, 512), overlap)
        []
    """
    tiles, coords = [], []
    width, height = image.size

    for x in range(0, width, tile_size[0] - overlap):

        # Shift back to extract tiles on the image
        if x + tile_size[0] >= width and x != 0:
            x = width - tile_size[0]

        for y in range(0, height, tile_size[1] - overlap):
            if y + tile_size[1] >= height and y != 0:
                y = height - tile_size[1]

            tile = normalize(
                np.array(image.crop((x, y, x + tile_size[0], y + tile_size[1])))
            )

            tiles.append(torch.Tensor(tile))
            coords.append((x, y))

    # Transpose the images from BHWC -> BCHW
    tiles = torch.stack(tiles).permute(0, 3, 1, 2)

    return tiles, coords


def create_batches(
    image_tensor: torch.Tensor, coords: List[Tuple[int, int]], batch_size: int
) -> Generator[types.BBox, None, None]:
    """ Creates batches of images based on the supplied params. The whole image
    is tiled first, the batches are generated.

    Args:
        image: The opencv opened image.
        tile_size: The height, width of the tiles to create.
        overlap: The amount of overlap between tiles.
        batch_size: The number of images to have per batch.

    Returns:
        Yields the image batch and the top left coordinate of the tile in the
        space of the original image.
    """

    for idx in range(0, image_tensor.shape[0], batch_size):
        yield image_tensor[idx : idx + batch_size], coords[idx : idx + batch_size]


def load_models(
    clf_timestamp: str = _PROD_MODELS["clf"], det_timestamp: str = _PROD_MODELS["det"]
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """ Loads the given time stamps for the classification and detector models.

    Args:
        clf_timestamp: Which classification model to load.
        det_timestamp: Which detection model to load.

    Returns:
        Returns both models.
    """

    clf_model = classifier.Classifier(
        timestamp=clf_timestamp, half_precision=torch.cuda.is_available()
    )
    clf_model.eval()

    # TODO(alex): Pass in the confidence for the detector.
    det_model = detector.Detector(
        timestamp=det_timestamp,
        confidence=0.2,
        half_precision=torch.cuda.is_available(),
    )
    det_model.eval()

    # Do FP16 when inferencing
    if torch.cuda.is_available():
        det_model.cuda()
        det_model.half()

        clf_model.cuda()
        clf_model.half()

    return clf_model, det_model


def find_all_targets(
    images: List[pathlib.Path],
    clf_timestamp: str = _PROD_MODELS["clf"],
    det_timestamp: str = _PROD_MODELS["det"],
    save_jsons: bool = False,
    visualization_dir: pathlib.Path = None,
) -> None:
    """ Entrypoint function if running this script as main.

    Args:
        images: A list of all the images to inference.
        clf_timestamp: The classification model to load.
        det_timestamp: The detection model to load.
        save_jsons: Whether or not to save the json files.
        visualization_dir: Where to save the visualizations, if any.
    """

    clf_model, det_model = load_models(clf_timestamp, det_timestamp)

    for image_path in images:

        image = Image.open(image_path)
        assert image is not None, f"Could not read {image_path}."

        targets, target_tiles = find_targets(image, clf_model, det_model)

        if visualization_dir is not None:
            visualize_image(
                image_path.name,
                np.array(image),
                visualization_dir,
                targets,
                target_tiles,
            )


@torch.no_grad()
def find_targets(
    image: Image.Image, clf_model: torch.nn.Module, det_model: torch.nn.Module,
) -> None:

    image_tensor, coords = tile_image(image, config.CROP_SIZE, config.CROP_OVERLAP)

    # Keep track of the tiles that were classified as having targets for
    # visualization.
    target_tiles, retval = [], []

    start = time.perf_counter()

    # Get the image slices.
    for tiles_batch, coords in create_batches(image_tensor, coords, 200):

        if torch.cuda.is_available():
            tiles_batch = tiles_batch.cuda().half()

        # Resize the slices for classification.
        tiles = torch.nn.functional.interpolate(tiles_batch, config.PRECLF_SIZE)

        # Call the pre-clf to find the target tiles.
        # TODO(alex): Pass in the classification confidence from cmdl.
        preds = clf_model.classify(tiles, probability=True)[:, 1] >= 0.90

        target_tiles += [coords[idx] for idx, val in enumerate(preds) if val]
        if preds.sum().item():
            for det_tiles, det_coords in create_batches(tiles_batch[preds], coords, 15):
                # Pass these target-containing tiles to the detector
                det_tiles = torch.nn.functional.interpolate(
                    det_tiles, config.DETECTOR_SIZE
                )
                boxes = det_model(det_tiles)

                retval.extend(zip(target_tiles, boxes))
        else:
            retval.extend(zip(coords, []))

    targets = globalize_boxes(retval, config.CROP_SIZE[0])
    print(time.perf_counter() - start)
    return globalize_boxes(retval, config.CROP_SIZE[0]), target_tiles


def globalize_boxes(
    results: List[postprocess.BoundingBox], img_size: int
) -> List[types.Target]:
    final_targets = []
    for coords, bboxes in results:
        for box in bboxes:
            relative_coords = box.box * torch.Tensor([img_size] * 4)
            relative_coords += torch.Tensor(2 * list(coords)).int()
            final_targets.append(
                types.Target(
                    x=int(relative_coords[0]),
                    y=int(relative_coords[1]),
                    width=int(relative_coords[2] - relative_coords[0]),
                    height=int(relative_coords[3] - relative_coords[1]),
                    shape=types.Shape[
                        config.OD_CLASSES[box.class_id].upper().replace("-", "_")
                    ],
                )
            )

    return final_targets


def visualize_image(
    image_name: str,
    image: np.ndarray,
    visualization_dir: pathlib.Path,
    targets: List[types.Target],
    clf_tiles: List[Tuple[int, int]],
) -> None:
    """ Function used to draw boxes and information onto image for visualizing the output
    of inference.

    Args:
        image_name: The original image name used for saving the visualization.
        image: The image array.
        visualization_dir: Where to save the visualizations.
        targets: A list of the targets that were found during inference.
        clf_tiles: Which tiles were classified as having targets in them.
    """

    for target in targets:
        top_left = (target.x, target.y)
        bottom_right = (target.x + target.width, target.y + target.height)
        image = cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 1)
        cv2.putText(
            image,
            target.shape.name.lower(),
            (target.x, target.y - 3),  # Shift text tinsy bit
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            1,
        )

    for group in clf_tiles:
        x, y = group
        w, h = config.CROP_SIZE
        tile = np.zeros((w, h, 3), dtype=np.uint8)
        cv2.addWeighted(
            tile, 0.3, image[y : y + h, x : x + w], 0.7, 0, image[y : y + h, x : x + w]
        )
    image = Image.fromarray(image)
    image.save(str(visualization_dir / image_name))


# TODO(alex) use this
def save_target_meta(filename_meta, filename_image, target):
    """ Save target metadata to a file. """
    with open(filename_meta, "w") as f:
        meta = {
            "x": target.x,
            "y": target.y,
            "width": target.width,
            "height": target.height,
            "orientation": target.orientation,
            "shape": target.shape.name.lower(),
            "background_color": target.background_color.name.lower(),
            "alphanumeric": target.alphanumeric,
            "alphanumeric_color": target.alphanumeric_color.name.lower(),
            "image": filename_image,
            "confidence": target.confidence,
        }

        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script used to find targets in an image"
    )
    parser.add_argument(
        "--image_path",
        required=False,
        type=pathlib.Path,
        help="Path to an image to inference.",
    )
    parser.add_argument(
        "--image_dir",
        required=False,
        type=pathlib.Path,
        help="Path to directory of images to inference.",
    )
    parser.add_argument(
        "--image_extension",
        required=False,
        type=str,
        help="Needed when an image directory is supplied.",
    )
    parser.add_argument(
        "--clf_timestamp",
        required=False,
        type=str,
        default=_PROD_MODELS["clf"],
        help="Timestamp of the classifier model to use.",
    )
    parser.add_argument(
        "--det_timestamp",
        required=False,
        type=str,
        default=_PROD_MODELS["det"],
        help="Timestamp of the detector model to use.",
    )
    parser.add_argument(
        "--visualization_dir",
        required=False,
        type=pathlib.Path,
        help="Optional directory to save visualization to.",
    )
    args = parser.parse_args()

    # Get either the image or images
    if args.image_path is None and args.image_dir is None:
        raise ValueError("Please supply either an image or directory of images.")
    if args.image_path is not None:
        imgs = [args.image_path.expanduser()]
    elif args.image_dir is not None:
        assert args.image_extension.startswith(".")
        imgs = args.image_dir.expanduser().glob(f"*{args.image_extension}")

    viz_dir = None
    if args.visualization_dir is not None:
        viz_dir = args.visualization_dir.expanduser()
        viz_dir.mkdir(exist_ok=True, parents=True)

    find_all_targets(imgs, args.clf_timestamp, args.det_timestamp, imgs, viz_dir)
