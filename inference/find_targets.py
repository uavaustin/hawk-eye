#!/usr/bin/env python3
""" Contains logic for finding targets in images. """

import argparse
import pathlib
import time
import json
from typing import List, Tuple, Generator

import albumentations
import cv2
import numpy as np
import torch

from core import classifier, detector
from inference import types
from data_generation import generate_config as config
from third_party.models import postprocess


def _val_augmentations() -> albumentations.Compose:
    return albumentations.Compose([albumentations.Normalize()])


def tile_image(
    image: np.ndarray, tile_size: Tuple[int, int], overlap: int  # (H, W)
) -> torch.Tensor:

    augmenation = _val_augmentations()

    tiles = []
    coords = []
    for x in range(0, image.shape[1], tile_size[1] - overlap):

        # Shift back to extract tiles on the image
        if x + tile_size[1] >= image.shape[1] and x != 0:
            x = image.shape[0] - tile_size[1]

        for y in range(0, image.shape[0], tile_size[0] - overlap):
            if y + tile_size[0] >= image.shape[0] and y != 0:
                y = image.shape[0] - tile_size[0]
            tile = augmenation(image=image[y : y + tile_size[0], x : x + tile_size[1]])[
                "image"
            ]
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


@torch.no_grad()
def find_targets(
    clf_model: torch.nn.Module,
    det_model: torch.nn.Module,
    images: List[pathlib.Path],
    save_jsons: bool = False,
    visualization_dir: pathlib.Path = None,
) -> None:
    for image_path in images:
        retval = []
        image = cv2.imread(str(image_path))
        assert image is not None, f"Could not read {image_path}."
        start = time.perf_counter()
        image_tensor, coords = tile_image(image, config.CROP_SIZE, config.CROP_OVERLAP)

        # Keep track of the tiles that were classified as having targets for
        # visualization.
        target_tiles = []

        # Get the image slices.
        for tiles_batch, coords in create_batches(image_tensor, coords, 200):

            if torch.cuda.is_available():
                tiles_batch = tiles_batch.cuda().half()

            # Resize the slices for classification.
            tiles = torch.nn.functional.interpolate(tiles_batch, config.PRECLF_SIZE)

            # Call the pre-clf to find the target tiles.
            preds = clf_model.classify(tiles, probability=True)[:, 1] > 0.8
            print(tiles_batch.shape, preds.shape)
            if preds.sum().item():
                for det_tiles, det_coords in create_batches(
                    tiles_batch[preds.long()], coords, 15
                ):
                    # Pass these target-containing tiles to the detector
                    det_tiles = torch.nn.functional.interpolate(
                        det_tiles, config.DETECTOR_SIZE
                    )
                    print(det_tiles.shape)
                    boxes = det_model(det_tiles)

                    retval.extend(zip(target_tiles, boxes))
            else:
                retval.extend(zip(coords, []))

        targets = globalize_boxes(retval, config.CROP_SIZE[0])
        print(time.perf_counter() - start)

        if visualization_dir is not None:
            visualize_image(
                image_path.name, image, visualization_dir, targets, target_tiles
            )


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
    """ Function used to draw boxes and information onto image for
    visualizing the output of inference. """
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

    cv2.imwrite(str(visualization_dir / image_name), image)


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
        default="dev",
        help="Timestamp of the classifier model to use.",
    )
    parser.add_argument(
        "--det_timestamp",
        required=False,
        type=str,
        default="dev",
        help="Timestamp of the detector model to use.",
    )
    parser.add_argument(
        "--visualization_dir",
        required=False,
        type=pathlib.Path,
        help="Optional directory to save visualization to.",
    )
    args = parser.parse_args()

    clf_model = classifier.Classifier(
        timestamp=args.clf_timestamp, half_precision=torch.cuda.is_available()
    )
    clf_model.eval()
    det_model = detector.Detector(
        timestamp=args.det_timestamp,
        confidence=0.3,
        half_precision=torch.cuda.is_available(),
    )
    det_model.eval()

    # Do FP16 when inferencing
    if torch.cuda.is_available():
        det_model.cuda()
        det_model.half()

        clf_model.cuda()
        clf_model.half()

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

    find_targets(clf_model, det_model, imgs, visualization_dir=viz_dir)
