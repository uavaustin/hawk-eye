#!/usr/bin/env python3

import argparse
import csv
import pathlib
import json
import shutil

from hawk_eye.data_generation import generate_config
from hawk_eye.data_generation import create_detection_data


def parse_labels(
    image_dir: pathlib.Path, save_dir: pathlib.Path, csv_path: pathlib.Path
):
    images = sorted(image_dir.glob("*"))
    save_dir = save_dir / "images"
    save_dir.mkdir(exist_ok=True, parents=True)
    with open(csv_path, newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=" ", quotechar="|")
        for row in spamreader:
            vals = row[0].split(",")
            original_tile_path = image_dir / vals[-3]
            tile_save_path = save_dir / vals[-3]
            tile_json = (tile_save_path).with_suffix(".json")
            class_name = vals[0]
            x1, y1, w, h = vals[1:5]
            img_w, img_h = vals[-2], vals[-1]
            label = {
                "bboxes": [
                    {
                        "class_id": generate_config.SHAPE_TYPES.index(class_name),
                        "x1": float(x1) / float(img_w),
                        "y1": float(y1) / float(img_h),
                        "w": float(w) / float(img_w),
                        "h": float(h) / float(img_h),
                    },
                ],
                "image_id": images.index(original_tile_path),
            }
            tile_json.write_text(json.dumps(label, indent=2))
            shutil.copy2(original_tile_path, tile_save_path)

    create_detection_data.create_coco_metadata(
        save_dir, save_dir.parent / "val_coco.json"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image_dir",
        type=pathlib.Path,
        required=False,
        help="Path to directory of tiles that were labeled.",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        required=True,
        help="Path to directory in which to store sliced images.",
    )
    parser.add_argument(
        "--csv_path",
        type=pathlib.Path,
        required=True,
        help="Path to the csv outputted from labeling app.",
    )
    args = parser.parse_args()

    parse_labels(args.image_dir, args.save_dir, args.csv_path)
