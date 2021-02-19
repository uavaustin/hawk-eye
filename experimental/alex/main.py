#!/usr/bin/env python3
import pathlib
import time

import cv2


img_dir = pathlib.Path("hawk_eye/data_generation/assets/competition-2019")
save_dir = pathlib.Path("/tmp/python_tiles")
save_dir.mkdir(exist_ok=True)

overlap = 0
tile_size = (512, 512)
for img in img_dir.glob("*.jpg"):

    start = time.perf_counter()
    image = cv2.imread(str(img))

    height, width, _ = image.shape
    x_step = width if width == tile_size[0] else tile_size[0] - overlap
    y_step = height if height == tile_size[1] else tile_size[1] - overlap

    for x in range(0, width, x_step):
        # Shift back to extract tiles on the image
        if x + tile_size[0] >= width and x != 0:
            x = width - tile_size[0]

        for y in range(0, height, y_step):
            if y + tile_size[1] >= height and y != 0:
                y = height - tile_size[1]

            tile = image[y : y + tile_size[1], x : x + tile_size[0]]

            cv2.imwrite(str(save_dir / f"{img.stem}_{x}_{y}{img.suffix}"), tile)

    end = time.perf_counter()
    print(f"{end - start:.4f}")

print(len(list(save_dir.glob("*"))))
