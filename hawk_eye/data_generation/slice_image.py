#!/usr/bin/env python3
""" Contains code to slice up an image into smaller tiles. """

from PIL import Image
import sys

if __name__ == "__main__":
    img_path = "hawk_eye/data_generation/data/test_flight_targets_20190215/EYED6011.JPG"
    img = Image.open(img_path)

    img.show()
