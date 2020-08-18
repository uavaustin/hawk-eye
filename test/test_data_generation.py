#!/usr/bin/env ptyhon3
""" Collection of unittests to test data generation scripts. """

import pathlib
import tempfile
import unittest

from data_generation import create_detection_data

# TODO This replaces the build.py. This would test as many
# of the shape generation utilities as possible, then also
# do end-to-end tests.
class DetectionDataGeneration(unittest.TestCase):
    def test_temp(self) -> None:
        num_imgs = 10
        with tempfile.TemporaryDirectory() as d:
            tmp_dir = pathlib.Path(d)
            create_detection_data.generate_all_images(tmp_dir / "train", num_imgs, 0)

            self.assertEqual(len((tmp_dir / "train").rglob("*.png")), num_imgs)


if __name__ == "__main__":
    unittest.main()
