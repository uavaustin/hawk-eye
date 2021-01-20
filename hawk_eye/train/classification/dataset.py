""" Datasets for loading data for our various training regimes. """

from typing import Tuple
import pathlib
import json
import random

import albumentations
import cv2
import torch

from hawk_eye.train import augmentations as augs


class ClfDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: pathlib.Path,
        img_ext: str = ".png",
        augmentations: albumentations.Compose = None,
    ) -> None:
        super().__init__()
        self.images = list(data_dir.glob(f"*{img_ext}"))
        assert self.images, f"No images found in {data_dir}."

        self.len = len(self.images)
        self.transform = augmentations
        self.data_dir = data_dir

        # Generate some simple stats about the data.
        self.num_bkgs = sum([1 for img in self.images if "background" in img.stem])
        self.num_targets = len(self.images) - self.num_bkgs

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(str(self.images[idx]))
        assert image is not None, f"Trouble readining {self.images[idx]}."

        image = torch.Tensor(self.transform(image=image)["image"])
        class_id = 0 if "background" in self.images[idx].stem else 1

        return image, class_id

    def __len__(self) -> int:
        return self.len

    def __str__(self) -> str:
        return f"{self.num_bkgs} backgrounds and {self.num_targets} targets."
