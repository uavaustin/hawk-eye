""" Datasets for loading various types of data. """

from typing import Tuple
import pathlib
import json
import random

import albumentations
import cv2
import torch

from train import augmentations as augs


class ClfDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: pathlib.Path, img_ext: str = ".png"):
        super().__init__()
        self.images = list(data_dir.glob(f"*{img_ext}"))
        assert self.images, f"No images found in {data_dir}."

        self.len = len(self.images)
        self.transform = classification_augmentations(224, 244)
        self.data_dir = data_dir

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(str(self.images[idx]))
        image = torch.Tensor(self.transform(image=image)["image"])
        image = image.permute(2, 0, 1)
        class_id = 0 if "background" in self.images[idx].stem else 1

        return image, class_id

    def __len__(self) -> int:
        return self.len


class DetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: pathlib.Path,
        metadata_path: pathlib.Path,
        img_ext: str = ".png",
        img_width: int = 512,
        img_height: int = 512,
    ) -> None:
        super().__init__()
        self.meta_data = json.loads(metadata_path.read_text())
        self.images = list(data_dir.glob(f"*{img_ext}"))
        assert self.images, f"No images found in {data_dir}."

        self.img_height = img_height
        self.img_width = img_width
        self.len = len(self.images)
        self.transform = augs.det_train_augs(img_height, img_width)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(str(self.images[idx]))
        assert image is not None, f"Trouble reading {self.images[idx]}."
        labels = json.loads(self.images[idx].with_suffix(".json").read_text())

        boxes = [
            torch.Tensor(
                [item["x1"], item["y1"], item["x1"] + item["w"], item["y1"] + item["h"]]
            )
            for item in labels["bboxes"]
        ]

        if boxes:
            boxes = torch.stack([torch.clamp(box, 0.0, 1.0) for box in boxes])

        category_ids = [label["class_id"] for label in labels["bboxes"]]

        return self.transform(
            image=image,
            bboxes=boxes,
            category_id=category_ids,
            image_id=labels["image_id"],
        )

    def __len__(self) -> int:
        return self.len


class TargetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: pathlib.Path,
        img_ext: str = ".png",
        img_width: int = 100,
        img_height: int = 100,
    ) -> None:
        super().__init__()
        self.images = list(data_dir.glob(f"*{img_ext}"))

        self.len = len(self.images)
        self.transform = feature_extraction_augmentations(100, 100)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ This dataset will return three randomly selected images. """
        image_path1 = random.choice(self.images)
        image_path2 = image_path1

        while image_path1 == image_path2:
            image_path2 = random.choice(self.images)

        image1 = cv2.imread(image_path1)
        image1 = torch.Tensor(self.transform(image=image1)["image"]).permute(2, 0, 1)

        image2 = cv2.imread(image_path1)
        image2 = torch.Tensor(self.transform(image=image2)["image"]).permute(2, 0, 1)

        image3 = cv2.imread(image_path2)
        image3 = torch.Tensor(self.transform(image=image3)["image"]).permute(2, 0, 1)

        return image1, image2, image3
