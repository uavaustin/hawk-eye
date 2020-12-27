""" Datasets for loading data for our various training regimes. """

import dataclasses
from typing import List, Tuple
import pathlib
import json
import random
import tempfile

import albumentations
import cv2
import torch

from hawk_eye.train import augmentations as augs


class ClfDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: pathlib.Path,
        img_ext: str = ".png",
        augs: albumentations.Compose = None,
    ) -> None:
        super().__init__()
        self.images = list(data_dir.glob(f"*{img_ext}"))
        assert self.images, f"No images found in {data_dir}."

        self.len = len(self.images)
        self.transform = augs
        self.data_dir = data_dir

        # Generate some simple stats about the data.
        self.num_bkgs = sum([1 for img in self.images if "background" in img.stem])
        self.num_targets = len(self.images) - self.num_bkgs

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(str(self.images[idx]))
        assert image is not None, f"Trouble reading {self.images[idx]}."

        image = torch.Tensor(self.transform(image=image)["image"])
        class_id = 0 if "background" in self.images[idx].stem else 1

        return image, class_id

    def __len__(self) -> int:
        return self.len

    def __str__(self) -> str:
        return f"{self.num_bkgs} backgrounds and {self.num_targets} targets."


class DetDataset(torch.utils.data.Dataset):
    """A COCO based dataset which can handel multiple input COCO datasets.
    To handle multiple, we have to remap the data into a merged dataset. """

    def __init__(
        self,
        data_dirs: List[pathlib.Path],
        metadata_paths: List[pathlib.Path],
        img_width: int = 512,
        img_height: int = 512,
        validation: bool = False,
    ) -> None:
        super().__init__()

        self.images = {}
        for metadata, data_dir in zip(metadata_paths, data_dirs):
            meta_data = json.loads(metadata.read_text())

            # Extract an image/id and id/image mapping
            local_image_ids = {
                data_dir / anno["file_name"]: anno["id"] for anno in meta_data["images"]
            }
            local_image_map = {
                anno["id"]: data_dir / anno["file_name"] for anno in meta_data["images"]
            }

            # Parse the annotations
            annotations = [
                {
                    "bbox": anno["bbox"],
                    "category_id": anno["category_id"],
                    "image_path": local_image_map[anno["image_id"]],
                    "image_id": anno["image_id"],
                    "area": anno["area"],
                    "id": anno["id"],
                    "iscrowd": 0,
                }
                for anno in meta_data["annotations"]
            ]
            # Assign the annotations to each image
            for anno in annotations:
                if anno["image_path"] not in self.images:
                    self.images[anno["image_path"]] = [anno]
                else:
                    self.images[anno["image_path"]].append(anno)

        # Now reassign all the image id's according to a mutli-dataset collation
        self.image_ids = {
            img_path: idx for idx, img_path in enumerate(self.images.keys())
        }

        self.img_height = img_height
        self.img_width = img_width
        self.len = len(self.images)
        self.transform = (
            augs.det_val_augs(img_height, img_width)
            if validation
            else augs.det_train_augs(img_height, img_width)
        )
        self.image_names = list(self.images.keys())

        # Finally, construct a temporary json to be the global COCO json
        self.json_file = pathlib.Path(tempfile.mkdtemp()) / "coco_eval.json"
        global_json = {
            "categories": json.loads(metadata_paths[0].read_text())["categories"]
        }
        global_json["images"] = [
            {"file_name": file_path.name, "id": idx}
            for file_path, idx in self.image_ids.items()
        ]
        global_json["annotations"] = []
        for labels in self.images.values():
            global_json["annotations"] += labels

        idx = 0
        for label in global_json["annotations"]:
            label["idx"] = 0
            idx += 1

        for anno in global_json["annotations"]:
            del anno["image_path"]

        self.json_file.write_text(json.dumps(global_json))

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_names[idx]
        labels = self.images[img_path]
        image = cv2.imread(str(img_path))

        assert image is not None, f"Trouble reading {img_path}."

        boxes = [torch.Tensor(item["bbox"]) for item in labels]
        if boxes:
            boxes = torch.stack(boxes)
            boxes[:, 2:] += boxes[:, :2]
            boxes /= torch.Tensor([self.img_width, self.img_height] * 2)
            boxes.clamp_(0.0, 1.0)

        category_ids = [label["category_id"] for label in labels]

        return self.transform(
            image=image,
            bboxes=boxes,
            category_ids=category_ids,
            image_ids=self.image_ids[img_path],
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
        self.transform = augs.feature_extraction_augmentations(100, 100)

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
