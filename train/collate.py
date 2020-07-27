""" Classes used to parse the output of the dataset during data loading. These classes
are needed to organize the detector dataset objects into tensors which can be stacked
in to batches. """

from typing import Tuple, List

import torch

from third_party.models import losses


class CollateVal:
    """ Simply return only the image tensors and the image ids for model evaluation. 
    The image id's are need for COCO metrics. """

    def __init__(self) -> None:
        pass

    def __call__(self, data_batch: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        images = []
        image_ids = []
        for item in data_batch:

            image_tensor = torch.Tensor(item["image"])
            images.append(image_tensor)
            image_ids.append(torch.Tensor([item["image_id"]]))

        # BHWC -> BCHW the images
        return torch.stack(images).permute(0, 3, 1, 2), image_ids


class Collate:
    def __init__(
        self, original_anchors: torch.Tensor, num_classes: int, image_size: int
    ) -> None:
        self.anchors = original_anchors.cpu()
        self.num_classes = num_classes
        self.image_size = image_size

    def __call__(
        self, data_batch: List[dict]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, boxes, labels = [], [], []

        for item in data_batch:

            if item["bboxes"]:
                boxes.append(torch.Tensor(item["bboxes"]) * self.image_size)
                labels.append(torch.Tensor(item["category_id"]))
            else:
                # If there are no annotations in the image, append empty tensors.
                boxes.append(torch.Tensor([]))
                labels.append(torch.Tensor([]))

            images.append(torch.Tensor(item["image"]))

        # Take the ground truth labels and boxes and find which original anchors
        # match to the ground truth boxes the best.
        gt_classes, gt_anchors_deltas = losses.get_ground_truth(
            self.anchors, boxes, labels, num_classes=self.num_classes
        )
        imgs = torch.stack(images).permute(0, 3, 1, 2)

        return imgs, gt_anchors_deltas, gt_classes
