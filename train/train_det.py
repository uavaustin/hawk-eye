#!/usr/bin/env python3
""" Train an object detector to localize the targets in images. """

import argparse
import pathlib
from typing import Tuple, List
import yaml
import tarfile
import tempfile
import datetime
import json
import time
import shutil

import torch
import numpy as np

from train import datasets
from train.train_utils import utils
from data_generation import generate_config
from third_party.models import losses
from third_party import coco_eval
from core import detector

_LOG_INTERVAL = 10
_IMG_WIDTH, _IMG_HEIGHT = generate_config.DETECTOR_SIZE
_SAVE_DIR = pathlib.Path("~/runs/uav-det").expanduser()


def detections_to_dict(bboxes: list, image_ids: torch.Tensor) -> List[dict]:
    """ Used to turn raw bounding box detections into a dictionary which can be
    serialized for the pycocotools package. """
    detections: List[dict] = []
    for image_boxes, image_id in zip(bboxes, image_ids):

        for bbox in image_boxes:
            box = bbox.box.int()
            # XYXY -> XYWH
            box[2:] -= box[:2]
            detections.append(
                {
                    "image_id": image_id.item(),
                    "category_id": bbox.class_id,
                    "bbox": box.tolist(),
                    "score": bbox.confidence,
                }
            )
    return detections


def train(model_cfg: dict, train_cfg: dict, save_dir: pathlib.Path = None) -> None:

    # TODO(alex) these paths should be in the generate config
    train_batch_size = train_cfg.get("train_batch_size", 8)
    train_loader = create_data_loader(
        train_cfg,
        generate_config.DATA_DIR / "detector_train/images",
        generate_config.DATA_DIR / "detector_train/train_coco.json",
        train_batch_size,
    )
    eval_batch_size = train_cfg.get("eval_batch_size", 8)
    eval_loader = create_data_loader(
        train_cfg,
        generate_config.DATA_DIR / "detector_val/images",
        generate_config.DATA_DIR / "detector_val/val_coco.json",
        eval_batch_size,
    )

    use_cuda = train_cfg.get("gpu", False)
    eval_results = None

    # Load the model and remove the classification head of the backbone.
    # We don't need the backbone to make classifications.
    det_model = detector.Detector(
        num_classes=len(generate_config.OD_CLASSES),
        model_params=model_cfg,
        confidence=0.05,
    )
    det_model.train()
    print(f"Model architecture: \n {det_model}")

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        det_model.cuda()

    epochs = train_cfg.get("epochs", 0)
    assert epochs > 0, "Please supply epoch > 0"

    optimizer = create_optimizer(train_cfg["optimizer"], det_model)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-2,
        total_steps=len(train_loader) * epochs,
        final_div_factor=1e9,
        div_factor=2,
        pct_start=2 * len(train_loader) / (len(train_loader) * epochs),
    )
    for epoch in range(epochs):

        all_losses = []
        clf_losses = []
        reg_losses = []

        for idx, (images, boxes, classes, _) in enumerate(train_loader):

            optimizer.zero_grad()

            if use_cuda:
                images = images.cuda()
                boxes = boxes.cuda()
                classes = classes.cuda()

            # Forward pass through detector
            cls_per_level, reg_per_level = det_model(images)

            # Compute the losses
            cls_loss, reg_loss = losses.compute_losses(
                det_model.anchors.all_anchors,
                gt_classes=list(classes.unbind(0)),
                gt_boxes=list(boxes.unbind(0)),
                cls_per_level=cls_per_level,
                reg_per_level=reg_per_level,
                num_classes=len(generate_config.OD_CLASSES),
            )
            total_loss = cls_loss + reg_loss
            clf_losses.append(cls_loss)
            reg_losses.append(reg_loss)
            # Propogate the gradients back through the model
            total_loss.backward()
            all_losses.append(total_loss.item())
            # Perform the parameter updates
            optimizer.step()
            lr_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            if idx % _LOG_INTERVAL == 0:
                print(
                    f"Epoch: {epoch} step {idx}, "
                    f"clf loss {sum(clf_losses) / len(clf_losses):.5}, "
                    f"reg loss {sum(reg_losses) / len(reg_losses):.5}, "
                    f"lr {lr:.5}"
                )

        # Call evaluation function
        det_model.eval()
        eval_results = eval(det_model, eval_loader, eval_results, use_cuda, save_dir)
        det_model.train()

        print(
            f"Epoch: {epoch}, Training loss {sum(all_losses) / len(all_losses):.5} \n"
            f"{eval_results}"
        )


def eval(
    det_model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    previous_best: dict,
    use_cuda: bool = False,
    save_dir: pathlib.Path = None,
) -> float:
    """ Evalulate the model against the evaulation set. Save the best
    weights if specified. Use the pycocotools package for metrics. """
    start = time.perf_counter()
    total_num = 0
    with torch.no_grad():
        detections_dict: List[dict] = []
        for images, _, _, image_ids in eval_loader:
            if torch.cuda.is_available():
                images = images.cuda()
            total_num += images.shape[0]
            detections = det_model(images)
            detections_dict.extend(detections_to_dict(detections, image_ids))

    print(f"Evaluated {total_num} images in {time.perf_counter() - start:.3} seconds.")
    if detections_dict:
        with tempfile.TemporaryDirectory() as d:
            tmp_json = pathlib.Path(d) / "det.json"
            tmp_json.write_text(json.dumps(detections_dict))
            results = coco_eval.get_metrics(
                "/home/alex/datasets/wheat/val.json", tmp_json
            )

    previous_best = results if previous_best is None else previous_best

    for (metric, old), new in zip(previous_best.items(), results.values()):
        if new >= old:
            previous_best[metric] = new
            utils.save_model(det_model, save_dir / f"detector-{metric}.pt")

    return previous_best


def create_data_loader(
    train_cfg: dict,
    data_dir: pathlib.Path,
    metadata_path: pathlib.Path,
    batch_size: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    assert data_dir.is_dir(), data_dir

    dataset = datasets.DetDataset(
        data_dir,
        metadata_path=metadata_path,
        img_ext=generate_config.IMAGE_EXT,
        img_width=512,
        img_height=512,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=True
    )
    return loader


def create_optimizer(optim_cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """ Take in optimizer config and create the optimizer for training. """
    name = optim_cfg.get("type", None)
    if name.lower() == "sgd":
        lr = float(optim_cfg["lr"])
        momentum = float(optim_cfg["momentum"])
        weight_decay = float(optim_cfg["weight_decay"])
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    elif name.lower() == "rmsprop":
        lr = float(optim_cfg["lr"])
        momentum = float(optim_cfg["momentum"])
        weight_decay = float(optim_cfg["weight_decay"])
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Improper optimizer supplied {name}.")

    return optimizer


if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description="Trainer code for classifcation models."
    )
    parser.add_argument(
        "--model_config",
        required=True,
        type=pathlib.Path,
        help="Path to yaml model definition.",
    )
    args = parser.parse_args()

    config_path = args.model_config.expanduser()
    assert config_path.is_file(), f"Can't find {config_path}."

    # Load the model config
    config = yaml.safe_load(config_path.read_text())
    model_cfg = config["model"]
    train_cfg = config["training"]

    save_dir = _SAVE_DIR / (datetime.datetime.now().isoformat().split(".")[0])
    save_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(config_path, save_dir / "config.yaml")

    train(model_cfg, train_cfg, save_dir)

    # Create tar archive.
    with tarfile.open(save_dir / "detector.tar.gz", mode="w:gz") as tar:
        for model_file in save_dir.glob("*"):
            tar.add(model_file, arcname=model_file.name)
