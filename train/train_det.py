#!/usr/bin/env python3
""" Generalized object detection script. This script will use as many gpus
as torch can find. If Nvidia's Apex is available, that will be used for
mixed precision training. """

import argparse
import pathlib
from typing import Tuple, List, Dict, Any
import yaml
import tarfile
import tempfile
import datetime
import json
import time
import shutil
import os

import torch
import numpy as np

try:
    import apex

    USE_APEX = True
except ImportError as e:
    USE_APEX = False
    print(f"{e} Apex not found. No mix precision used.")

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


def train(
    local_rank: int,
    world_size: int,
    model_cfg: dict,
    train_cfg: dict,
    save_dir: pathlib.Path = None,
) -> None:

    # Do some general setup. When using distributed training and Apex, the device needs
    # to be set before loading the model.
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    is_main = local_rank == 0

    if world_size > 1:
        assert (
            torch.distributed.is_nccl_available()
        ), "NCCL must be avaliable for parallel training."
        torch.distributed.init_process_group(
            "nccl", init_method="env://", world_size=world_size, rank=local_rank,
        )

    # TODO(alex) these paths should be in the generate config
    train_batch_size = train_cfg.get("train_batch_size", 8)
    train_loader, train_sampler = create_data_loader(
        train_cfg,
        generate_config.DATA_DIR / "detector_train/images",
        generate_config.DATA_DIR / "detector_train/train_coco.json",
        train_batch_size,
        world_size,
        shuffle=True
    )
    eval_batch_size = train_cfg.get("eval_batch_size", 8)
    eval_loader, _ = create_data_loader(
        train_cfg,
        generate_config.DATA_DIR / "detector_val/images",
        generate_config.DATA_DIR / "detector_val/val_coco.json",
        eval_batch_size,
        world_size,
        shuffle=False
    )

    # Load the model and remove the classification head of the backbone.
    # We don't need the backbone to make classifications.
    model = detector.Detector(
        num_classes=len(generate_config.OD_CLASSES),
        model_params=model_cfg,
        confidence=0.05,
    )
    model.to(device)
    model.train()
    print(f"Model architecture: \n {model}")

    # Construct the optimizer and wrap with Apex if available.
    optimizer = create_optimizer(train_cfg["optimizer"], model)
    if USE_APEX:
        model, optimizer = apex.amp.initialize(
            model, optimizer, opt_level="O1", verbosity=is_main
        )

    # Adjust the model for distributed training. If we are using apex, wrap the model
    # with Apex's utilies, else PyTorch.
    if USE_APEX and world_size:
        model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
    elif world_size:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )

    epochs = train_cfg.get("epochs", 0)
    assert epochs > 0, "Please supply epoch > 0"

    eval_results = None

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

        # Set the train loader's epoch so data will be re-shuffled.
        train_sampler.set_epoch(epoch)

        for idx, (images, boxes, classes, _) in enumerate(train_loader):

            optimizer.zero_grad()

            if use_cuda:
                images = images.to(device)
                boxes = boxes.to(device)
                classes = classes.to(device)

            # Forward pass through detector
            cls_per_level, reg_per_level = model(images)

            # Compute the losses
            cls_loss, reg_loss = losses.compute_losses(
                model.module.anchors.all_anchors,
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
            if USE_APEX:
                with apex.amp.scale_loss(total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()

            all_losses.append(total_loss.item())
            # Perform the parameter updates
            optimizer.step()
            lr_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            if idx % _LOG_INTERVAL == 0 and is_main == 0:
                print(
                    f"Epoch: {epoch} step {idx}, "
                    f"clf loss {sum(clf_losses) / len(clf_losses):.5}, "
                    f"reg loss {sum(reg_losses) / len(reg_losses):.5}, "
                    f"lr {lr:.5}"
                )

        # Call evaluation function
        model.eval()
        eval_results = eval(model, eval_loader, eval_results, use_cuda, save_dir)
        model.train()

        print(
            f"Epoch: {epoch}, Training loss {sum(all_losses) / len(all_losses):.5} \n"
            f"{eval_results}"
        )


def eval(
    model: torch.nn.Module,
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
            detections = model(images)
            detections_dict.extend(detections_to_dict(detections, image_ids))

    print(f"Evaluated {total_num} images in {time.perf_counter() - start:.3} seconds.")
    if detections_dict:
        with tempfile.TemporaryDirectory() as d:
            tmp_json = pathlib.Path(d) / "det.json"
            tmp_json.write_text(json.dumps(detections_dict))
            results = coco_eval.get_metrics(
                "/home/alex/Desktop/projects/uav/hawk-eye/data_generation/data/detector_val/val_coco.json", tmp_json
            )

    previous_best = results if previous_best is None else previous_best

    for (metric, old), new in zip(previous_best.items(), results.values()):
        if new >= old:
            previous_best[metric] = new
            utils.save_model(model, save_dir / f"detector-{metric}.pt")

    return previous_best


def create_data_loader(
    train_cfg: dict,
    data_dir: pathlib.Path,
    metadata_path: pathlib.Path,
    batch_size: int,
    world_size: int,
    shuffle: bool
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.Sampler]:

    assert data_dir.is_dir(), data_dir

    dataset = datasets.DetDataset(
        data_dir,
        metadata_path=metadata_path,
        img_ext=generate_config.IMAGE_EXT,
        img_width=512,
        img_height=512,
    )

    # If using distributed training, use a DistributedSampler to load exclusive sets
    # of data.
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, sampler=sampler
    )
    return loader, sampler


def create_optimizer(optim_cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """ Take in optimizer config and create the optimizer for training. """
    name = optim_cfg.get("type", None)
    if name.lower() == "sgd":
        lr = float(optim_cfg["lr"])
        momentum = float(optim_cfg["momentum"])
        weight_decay = float(optim_cfg["weight_decay"])
        optimizer = torch.optim.SGD(
            add_weight_decay(model, weight_decay),
            lr=lr,
            momentum=momentum,
            weight_decay=0,
            nesterov=True,
        )
    elif name.lower() == "rmsprop":
        lr = float(optim_cfg["lr"])
        momentum = float(optim_cfg["momentum"])
        weight_decay = float(optim_cfg["weight_decay"])
        optimizer = torch.optim.RMSprop(
            add_weight_decay(model, weight_decay), lr=lr, momentum=momentum, weight_decay=0
        )
    else:
        raise ValueError(f"Improper optimizer supplied: {name}.")

    return optimizer


def add_weight_decay(
    model: torch.nn.Module, weight_decay: float = 1e-5
) -> List[Dict[str, Any]]:
    """ Add weight decay to only the dense layer weights, not their biases 
    or not the norm layers.

    Args:
        model: The model where the weight decay is applied.
        weight_decay: How much decay to apply.
    Returns:
        A list of dictionaries encapsulating which params need weight decay.
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


if __name__ == "__main__":
    torch.cuda.random.manual_seed(42)
    torch.random.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description="Trainer code for RetinaNet-based detection models."
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

    use_cuda = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_cuda else 1  # GPUS or a CPU

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12345"
    torch.multiprocessing.spawn(
        train,
        (world_size, model_cfg, train_cfg, save_dir),
        nprocs=world_size,
        join=True,
    )

    # Create tar archive.
    with tarfile.open(save_dir / "detector.tar.gz", mode="w:gz") as tar:
        for model_file in save_dir.glob("*"):
            tar.add(model_file, arcname=model_file.name)
