#!/usr/bin/env python3
""" Generalized object detection script. This script will use as many gpus as PyTorch
can find. If Nvidia's Apex is available, that will be used for mixed precision
training to speed the process up. """

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


def detections_to_dict(
    bboxes: list, image_ids: torch.Tensor, image_size: torch.Tensor
) -> List[dict]:
    """ Used to turn raw bounding box detections into a dictionary which can be
    serialized for the pycocotools package. """
    detections: List[dict] = []
    for image_boxes, image_id in zip(bboxes, image_ids):

        for bbox in image_boxes:
            box = bbox.box
            # XYXY -> XYWH
            box[2:] -= box[:2]
            box *= image_size
            detections.append(
                {
                    "image_id": image_id.item(),
                    "category_id": bbox.class_id,
                    "bbox": box.int().tolist(),
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
    """ Main training loop that will also call out to separate evaluation function.

    Args:
        local_rank: The process's rank. 0 if there is no distributed training.
        world_size: How many devices are participating in training.
        model_cfg: The model's training params.
        train_cfg: The config of training parameters.
        save_dir: Where to save the model archive.
    """

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
        shuffle=True,
        image_size=model_cfg.get("image_size", 512)
    )
    eval_batch_size = train_cfg.get("eval_batch_size", 8)
    eval_loader, _ = create_data_loader(
        train_cfg,
        generate_config.DATA_DIR / "detector_val/images",
        generate_config.DATA_DIR / "detector_val/val_coco.json",
        eval_batch_size,
        world_size,
        shuffle=False,
        image_size=model_cfg.get("image_size", 512)
    )

    # Load the model and remove the classification head of the backbone.
    # We don't need the backbone to make classifications.
    model = detector.Detector(
        num_classes=len(generate_config.OD_CLASSES),
        model_params=model_cfg,
        confidence=0.05,  # TODO(alex): Make configurable?
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

    lr_config = train_cfg.get("lr_schedule", {})
    warm_up_percent = lr_config.get("warmup_fraction", 0) * epochs / 100
    start_lr = float(lr_config.get("start_lr"))
    max_lr = float(lr_config.get("max_lr"))
    end_lr = float(lr_config.get("end_lr"))
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=len(train_loader) * epochs,
        final_div_factor=start_lr / end_lr,
        div_factor=max_lr / start_lr,
        pct_start=warm_up_percent,
    )
    for epoch in range(epochs):

        all_losses, clf_losses, reg_losses = [], [], []

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
        if is_main:
            print("Starting evaluation")
        model.eval()
        start = time.perf_counter()
        eval_results = eval(model, eval_loader, eval_results, save_dir)
        model.train()

        if is_main:
            print(
                f"Epoch: {epoch}, Training loss {sum(all_losses) / len(all_losses):.5} \n"
                f"{eval_results}"
            )
            print(f"Evaluation took {time.perf_counter() - start:.3} seconds.")


@torch.no_grad()
def eval(
    model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    previous_best: dict,
    save_dir: pathlib.Path = None,
) -> dict:
    """ Evalulate the model against the evaulation set. Save the best weights if
    specified. Use the pycocotools package for metrics.

    Args:
        model: The model to evaluate.
        eval_loader: The eval dataset loader.
        previous_best: The current best eval metrics.
        save_dir: Where to save the model weights.

    Returns:
        The updated best metrics.
    """
    detections_dict: List[dict] = []
    for images, _, _, image_ids in eval_loader:
        if torch.cuda.is_available():
            images = images.cuda()
        detections = model(images)
        detections_dict.extend(
            detections_to_dict(detections, image_ids, model.module.image_size)
        )

    if detections_dict:
        with tempfile.TemporaryDirectory() as d:
            tmp_json = pathlib.Path(d) / "det.json"
            tmp_json.write_text(json.dumps(detections_dict))
            results = coco_eval.get_metrics(
                generate_config.DATA_DIR / "detector_val/val_coco.json",
                tmp_json,
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
    shuffle: bool,
    image_size: int
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.Sampler]:
    """ Simple function to create the dataloaders for training and evaluation.

    Args:
        training_cfg: The parameters related to the training regime.
        data_dir: The directory where the images are located.
        metadata_path: The path to the COCO metadata json.
        batch_size: The loader's batch size.
        world_size: World size is needed to determine if a distributed sampler is needed.
        shuffle: To shuffle the data or not when sampling. False for validation.
        image_size: Size of input images into the model. NOTE: we force square images.

    Returns:
        The dataloader and the loader's sampler. For _training_ we have to set the
        sampler's epoch to reshuffle.
    """

    assert data_dir.is_dir(), data_dir

    dataset = datasets.DetDataset(
        data_dir,
        metadata_path=metadata_path,
        img_ext=generate_config.IMAGE_EXT,
        img_width=image_size,
        img_height=image_size,
    )

    # If using distributed training, use a DistributedSampler to load exclusive sets
    # of data per process.
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, sampler=sampler
    )
    return loader, sampler


def create_optimizer(optim_cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """ Take in optimizer config and create the optimizer for training.

    Args:
        optim_cfg: The parameters for the optimizer generation.
        model: The model to create an optimizer for. We need to read this model's
            parameters.

    Returns:
        An optimizer for the model.
    """

    name = optim_cfg.get("type", "sgd")  # Default to SGD, most reliable.
    if name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            add_weight_decay(model, float(optim_cfg["weight_decay"])),
            lr=float(optim_cfg["lr"]),
            momentum=float(optim_cfg["momentum"]),
            weight_decay=0,
            nesterov=True,
        )
    elif name.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(
            add_weight_decay(model, float(optim_cfg["weight_decay"])),
            lr=float(optim_cfg["lr"]),
            momentum=float(optim_cfg["momentum"]),
            weight_decay=0,
        )
    else:
        raise ValueError(f"Improper optimizer supplied: {name}.")

    return optimizer


def add_weight_decay(
    model: torch.nn.Module, weight_decay: float = 1e-5
) -> List[Dict[str, Any]]:
    """ Add weight decay to only the convolutional kernels. Do not add weight decay
    to biases and BN. TensorFlow does this automatically, but for some reason this is
    the PyTorch default.

    Args:
        model: The model where the weight decay is applied.
        weight_decay: How much decay to apply.

    Returns:
        A list of dictionaries encapsulating which params need weight decay.
    """
    decay, no_decay = [], []
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
