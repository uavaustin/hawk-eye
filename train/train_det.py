#!/usr/bin/env python3
""" Generalized object detection training script. This script will use as many gpus as
PyTorch can find. If Nvidia's Apex is available, that will be used for mixed precision
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

from train import datasets, collate
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
    serialized for the pycocotools package. 
    
    Args:
        bboxes: A list of bounding boxes to save.
        image_ids: The ids of the images which the boxes correspond to. We need these 
            in order to match predictions to ground truth.
        image_size: The boxes are original normalized, so we need to project them to 
            image coordinates.

    Returns:
        A list of box dictionaries in the COCO format.
    """

    detections = []
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

    # If we are using distributed training, initialize the backend through which process
    # can communicate to each other.
    if world_size > 1:
        assert (
            torch.distributed.is_nccl_available()
        ), "NCCL must be avaliable for parallel training."
        torch.distributed.init_process_group(
            "nccl", init_method="env://", world_size=world_size, rank=local_rank,
        )
    # Load the model.
    # TODO(alex): Maybe set the number of classes based off the dataset? Reading from the
    # config could lead to hard-to-find bugs.
    model = detector.Detector(
        num_classes=len(generate_config.OD_CLASSES),
        model_params=model_cfg,
        confidence=0.05,  # TODO(alex): Make configurable?
    )
    model.to(device)
    model.train()
    print(f"Model architecture: \n {model}")

    # TODO(alex) these paths should be in the generate config
    train_batch_size = train_cfg.get("train_batch_size")
    train_loader, train_sampler = create_data_loader(
        train_cfg,
        generate_config.DATA_DIR / "detector_train/images",
        generate_config.DATA_DIR / "detector_train/train_coco.json",
        model.anchors.all_anchors,
        train_batch_size,
        world_size,
        val=False,
        image_size=model_cfg.get("image_size", 512),
    )
    eval_batch_size = train_cfg.get("eval_batch_size")
    eval_loader, _ = create_data_loader(
        train_cfg,
        generate_config.DATA_DIR / "detector_val/images",
        generate_config.DATA_DIR / "detector_val/val_coco.json",
        model.anchors.all_anchors,
        eval_batch_size,
        world_size,
        val=True,
        image_size=model_cfg.get("image_size", 512),
    )

    # Construct the optimizer and wrap with Apex if available.
    optimizer = utils.create_optimizer(train_cfg["optimizer"], model)
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

    eval_start_epoch = train_cfg.get("eval_start_epoch", 10)
    eval_results = {}

    # Create the learning rate scheduler.
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

    # Begin training. Loop over all the epochs and run through the training data, then
    # the evaluation data. Save the best weights for the various metrics we capture.
    for epoch in range(epochs):

        all_losses, clf_losses, reg_losses = [], [], []

        # Set the train loader's epoch so data will be re-shuffled.
        train_sampler.set_epoch(epoch)

        for idx, (images, gt_regressions, gt_classes) in enumerate(train_loader):

            optimizer.zero_grad()

            if use_cuda:
                images = images.to(device)
                gt_regressions = gt_regressions.to(device)
                gt_classes = gt_classes.to(device)

            # Forward pass through detector
            cls_per_level, reg_per_level = model(images)

            # Compute the losses
            cls_loss, reg_loss = losses.compute_losses(
                gt_classes=gt_classes,
                gt_anchors_deltas=gt_regressions,
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

            if idx % _LOG_INTERVAL == is_main == 0:
                print(
                    f"Epoch: {epoch} step {idx}, "
                    f"clf loss {sum(clf_losses) / len(clf_losses):.5}, "
                    f"reg loss {sum(reg_losses) / len(reg_losses):.5}, "
                    f"lr {lr:.5}"
                )

        # Call evaluation function if past eval delay.
        if is_main and epoch >= eval_start_epoch:
            print("Starting evaluation")
            model.eval()
            start = time.perf_counter()
            eval_results, improved_metics = eval(
                model, eval_loader, eval_results, save_dir
            )
            model.train()
            print(f"Evaluation took {time.perf_counter() - start:.3} seconds.")
            print(f"Improved metrics: {improved_metics}")

        if is_main:
            print(
                f"Epoch: {epoch}, Training loss {sum(all_losses) / len(all_losses):.5} \n"
                f"{eval_results}"
            )


@torch.no_grad()
def eval(
    model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    previous_best: dict,
    save_dir: pathlib.Path = None,
) -> Tuple[dict, List[str]]:
    """ Evalulate the model against the evaulation set. Save the best weights if
    specified. Use the pycocotools package for metrics.

    Args:
        model: The model to evaluate.
        eval_loader: The eval dataset loader.
        previous_best: The current best eval metrics.
        save_dir: Where to save the model weights.

    Returns:
        The updated best metrics and a list of the metrics that improved.
    """
    detections_dict: List[dict] = []
    for images, image_ids in eval_loader:
        if torch.cuda.is_available():
            images = images.cuda()
        detections = model(images)
        detections_dict.extend(
            detections_to_dict(detections, image_ids, model.module.image_size)
        )
    results = {}
    if detections_dict:
        with tempfile.TemporaryDirectory() as d:
            tmp_json = pathlib.Path(d) / "det.json"
            tmp_json.write_text(json.dumps(detections_dict))
            results = coco_eval.get_metrics(
                generate_config.DATA_DIR / "detector_val/val_coco.json", tmp_json,
            )

        # If there are the first results, set the previous to the current.
        previous_best = results if not previous_best else previous_best

    improved = []
    for (metric, old), new in zip(previous_best.items(), results.values()):
        if new >= old:
            improved.append(metric)
            previous_best[metric] = new
            utils.save_model(model, save_dir / f"detector-{metric}.pt")

    return previous_best, improved


def create_data_loader(
    train_cfg: dict,
    data_dir: pathlib.Path,
    metadata_path: pathlib.Path,
    anchors: torch.tensor,
    batch_size: int,
    world_size: int,
    val: bool,
    image_size: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.Sampler]:
    """ Simple function to create the dataloaders for training and evaluation.

    Args:
        training_cfg: The parameters related to the training regime.
        data_dir: The directory where the images are located.
        metadata_path: The path to the COCO metadata json.
        anchors: The tensor of anchors in the model.
        batch_size: The loader's batch size.
        world_size: World size is needed to determine if a distributed sampler is needed.
        val: Wether or not this loader is for validation.
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
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=val)

    if val:
        collate_fn = collate.CollateVal()
    else:
        collate_fn = collate.Collate(
            num_classes=len(generate_config.OD_CLASSES),
            original_anchors=anchors,
            image_size=512,
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=sampler,
        collate_fn=collate_fn,
    )
    return loader, sampler


if __name__ == "__main__":
    # Training will probably always be undeterministic due to async CUDA calls,
    # but this gets us a bit closer to repeatability.
    torch.cuda.random.manual_seed(42)
    torch.random.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description="Trainer code for RetinaNet-based detection models."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=pathlib.Path,
        help="Path to yaml model definition.",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser()
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
