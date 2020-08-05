#!/usr/bin/env python3
""" Train a classifier to classify images as backgroud or targets. This script will use
distributed training if available any PyTorch AMP. """

import argparse
import datetime
import pathlib
from typing import Tuple
import tarfile
import shutil
import os
import yaml

import torch
import numpy as np

from train import datasets
from train.train_utils import utils
from core import classifier
from data_generation import generate_config
from third_party.models import losses

_LOG_INTERVAL = 50
_SAVE_DIR = pathlib.Path("~/runs/uav-clf").expanduser()


def train(
    local_rank: int,
    world_size: int,
    model_cfg: dict,
    train_cfg: dict,
    save_dir: pathlib.Path = None,
) -> None:
    # Do some general s
    # etup. When using distributed training and Apex, the device needs
    # to be set before loading the model.
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    is_main = local_rank == 0

    # If we are using distributed training, initialize the backend through which process
    # can communicate to each other.
    if world_size > 1:
        torch.distributed.init_process_group(
            "nccl", init_method="env://", world_size=world_size, rank=local_rank
        )

    # TODO(alex) these paths should be in the generate config
    batch_size = train_cfg.get("batch_size", 4)
    train_loader = create_data_loader(
        batch_size, generate_config.DATA_DIR / "clf_train"
    )
    eval_loader = create_data_loader(batch_size, generate_config.DATA_DIR / "clf_val")

    highest_score = {"base": 0, "swa": 0}

    clf_model = classifier.Classifier(
        backbone=model_cfg.get("backbone", None), num_classes=2
    )
    clf_model.to(device)
    if is_main:
        print("Model: \n", clf_model)

    optimizer = utils.create_optimizer(train_cfg["optimizer"], clf_model)
    lr_params = train_cfg.get("lr_schedule", {})

    # Setup mixed precision abilities if specified.
    scaler = torch.cuda.amp.GradScaler(init_scale=1)

    epochs = train_cfg.get("epochs", 0)
    assert epochs > 0, "Please supply epoch > 0"

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
    global_step = 0

    for epoch in range(epochs):
        all_losses = []

        for idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            global_step += 1

            # BHWC -> BCHW
            data = data.permute(0, 3, 1, 2)
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            # Compute one hot encoding for focal loss function.
            labels_one_hot = torch.zeros(
                labels.shape[0], 2, device=labels.device
            ).scatter_(1, labels, 1)

            with torch.cuda.amp.autocast():
                out = clf_model(data)
                loss = losses.sigmoid_focal_loss(out, labels_one_hot, reduction="mean")
            all_losses.append(loss.item())

            # Propogate the gradients back through the model and update the weights.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            if idx % _LOG_INTERVAL == 0 and is_main:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch: {epoch} step {idx}, loss "
                    f"{sum(all_losses) / len(all_losses):.5}. lr: {lr:.4}"
                )

        # Call evaluation function
        clf_model.eval()
        highest_score = eval_acc = eval(
            clf_model, eval_loader, device, world_size, highest_score, save_dir
        )
        clf_model.train()

        if is_main:
            print(
                f"Epoch: {epoch}, Training loss {sum(all_losses) / len(all_losses):.5} \n"
                f"Base accuracy: {eval_acc['base']:.4} \n"
            )


@torch.no_grad()
def eval(
    clf_model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    device: torch.device,
    world_size: int,
    previous_best: dict = None,
    save_dir: pathlib.Path = None,
) -> float:
    """ Evalulate the model against the evaulation set. Save the best
    weights if specified.

    Args:
        model: The model to evaluate.
        eval_loader: The eval dataset loader.
        previous_best: The current best eval metrics.
        save_dir: Where to save the model weights.

    Returns:
        The updated best metrics and a list of the metrics that improved.
    """
    num_correct = total_num = 0

    for data, labels in eval_loader:
        data = data.permute(0, 3, 1, 2)
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        out = clf_model(data)
        _, predicted = torch.max(out.data, 1)

        num_correct += (predicted == labels).sum().item()
        total_num += data.shape[0]

    accuracy = {"base": num_correct / total_num}
    # Make sure processes get to this point.
    if world_size > 1:
        torch.distributed.barrier()

    if accuracy["base"] > previous_best["base"]:
        print(f"Saving model with accuracy {accuracy}.")

        # Delete the previous best
        utils.save_model(clf_model, save_dir / "classifier.pt")

        return accuracy
    else:
        return previous_best


def create_data_loader(
    batch_size: dict, data_dir: pathlib.Path
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    assert data_dir.is_dir(), data_dir

    dataset = datasets.ClfDataset(data_dir, img_ext=generate_config.IMAGE_EXT)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=True
    )
    return loader


if __name__ == "__main__":
    # Training will probably always be undeterministic due to async CUDA calls,
    # but this gets us a bit closer to repeatability.
    torch.cuda.random.manual_seed(42)
    torch.random.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description="Trainer code for classifcation models."
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

    # Copy in this config file to the save dir. The config file will be used to load the
    # saved model.
    save_dir = _SAVE_DIR / (
        datetime.datetime.now().isoformat().split(".")[0].replace(":", "-")
    )
    save_dir.mkdir(parents=True)
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
    with tarfile.open(save_dir / f"{save_dir.name}.tar.gz", mode="w:gz") as tar:
        for model_file in save_dir.glob("*"):
            tar.add(model_file, arcname=model_file.name)

    print(f"Saved model to {save_dir / save_dir.name}.tar.gz")
