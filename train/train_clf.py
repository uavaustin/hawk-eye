#!/usr/bin/env python3
""" Train a model to classify images as background or targets. This script will use
distributed training if available any PyTorch AMP to speed up training. """

import argparse
import datetime
import pathlib
from typing import Tuple
import tarfile
import shutil
import os
import yaml

import apex
import torch
import numpy as np

from train import datasets
from train.train_utils import utils
from core import classifier, pull_assets
from data_generation import generate_config
from third_party.models import losses

_LOG_INTERVAL = 1
_SAVE_DIR = pathlib.Path("~/runs/uav-clf").expanduser()


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
        torch.cuda.set_device(local_rank)
    is_main = local_rank == 0

    # If we are using distributed training, initialize the backend through which process
    # can communicate to each other.
    if world_size > 1:
        torch.distributed.init_process_group(
            "nccl", world_size=world_size, rank=local_rank
        )

    # TODO(alex) these paths should be in the generate config
    batch_size = train_cfg.get("batch_size", 4)
    train_loader, train_sampler = create_data_loader(
        batch_size,
        generate_config.DATA_DIR / "clf_train",
        world_size=world_size,
        val=False,
    )
    eval_loader, _ = create_data_loader(
        batch_size,
        generate_config.DATA_DIR / "clf_val",
        world_size=world_size,
        val=True,
    )

    highest_score = {"base": 0, "swa": 0}

    clf_model = classifier.Classifier(
        backbone=model_cfg.get("backbone", None), num_classes=2
    )
    clf_model.to(device)
    if is_main:
        print("Model: \n", clf_model)

    optimizer = utils.create_optimizer(train_cfg["optimizer"], clf_model)
    clf_model, optimizer = apex.amp.initialize(clf_model, optimizer, opt_level="O1")

    lr_params = train_cfg.get("lr_schedule", {})

    if world_size > 1:
        clf_model = apex.parallel.DistributedDataParallel(
            clf_model, delay_allreduce=True
        )

    epochs = train_cfg.get("epochs", 0)
    assert epochs > 0, "Please supply epoch > 0"

    # Create the learning rate scheduler.
    lr_config = train_cfg.get("lr_schedule", {})
    warm_up_percent = lr_config.get("warmup_fraction", 0)
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
        # Set the train loader's epoch so data will be re-shuffled.
        if world_size > 1:
            train_sampler.set_epoch(epoch)

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

            out = clf_model(data)
            loss = losses.sigmoid_focal_loss(
                out, labels_one_hot, reduction="sum"
            ) / max(1, labels_one_hot.shape[0])

            all_losses.append(loss.item())

            # Propogate the gradients back through the model.
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()
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
            clf_model, eval_loader, device, is_main, world_size, highest_score, save_dir
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
    is_main: bool,
    world_size: int,
    previous_best: dict = None,
    save_dir: pathlib.Path = None,
) -> float:
    """ Evaluate the model against the evaulation set. Save the best
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
    if world_size > 1:
        # Make sure processes get to this point.
        torch.distributed.barrier()

    if accuracy["base"] > previous_best["base"] and is_main:
        print(f"Saving model with accuracy {accuracy}.")

        # Delete the previous best
        utils.save_model(clf_model, save_dir / "classifier.pt")

        return accuracy
    else:
        return previous_best


def create_data_loader(
    batch_size: dict, data_dir: pathlib.Path, world_size: int, val: bool
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.Sampler]:
    """ This function acts as a DataLoader factory. Depending on the type of input
    data, either train or val, we construct a DataLoader which will be used during
    training. If we are using distributed training, a DistributedSampler is used to
    equally divide up the data between processes.

    Args:
        batch_size: Number of images per batch.
        data_dir: Where the images are located.
        world_size: How many training processes.
        val: Whether or not this is val or train data. This influences the sampler.

    Returns:
        The DataLoader and the sampler for the load. We need access to the sampler
            during _training_ so the data is shuffled each epoch.
    """
    assert data_dir.is_dir(), data_dir

    dataset = datasets.ClfDataset(data_dir, img_ext=generate_config.IMAGE_EXT)
    # If using distributed training, use a DistributedSampler to load exclusive sets
    # of data per process.
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=val)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, sampler=sampler
    )
    return loader, sampler


if __name__ == "__main__":
    # Training will probably always be undeterministic due to async CUDA calls,
    # but this gets us closer to repeatability.
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
        datetime.datetime.now().isoformat().split(".")[0].replace(":", ".")
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
    save_archive = save_dir / f"{save_dir.name}.tar.gz"
    with tarfile.open(save_archive, mode="w:gz") as tar:
        for model_file in save_dir.glob("*"):
            tar.add(model_file, arcname=model_file.name)

    # pull_assets.upload_model("classifier", save_archive)

    print(f"Saved model to {save_dir / save_dir.name}.tar.gz")
