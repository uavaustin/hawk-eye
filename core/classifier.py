""" A classifier model which wraps around a backbone. This setup allows for easy
interchangeability during experimentation and a reliable way to load saved models. """
import pathlib
import yaml

import torch

from core import pull_assets
from third_party.rexnet import rexnet


class Classifier(torch.nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        timestamp: str = None,
        backbone: str = None,
        half_precision: bool = False,
    ) -> None:
        """
        Args:
            img_width: The width of the input images.
            img_height: The height of the input images.
            num_classes: The number of classes to predict.
            timestamp: The timestamp of the model to download from bintray.
            backbone: A string designating which model to load.
            use_cuda: Wether this model is going to be used on gpu.
            half_precision: Wether to use half precision for inference. For now
                half_precision doesn't work well with training. Maybe in PyTorch 1.6.0.
        """
        super().__init__()
        self.num_classes = num_classes
        self.use_cuda = torch.cuda.is_available()
        self.half_precision = half_precision
        if backbone is None and timestamp is None:
            raise ValueError("Must supply either model timestamp or backbone to load")

        # If a version is given, download from bintray
        if timestamp is not None:
            # Download the model or find it locally.
            model_path = pull_assets.download_model("classifier", timestamp)
            config = yaml.safe_load((model_path / "config.yaml").read_text())["model"]
            backbone = config.get("backbone", None)
            # Construct the model, then load the state
            self.model = self._load_backbone(backbone)
            self.load_state_dict(
                torch.load(model_path / "classifier.pt", map_location="cpu")
            )
            self.image_size = config["image_size"]
        else:
            # If no version supplied, just load the backbone
            self.model = self._load_backbone(backbone)

        self.model.eval()
        if self.use_cuda and self.half_precision:
            self.model.cuda()
            self.model.half()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # If using cuda and not training, assume inference.
        if self.use_cuda and self.half_precision:
            x = x.half()
        return self.model(x)

    def _load_backbone(self, backbone: str) -> torch.nn.Module:
        """ Load the supplied backbone. """
        if backbone == "rexnet":
            model = rexnet.ReXNet(num_classes=self.num_classes)
        else:
            raise ValueError(f"Unsupported backbone {backbone}.")

        return model

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """ Take in an image batch and return the class for each image. """
        if self.use_cuda and self.half_precision:
            x = x.half()
        _, predicted = torch.max(self.model(x).data, 1)
        return predicted
