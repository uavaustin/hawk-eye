""" A detector model which wraps around a feature extraction backbone, fpn, and RetinaNet
head.This allows for easy interchangeability during experimentation and a reliable way to
load saved models. """

import collections
import pathlib
from typing import List
import yaml

import torch

from hawk_eye.core import asset_manager
from hawk_eye.core import fpn
from third_party.vovnet import vovnet
from third_party.models import postprocess
from third_party.models import regression
from third_party.models import anchors
from third_party.models import retinanet_head


class Detector(torch.nn.Module):
    def __init__(
        self,
        model_params: dict = None,
        timestamp: str = None,
        confidence: float = 0.05,
        num_detections_per_image: int = 100,
        half_precision: bool = False,
    ) -> None:
        super().__init__()
        self.half_precision = half_precision
        self.num_detections_per_image = num_detections_per_image
        self.confidence = confidence

        if model_params is None and timestamp is None:
            raise ValueError("Must supply either model timestamp or backbone to load")

        # If a timestamp is given, download it.
        if timestamp is not None:

            if pathlib.Path(timestamp).is_dir():
                model_path = pathlib.Path(timestamp)
            else:
                # Download the model. This has the yaml containing the backbone.
                model_path = asset_manager.download_model("detector", timestamp)

            config = yaml.safe_load((model_path / "config.yaml").read_text())
            model_params = config["model"]
            self._load_params(config["model"])
        else:
            self._load_params(model_params)

        self.backbone = self._load_backbone(self.backbone)
        self.backbone.delete_classification_head()

        self.fpn = self._load_fpn(self.fpn_type, self.backbone.get_pyramid_channels())

        assert len(self.anchor_sizes) == len(self.fpn_levels)
        self.anchors = anchors.AnchorGenerator(
            img_height=self.img_height,
            img_width=self.img_width,
            pyramid_levels=self.fpn_levels,
            aspect_ratios=self.aspect_ratios,
            sizes=self.anchor_sizes,
            anchor_scales=self.anchor_scales,
        )

        # Create the retinanet head.
        self.retinanet_head = retinanet_head.RetinaNetHead(
            self.num_classes,
            in_channels=self.fpn_channels,
            anchors_per_cell=self.anchors.num_anchors_per_cell,
            num_convolutions=self.num_head_convs,
            use_dw=self.retinanet_head_dw,
        )

        if torch.cuda.is_available():
            self.anchors.all_anchors = self.anchors.all_anchors.cuda()
            self.anchors.anchors_over_all_feature_maps = [
                anchors.cuda() for anchors in self.anchors.anchors_over_all_feature_maps
            ]
            self.cuda()

        self.image_size = model_params.get("image_size", 512)
        self.postprocess = postprocess.PostProcessor(
            num_classes=self.num_classes,
            image_size=self.image_size,
            all_anchors=self.anchors.all_anchors,
            regressor=regression.Regressor(),
            max_detections_per_image=num_detections_per_image,
            score_threshold=confidence,
            nms_threshold=0.2,
        )

        # After all the components are initialized, load the weights.
        if timestamp is not None:
            self.load_state_dict(
                torch.load(model_path / "min-loss.pt", map_location="cpu")
            )

        self.eval()

    def _load_params(self, config: dict) -> None:
        """Function to parse the model definition params for later building."""
        self.backbone = config.get("backbone")
        assert self.backbone is not None, "Please supply a backbone!"

        fpn_params = config.get("fpn")
        head_params = config.get("retinanet_head")
        assert fpn_params is not None, "Must supply a fpn section in the config."

        self.fpn_type = fpn_params.get("type")
        assert self.fpn_type is not None, "Must supply a fpn type."

        self.fpn_channels = fpn_params.get("num_channels", 128)
        self.fpn_use_dw = fpn_params.get("use_dw", False)
        self.num_head_convs = head_params.get("num_levels", 3)

        self.fpn_levels = fpn_params.get("levels", [3, 4, 5, 6, 7])
        self.retinanet_head_dw = head_params.get("use_dw")

        anchor_params = config.get("anchors")
        assert anchor_params is not None, "Please add an anchor section."
        self.aspect_ratios = anchor_params.get("aspect_ratios", [0.5, 1, 2])
        self.anchor_sizes = anchor_params.get("sizes", [16, 32, 64, 128, 256])
        self.anchor_scales = anchor_params.get("scales", [0.75, 1.0, 1.25])

        self.img_height, self.img_width = config.get("img_size", [512, 512])
        self.num_classes = config.get("num_classes", 10)

    def _load_backbone(self, backbone: str) -> torch.nn.Module:
        """Load the supplied backbone."""
        if "vovnet" in backbone:
            model = vovnet.VoVNet(backbone)
        else:
            raise ValueError(f"Unsupported backbone {backbone}.")

        return model

    def _load_fpn(self, fpn_name: str, features: List[int]) -> torch.nn.Module:
        if "retinanet" in fpn_name:
            fpn_ = fpn.FPN(
                in_channels=features[-3:],
                out_channels=self.fpn_channels,
                num_levels=len(self.fpn_levels),
                use_dw=self.fpn_use_dw,
            )

        return fpn_

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        levels = self.backbone.forward_pyramids(x)
        # Only keep the levels specified during construction.
        levels = collections.OrderedDict(
            [item for item in levels.items() if item[0] in self.fpn_levels]
        )
        levels = self.fpn(levels)
        classifications, regressions = self.retinanet_head(levels)

        if self.training:
            return classifications, regressions
        else:
            return self.postprocess(classifications, regressions)
