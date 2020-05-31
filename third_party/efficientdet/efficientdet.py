from typing import List
import collections

import torch

import efficientnet, bifpn
from third_party import (
    retinanet_head,
    postprocess,
    regression,
    anchors,
)

_MODEL_SCALES = {
    # (resolution, backbone, bifpn channels, num bifpn layers, head layers)
    "efficientdet-b0": (512, "efficientnet-b0", 64, 3, 3),
    "efficientdet-b1": (640, "efficientnet-b1", 88, 4, 3),
    "efficientdet-b2": (768, "efficientnet-b2", 112, 5, 3),
    "efficientdet-b3": (896, "efficientnet-b3", 160, 6, 4),
    "efficientdet-b4": (1024, "efficientnet-b4", 224, 7, 4),
    "efficientdet-b5": (1280, "efficientnet-b5", 288, 7, 4),
}


class EfficientDet(torch.nn.Module):
    """ Implementatin of EfficientDet originally proposed in 
    [1] Mingxing Tan, Ruoming Pang, Quoc Le.
    EfficientDet: Scalable and Efficient Object Detection.
    CVPR 2020, https://arxiv.org/abs/1911.09070 """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "efficientdet-b0",
        levels: List[int] = [3, 4, 5, 6, 7],
        num_levels_extracted: int = 3,
        num_detections_per_image: int = 3,
        score_threshold: float = 0.05,
    ) -> None:
        """ 
        Args:
            params: (bifpn channels, num bifpns, num retina net convs)
        Usage:
        >>> net = EfficientDet(10).train()
        >>> with torch.no_grad():
        ...     out = net(torch.randn(1, 3, 512, 512))
        >>> len(out)
        2
        """
        super().__init__()
        self.levels = levels
        self.num_pyramids = len(levels)
        self.num_levels_extracted = num_levels_extracted
        self.num_detections_per_image = num_detections_per_image

        self.backbone = efficientnet.EfficientNet(
            _MODEL_SCALES[backbone][1], num_classes=num_classes
        )
        self.backbone.delete_classification_head()

        # Get the output feature for the pyramids we need
        features = self.backbone.get_pyramid_channels()[-num_levels_extracted:]

        params = _MODEL_SCALES[backbone]

        # Create the BiFPN with the supplied parameter options.
        self.fpn = bifpn.BiFPN(
            in_channels=features,
            out_channels=params[2],
            num_bifpns=params[3],
            levels=[3, 4, 5],
            bifpn_height=5,
        )
        self.anchors = anchors.AnchorGenerator(
            img_height=params[0],
            img_width=params[0],
            pyramid_levels=levels,
            anchor_scales=[1.0, 1.25, 1.50],
        )
        # Create the retinanet head.
        self.retinanet_head = retinanet_head.RetinaNetHead(
            num_classes,
            in_channels=params[2],
            anchors_per_cell=self.anchors.num_anchors_per_cell,
            num_convolutions=params[4],
        )

        if torch.cuda.is_available():
            self.anchors.all_anchors = self.anchors.all_anchors.cuda()
            self.anchors.anchors_over_all_feature_maps = [
                anchors.cuda() for anchors in self.anchors.anchors_over_all_feature_maps
            ]

        self.postprocess = postprocess.PostProcessor(
            num_classes=num_classes,
            anchors_per_level=self.anchors.anchors_over_all_feature_maps,
            regressor=regression.Regressor(),
            max_detections_per_image=num_detections_per_image,
            score_threshold=score_threshold,
        )

        self.eval()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        levels = self.backbone.forward_pyramids(x)
        # Only keep the levels specified during construction.
        levels = collections.OrderedDict(
            [item for item in levels.items() if item[0] in self.levels]
        )

        levels = self.fpn(levels)
        classifications, regressions = self.retinanet_head(levels)

        if self.training:
            return classifications, regressions
        else:
            return self.postprocess(classifications, regressions)
