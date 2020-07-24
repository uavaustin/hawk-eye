""" This is a collections of layers which expand the incoming feature layers
into box regressions and class probabilities. """

import copy
import collections
import math
from typing import Tuple, List

import torch


def depthwise(in_channels: int, out_channels: int):
    """ A depthwise separable linear layer. """
    return [
        torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=in_channels,
        ),
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
    ]


def conv3x3(in_channels: int, out_channels: int):
    """ Simple Conv2d layer. """
    return [
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
    ]


class RetinaNetHead(torch.nn.Module):
    """ This model head contains two components: classification and box regression.
    See the original RetinaNet paper for more details,
    https://arxiv.org/pdf/1708.02002.pdf. The official efficientdet implementation also
    applies separate batch norms per level. """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        anchors_per_cell: int,
        num_convolutions: int = 4,  # Original paper proposes 4 convs
        use_dw: bool = False,
        num_levels: int = 5,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels

        if use_dw:
            conv = depthwise
        else:
            conv = conv3x3

        # Create the two subnets, but separate batch norms for each layer and each level.
        self.classification_subnet = torch.nn.ModuleList()
        self.classification_bns = torch.nn.ModuleList()
        self.classification_acts = torch.nn.ModuleList()
        for idx in range(num_convolutions):
            self.classification_subnet.extend(conv(in_channels, in_channels))
            self.classification_acts.append(torch.nn.ReLU(inplace=True))

        for conv_idx in range(num_levels):
            level_bns = torch.nn.ModuleList()
            for level_idx in range(num_convolutions):
                level_bns.append(
                    torch.nn.BatchNorm2d(in_channels, eps=1e-3, momentum=1e-2)
                )
            self.classification_bns.append(level_bns)

        # NOTE same basic architecture between box regression and classification
        self.regression_subnet = copy.deepcopy(self.classification_subnet)
        self.regression_bns = copy.deepcopy(self.classification_bns)
        self.regression_acts = copy.deepcopy(self.classification_acts)

        # Here is where the two subnets diverge. The classification net expands the input
        # into (anchors_num * num_classes) filters because it predicts 'the probability
        # of object presence at each spatial postion for each of the A anchors
        self.cls_pred = torch.nn.Sequential(
            *conv(in_channels, anchors_per_cell * num_classes)
        )

        # The regerssion expands the input into (4 * A) channels. So each x,y in the
        # feature map has (4 * A) channels where 4 represents (dx, dy, dw, dh). The
        # regressions for each component of each anchor box.
        self.reg_pred = torch.nn.Sequential(*conv(in_channels, anchors_per_cell * 4))

        # Initialize the model
        if not use_dw:
            for subnet in [
                self.regression_subnet,
                self.classification_subnet,
                self.cls_pred,
                self.reg_pred,
            ]:
                for layer in subnet.modules():
                    if isinstance(layer, torch.nn.Conv2d):
                        torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                        torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve classification stability.
        prior_prob = 0.01
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_pred[-1].bias, bias_value)

    def __call__(
        self, feature_maps: collections.OrderedDict
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """ Applies the regression and classification subnets to each of the
        incoming feature maps. """
        assert len(feature_maps) == self.num_levels

        bbox_regressions_inter = []
        classifications_inter = []
        bbox_regressions = []
        classifications = []

        for level_idx, level in enumerate(feature_maps.values()):

            for conv_idx, (conv, act) in enumerate(
                zip(self.classification_subnet, self.classification_acts)
            ):
                if conv_idx == 0:
                    classifications_inter.append(level)

                classifications_inter.append(
                    act(
                        self.classification_bns[level_idx][conv_idx](
                            conv(classifications_inter[-1])
                        )
                    )
                )
            classifications.append(self.cls_pred(classifications_inter[-1]))

            for conv_idx, (conv, act) in enumerate(
                zip(self.regression_subnet, self.regression_acts)
            ):
                if conv_idx == 0:
                    bbox_regressions_inter.append(level)

                bbox_regressions_inter.append(
                    act(
                        self.regression_bns[level_idx][conv_idx](
                            conv(bbox_regressions_inter[-1])
                        )
                    )
                )
            bbox_regressions.append(self.reg_pred(bbox_regressions_inter[-1]))

        return classifications, bbox_regressions
