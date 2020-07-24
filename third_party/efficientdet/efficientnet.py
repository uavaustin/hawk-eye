""" Code to implement an efficientnet in PyTorch.

The architecture is based on scaling four model parameters:
depth (more layers), width (more filters per layer), resolution
(larger input images), and dropout. """

import collections
import math
import dataclasses
from typing import Tuple, List

import torch
import numpy as np


@dataclasses.dataclass
class ModelScales:
    width_coefficient: int
    depth_coefficient: int
    resolution: int
    dropout_rate: int


# Seen here:
# https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_builder.py#L35
_MODEL_SCALES = {
    # (width_coefficient, depth_coefficient, resolution, dropout_rate)
    "efficientnet-lite0": ModelScales(1.0, 1.0, 224, 0.2),
    "efficientnet-lite1": ModelScales(1.0, 1.1, 240, 0.2),
    "efficientnet-lite2": ModelScales(1.1, 1.2, 260, 0.3),
    "efficientnet-lite3": ModelScales(1.2, 1.4, 280, 0.3),
    "efficientnet-lite4": ModelScales(1.4, 1.8, 300, 0.3),
    "efficientnet-b0": ModelScales(1.0, 1.0, 224, 0.2),
    "efficientnet-b1": ModelScales(1.0, 1.1, 240, 0.2),
    "efficientnet-b2": ModelScales(1.1, 1.2, 260, 0.3),
    "efficientnet-b3": ModelScales(1.2, 1.4, 300, 0.3),
    "efficientnet-b4": ModelScales(1.4, 1.8, 380, 0.4),
    "efficientnet-b5": ModelScales(1.6, 2.2, 456, 0.4),
    "efficientnet-b6": ModelScales(1.8, 2.6, 528, 0.5),
    "efficientnet-b7": ModelScales(2.0, 3.1, 600, 0.5),
    "efficientnet-b8": ModelScales(2.2, 3.6, 672, 0.5),
    "efficientnet-l2": ModelScales(4.3, 5.3, 800, 0.5),
}


# These are the default parameters for the model's mobile inverted residual bottleneck
# layers.
@dataclasses.dataclass
class MBConvBlockArgs:
    kernel_size: int
    repeats: int
    filters_in: int
    filters_out: int
    expand_ratio: float
    strides: int
    se_ratio: float


# TODO(alex): Create a classification specific set of block args with smaller channels depths
# Taken from official implementation:
# https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_builder.py#L170
_DEFAULT_BLOCKS_ARGS = [
    MBConvBlockArgs(3, 1, 32, 16, 1, 1, 0.25),
    MBConvBlockArgs(3, 2, 16, 24, 6, 2, 0.25),
    MBConvBlockArgs(5, 2, 24, 40, 6, 2, 0.25),
    MBConvBlockArgs(3, 3, 40, 80, 6, 1, 0.25),
    MBConvBlockArgs(5, 3, 80, 112, 6, 1, 0.25),
    MBConvBlockArgs(5, 4, 112, 192, 6, 2, 0.25),
    MBConvBlockArgs(3, 1, 192, 320, 6, 1, 0.25),
]

# https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_builder.py#L185
_BATCH_NORM_MOMENTUM = 1e-2
_BATCH_NORM_EPS = 1e-3


class Swish(torch.nn.Module):
    """ Swish activation function presented here:
    https://arxiv.org/pdf/1710.05941.pdf. """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid_(x)


def get_activiation(lite: bool) -> torch.nn.Module:
    if lite:
        return torch.nn.ReLU(inplace=True)
    else:
        return Swish()


def round_filters(
    filters: int, scale: float, skip: bool = False, min_depth: int = 8
) -> int:
    """ This function is taken from the original tf repo. It ensures that all layers have
    a channel depth that is divisible by 8. It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py. """

    if skip:
        return filters

    filters *= scale
    new_filters = max(min_depth, int(filters + min_depth / 2) // min_depth * min_depth)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += min_depth

    return int(new_filters)


def round_repeats(repeats: int, depth_multiplier: int) -> int:
    """ 
    Round off the number of repeats. This determine how many times to repeat a block.
    """
    return int(math.ceil(depth_multiplier * repeats))


def depthwise(channels: int, kernel_size: int, stride: int) -> List[torch.nn.Module]:
    """ Depthwise convolution where the number of groups == number of filters. """
    return [
        torch.nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=channels,
            padding=kernel_size // 2,
            bias=False,
        ),
        torch.nn.BatchNorm2d(
            num_features=channels, momentum=_BATCH_NORM_MOMENTUM, eps=_BATCH_NORM_EPS
        ),
    ]


# TODO(alex): Look to add the more efficient ECA attention layer here.
class SqueezeExcitation(torch.nn.Module):
    """  See here for one of the original implementations:
    https://arxiv.org/pdf/1709.01507.pdf. The layer 'adaptively recalibrates
    channel-wise feature responses by explicitly  modeling interdependencies
    between channels.' """

    def __init__(
        self, expanded_channels: int, in_channels: int, se_ratio: float
    ) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),  # 1 x 1 x in_channels
            torch.nn.Conv2d(
                in_channels=expanded_channels,
                out_channels=max(1, int(in_channels * se_ratio)),  # Squeeze filters
                kernel_size=1,
                stride=1,
                bias=True,
            ),
            Swish(),
            torch.nn.Conv2d(
                in_channels=max(1, int(in_channels * se_ratio)),  # Expand out
                out_channels=expanded_channels,
                kernel_size=1,
                stride=1,
                bias=True,
            ),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply the squeezing and excitation, then elementwise multiplacation of
        the excitation 1 x 1 x out_channels tensor. """
        return x * torch.sigmoid_(self.layers(x))


class MBConvBlock(torch.nn.Module):
    """ Mobile inverted residual bottleneck layer. """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        expand_ratio: int,
        stride: int,
        se_ratio: int,
        is_lite: bool,
    ) -> None:
        """ Args:
        in_channels: The number of input channels to the block.
        out_channels: The number of output channels from the block.
        kernel_size: Size of the convolutional kernel.
        expand_ratio: Ratio to expand the number of channels to.
        stride: Convolutional stride.
        se_ratio: The channel depth reduction ratio to apply during sequeeze excitation.
        is_lite: Wether model it lite version. Effects SE layer andactivation.
        """
        super().__init__()
        self.skip = in_channels == out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # NOTE get expanded channels
        expanded_channels = in_channels * expand_ratio
        self.layers = []

        # Add expansion layer if expansion is required.
        if expand_ratio != 1:
            self.layers += [
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=expanded_channels,
                    kernel_size=1,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(
                    num_features=expanded_channels,
                    momentum=_BATCH_NORM_MOMENTUM,
                    eps=_BATCH_NORM_EPS,
                ),
                get_activiation(is_lite),
            ]
        self.layers.extend(
            [
                *depthwise(
                    channels=expanded_channels, kernel_size=kernel_size, stride=stride
                ),
                get_activiation(is_lite),
            ]
        )

        if is_lite:
            self.layers.append(
                SqueezeExcitation(expanded_channels, in_channels, se_ratio),
            )

        self.layers += [
            torch.nn.Conv2d(
                in_channels=expanded_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=True,
            ),
            torch.nn.BatchNorm2d(
                num_features=out_channels,
                momentum=_BATCH_NORM_MOMENTUM,
                eps=_BATCH_NORM_EPS,
            ),
        ]
        self.layers = torch.nn.Sequential(*self.layers)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(alex): look to add drop connect here.
        out = self.layers(x)
        return out


class EfficientNet(torch.nn.Module):
    def __init__(self, backbone: str, num_classes: int) -> None:
        """ Args:
        backbone: The model type to construct. See _MODEL_SCALES.
        num_classes: How many classes the classifier will predict.

        Usage:
        >>> input = torch.randn(1, 3, 244, 244, requires_grad=False)
        >>> net = EfficientNet("efficientnet-lite0", 10)
        >>> net(input).shape
        torch.Size([1, 10])

        >>> net = EfficientNet("efficientnet-b0", 10)
        >>> net(input).shape
        torch.Size([1, 10])

        >>> net = EfficientNet("efficientnet-b1", 10)
        >>> net(input).shape
        torch.Size([1, 10])

        >>> net = EfficientNet("efficientnet-b2", 10)
        >>> net(input).shape
        torch.Size([1, 10])

        >>> net = EfficientNet("efficientnet-b3", 10)
        >>> net(input).shape
        torch.Size([1, 10])

        >>> net = EfficientNet("efficientnet-b4", 10)
        >>> net(input).shape
        torch.Size([1, 10])

        >>> net = EfficientNet("efficientnet-b6", 10)
        >>> net(input).shape
        torch.Size([1, 10])

        >>> net = EfficientNet("efficientnet-b7", 10)
        >>> net(input).shape
        torch.Size([1, 10])

        >>> net = EfficientNet("efficientnet-b8", 10)
        >>> net(input).shape
        torch.Size([1, 10])

        >>> net = EfficientNet("efficientnet-l2", 10)
        >>> net(input).shape
        torch.Size([1, 10])
        """
        super().__init__()
        self.block_features = []
        self.backbone = backbone
        self.model = collections.OrderedDict()
        scale_params = _MODEL_SCALES[backbone]

        # Add the first layer, a simple 3x3 filter conv layer.
        out_channels = round_filters(32, scale_params[0], "lite" in backbone)
        self.model["stem"] = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=False,
            ),
            torch.nn.BatchNorm2d(
                out_channels, momentum=_BATCH_NORM_MOMENTUM, eps=_BATCH_NORM_EPS
            ),
            get_activiation("lite" in backbone),
        )

        # Now loop over the MBConv layer params
        for idx, mb_params in enumerate(_DEFAULT_MB_BLOCKS_ARGS):
            out_channels = round_filters(
                filters=mb_params["filters_out"], scale=scale_params[0]
            )
            in_channels = round_filters(
                filters=mb_params["filters_in"], scale=scale_params[0]
            )
            repeats = round_repeats(mb_params["repeats"], scale_params[1])

            # This first block is removed from the repeats section it may adjust the
            # input by downsampling or channel depth expansion.
            mb_conv_block = [
                MBConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=mb_params["kernel_size"],
                    expand_ratio=mb_params["expand_ratio"],
                    stride=mb_params["strides"],
                    se_ratio=mb_params["se_ratio"],
                    is_lite="lite" in backbone,
                )
            ]
            # Keep track of the filters for object detectors
            if mb_params["strides"] == 2:
                self.block_features.append(in_channels)
            elif idx == len(_DEFAULT_MB_BLOCKS_ARGS) - 1:
                self.block_features.append(out_channels)

            in_channels = out_channels
            for _ in range(repeats - 1):
                mb_conv_block.append(
                    MBConvBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=mb_params["kernel_size"],
                        expand_ratio=mb_params["expand_ratio"],
                        stride=1,
                        se_ratio=mb_params["se_ratio"],
                        is_lite="lite" in backbone,
                    )
                )

            self.model[f"Block{idx}"] = torch.nn.Sequential(*mb_conv_block)

        self.model = torch.nn.Sequential(self.model)

        # Create a classification head that can later be deleted by object detectors.
        out_channels = round_filters(1280, scale_params[0], "lite" in backbone)
        self.model_head = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(
                num_features=out_channels,
                momentum=_BATCH_NORM_MOMENTUM,
                eps=_BATCH_NORM_EPS,
            ),
            get_activiation("lite" in backbone),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Dropout(p=scale_params[-1], inplace=True),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=out_channels, out_features=num_classes),
        )

        self.apply(init)
        self.model.eval()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model_head(self.model(x))

    def forward_pyramids(self, x: torch.Tensor) -> collections.OrderedDict:
        """ Get the outputs at each level.

        Usage:
        >>> net = EfficientNet("efficientnet-b0", 2)
        >>> net.delete_classification_head()
        >>> with torch.no_grad():
        ...    levels = net.forward_pyramids(torch.randn(1, 3, 512, 512))
        >>> [level.shape[-1] for level in levels.values()]
        [256, 128, 64, 32, 16]
        >>> [level.shape[1] for level in levels.values()] == net.get_pyramid_channels()
        True
        """
        levels = collections.OrderedDict()
        levels[1] = self.model[0:2](x)
        levels[2] = self.model[2](levels[1])
        levels[3] = self.model[3](levels[2])
        levels[4] = self.model[4:6](levels[3])
        levels[5] = self.model[6:](levels[4])
        return levels

    def get_pyramid_channels(self) -> List[int]:
        """ Return the number of channels from each pyramid level. We only care
        about the output channels of each MBConv block.

        >>> net = EfficientNet("efficientnet-b0", 2)
        >>> net.get_pyramid_channels()
        [16, 24, 40, 112, 320]
        """
        return self.block_features

    def delete_classification_head(self) -> None:
        del self.model_head


def init(module: torch.nn.Module):
    """ Function called by model.apply which will recursivly traverse the model's
    modules and initialize them with the defined values. """

    if isinstance(module, torch.nn.Conv2d):
        fan_out = module.out_channels * module.kernel_size[0] * module.kernel_size[1]
        torch.nn.init.normal_(module.weight, mean=0.0, std=np.sqrt(2.0 / fan_out))

        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    elif isinstance(module, torch.nn.BatchNorm2d):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)

    elif isinstance(module, torch.nn.Linear):
        init_range = 1.0 / math.sqrt(module.weight.shape[1])
        torch.nn.init.uniform_(module.weight, -init_range, init_range)
