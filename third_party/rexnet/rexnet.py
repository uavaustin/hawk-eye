import dataclasses
import math

import torch


@dataclasses.dataclass
class ReXNetParams:
    stem_depth: int
    input_depth: int
    final_depth: int
    width_ratio: float


@dataclasses.dataclass
class BlockParams:
    repeats: int
    stride: int


_MODELS = {
    "rexnet-v1": ReXNetParams(32, 16, 180, 1.0),
    "rexnet-lite0": ReXNetParams(32, 16, 90, 0.1),
}

_BLOCKS = [
    BlockParams(1, 1),
    BlockParams(2, 2),
    BlockParams(2, 2),
    BlockParams(2, 2),
    BlockParams(2, 1),
    BlockParams(3, 2),
]

_BLOCKS_LITE = [
    BlockParams(1, 1),
    BlockParams(1, 2),
    BlockParams(1, 2),
    BlockParams(2, 2),
    BlockParams(2, 1),
    BlockParams(2, 2),
]


class Swish(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid_(x)


def get_activation(name: str) -> torch.nn.Module:
    """ Get the activation based on the model architecture. """
    if "lite" in name:
        return torch.nn.Relu(inplace=True)
    else:
        return Swish()


class SE(torch.nn.Module):
    """ Squeeze-excitation level heavly used in efficientnets for learning
    channel interdependence. """

    def __init__(self, channels: int, se_ratio: int) -> None:
        super().__init__()
        self.se = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(channels, channels // se_ratio, kernel_size=1, bias=True),
            torch.nn.BatchNorm2d(channels // se_ratio),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channels // se_ratio, channels, kernel_size=1),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid_(self.se(x))


class LinearBottleneck(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int,
        stride: int,
        use_se: bool,
        se_ratio: int,
    ) -> None:
        super().__init__()

        self.layers = []
        mid_channels = in_channels
        if expansion != 1:
            mid_channels = in_channels * expansion
            self.layers.extend(
                [
                    torch.nn.Conv2d(in_channels, mid_channels, kernel_size=1),
                    torch.nn.BatchNorm2d(mid_channels),
                    Swish(),
                ]
            )
        self.layers.extend(
            [
                torch.nn.Conv2d(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=2,
                    groups=mid_channels,
                ),
                torch.nn.BatchNorm2d(mid_channels),
            ]
        )

        if use_se:
            self.layers.extend(
                [SE(mid_channels, se_ratio), torch.nn.ReLU6(inplace=True)]
            )
        self.layers.extend(
            [
                torch.nn.Conv2d(mid_channels, out_channels, kernel_size=1),
                torch.nn.BatchNorm2d(out_channels),
            ]
        )
        self.layers = torch.nn.Sequential(*self.layers)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ReXNet(torch.nn.Module):
    def __init__(self, num_classes: int, model_type: str = "rexnet-v1") -> None:
        super().__init__()
        params = _MODELS[model_type]
        blocks = _BLOCKS if "lite" not in model_type else _BLOCKS_LITE
        self.num_layers = sum(block.repeats for block in blocks)
        stem_channels = 32 / params.width_ratio if params.width_ratio > 1.0 else 32
        print(stem_channels)
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, stem_channels, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(stem_channels),
            Swish(),
        )
        in_channels = stem_channels

        layers = []
        # Now add the linear bottleneck layers.
        for block_idx, block in enumerate(blocks):
            for idx in range(block.repeats):
                out_channels = int(
                    round(
                        params.input_depth
                        + params.final_depth * len(layers) / self.num_layers
                    )
                )
                layers.append(
                    LinearBottleneck(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        expansion=6 if block_idx else 1,
                        stride=1 if idx else block.stride,
                        use_se=True if len(layers) > 2 else False,
                        se_ratio=12,
                    )
                )
                in_channels = out_channels
        self.bottleneck_layers = torch.nn.Sequential(*layers)

        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, int(1280 * params.width_ratio), kernel_size=1),
            torch.nn.BatchNorm2d(int(1280 * params.width_ratio)),
            Swish(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Dropout(0.2, inplace=True),
            torch.nn.Linear(int(1280 * params.width_ratio), num_classes, bias=True),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(self.bottleneck_layers(self.stem(x)))
        return self.out(x.view(x.shape[0], -1))
