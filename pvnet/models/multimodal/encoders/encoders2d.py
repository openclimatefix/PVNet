"""Encoder modules for the satellite/NWP data.

These networks naively stack the sequences into extra channels before putting through their
architectures.
"""

from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Type, Union

import torch
from torch import Tensor, nn
from torchvision.models.convnext import CNBlock, CNBlockConfig, LayerNorm2d
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.utils import _log_api_usage_once

from pvnet.models.multimodal.encoders.basic_blocks import AbstractNWPSatelliteEncoder


class NaiveEfficientNet(AbstractNWPSatelliteEncoder):
    """An implementation of EfficientNet from `efficientnet_pytorch`.

    This model is quite naive, and just stacks the sequence into channels.
    """

    def __init__(
        self,
        sequence_length: int,
        image_size_pixels: int,
        in_channels: int,
        out_features: int,
        model_name: str = "efficientnet-b0",
    ):
        """An implementation of EfficientNet from `efficientnet_pytorch`.

        This model is quite naive, and just stacks the sequence into channels.

        Args:
            sequence_length: The time sequence length of the data.
            image_size_pixels: The spatial size of the image. Assumed square.
            in_channels: Number of input channels.
            out_features: Number of output features.
            model_name: Name of EfficientNet model to construct.

        Notes:
            The `efficientnet_pytorch` package must be installed to use `EncoderNaiveEfficientNet`.
            See https://github.com/lukemelas/EfficientNet-PyTorch for install instructions.
        """

        from efficientnet_pytorch import EfficientNet

        super().__init__(sequence_length, image_size_pixels, in_channels, out_features)

        self.model = EfficientNet.from_name(
            model_name,
            in_channels=in_channels * sequence_length,
            image_size=image_size_pixels,
            num_classes=out_features,
        )

    def forward(self, x):
        """Run model forward"""
        bs, s, c, h, w = x.shape
        x = x.reshape((bs, s * c, h, w))
        return self.model(x)


class NaiveResNet(nn.Module):
    """A ResNet model modified from one in torchvision [1].

    Modified allow different number of input channels. This model is quite naive, and just stacks
    the sequence into channels.

    Example use:
        ```
        resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
        resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
        ```

    Sources:
         [1] https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
         [2] https://pytorch.org/hub/pytorch_vision_resnet
    """

    def __init__(
        self,
        sequence_length: int,
        image_size_pixels: int,
        in_channels: int,
        out_features: int,
        layers: List[int] = [2, 2, 2, 2],
        block: str = "bottleneck",
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        """A ResNet model modified from one in torchvision [1].

        Args:
            sequence_length: The time sequence length of the data.
            image_size_pixels: The spatial size of the image. Assumed square.
            in_channels: Number of input channels.
            out_features: Number of output features.
            layers: See [1] and [2].
            block: See [1] and [2].
            zero_init_residual: See [1] and [2].
            groups: See [1] and [2].
            width_per_group: See [1] and [2].
            replace_stride_with_dilation: See [1] and [2].
            norm_layer: See [1] and [2].

        Sources:
             [1] https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
             [2] https://pytorch.org/hub/pytorch_vision_resnet
        """
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # Account for stacking sequences into more channels
        in_channels = in_channels * sequence_length

        block = {
            "basic": BasicBlock,
            "bottleneck": Bottleneck,
        }[block]

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, out_features)
        self.final_act = nn.LeakyReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an
        # identity. This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.final_act(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """Run model forward"""
        bs, s, c, h, w = x.shape
        x = x.reshape((bs, s * c, h, w))
        return self._forward_impl(x)


class NaiveConvNeXt(nn.Module):
    """A NaiveConvNeXt model [1] modified from one in torchvision [2].

    Mopdified to allow different number of input channels, and smaller spatial inputs. This model is
    quite naive, and just stacks the sequence into channels.

    Example usage:
        ```
        block_setting = [
            CNBlockConfig(96, 192, 3),
            CNBlockConfig(192, 384, 3),
            CNBlockConfig(384, 768, 9),
            CNBlockConfig(768, None, 3),
        ]

        sequence_len = 12
        channels = 2
        pixels=24

        convnext_tiny = ConvNeXt(
            sequence_length=12,
            image_size_pixels=24,
            in_channels=2,
            out_features=128,
            block_setting=block_setting,
            stochastic_depth_prob=0.1,
        )
        ```

    Sources:
        [1] https://arxiv.org/abs/2201.03545
        [2] https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
        [3] https://pytorch.org/vision/main/models/convnext.html

    """

    def __init__(
        self,
        sequence_length: int,
        image_size_pixels: int,
        in_channels: int,
        out_features: int,
        block_setting: List[CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        """A ConvNeXt model [1] modified from one in torchvision [2].

        Args:
            sequence_length: The time sequence length of the data.
            image_size_pixels: The spatial size of the image. Assumed square.
            in_channels: Number of input channels.
            out_features: Number of output features.
            block_setting: See [2] and [3].
            stochastic_depth_prob: See [2] and [3].
            layer_scale: See [2] and [3].
            block: See [2] and [3].
            norm_layer: See [2] and [3].
            **kwargs: See [2] and [3].

        Sources:
            [1] https://arxiv.org/abs/2201.03545
            [2] https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
            [3] https://pytorch.org/vision/main/models/convnext.html
        """
        super().__init__()
        _log_api_usage_once(self)

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (
            isinstance(block_setting, Sequence)
            and all([isinstance(s, CNBlockConfig) for s in block_setting])
        ):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        # Account for stacking sequences into more channels
        in_channels = in_channels * sequence_length

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                in_channels,
                firstconv_output_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels
            if lastblock.out_channels is not None
            else lastblock.input_channels
        )
        self.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels),
            nn.Flatten(1),
            nn.Linear(lastconv_output_channels, out_features),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Run model forward"""
        bs, s, c, h, w = x.shape
        x = x.reshape((bs, s * c, h, w))
        return self._forward_impl(x)
