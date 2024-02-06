"""Encoder modules for the satellite/NWP data based on 3D concolutions.
"""
from typing import List, Union

import torch
from torch import nn
from torchvision.transforms import CenterCrop

from pvnet.models.multimodal.encoders.basic_blocks import (
    AbstractNWPSatelliteEncoder,
    ResidualConv3dBlock,
)


class DefaultPVNet(AbstractNWPSatelliteEncoder):
    """This is the original encoding module used in PVNet, with a few minor tweaks."""

    def __init__(
        self,
        sequence_length: int,
        image_size_pixels: int,
        in_channels: int,
        out_features: int,
        number_of_conv3d_layers: int = 4,
        conv3d_channels: int = 32,
        fc_features: int = 128,
        spatial_kernel_size: int = 3,
        temporal_kernel_size: int = 3,
        padding: Union[int, List[int]] = (1, 0, 0),
    ):
        """This is the original encoding module used in PVNet, with a few minor tweaks.

        Args:
            sequence_length: The time sequence length of the data.
            image_size_pixels: The spatial size of the image. Assumed square.
            in_channels: Number of input channels.
            out_features: Number of output features.
            number_of_conv3d_layers: Number of convolution 3d layers that are used.
            conv3d_channels: Number of channels used in each conv3d layer.
            fc_features: number of output nodes out of the hidden fully connected layer.
            spatial_kernel_size: The spatial size of the kernel used in the conv3d layers.
            temporal_kernel_size: The temporal size of the kernel used in the conv3d layers.
            padding: The padding used in the conv3d layers. If an int, the same padding
                is used in all dimensions
        """
        super().__init__(sequence_length, image_size_pixels, in_channels, out_features)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        # Check that the output shape of the convolutional layers will be at least 1x1
        cnn_spatial_output_size = (
            image_size_pixels
            - ((spatial_kernel_size - 2 * padding[1]) - 1) * number_of_conv3d_layers
        )
        cnn_sequence_length = (
            sequence_length
            - ((temporal_kernel_size - 2 * padding[0]) - 1) * number_of_conv3d_layers
        )
        if not (cnn_spatial_output_size >= 1):
            raise ValueError(
                f"cannot use this many conv3d layers ({number_of_conv3d_layers}) with this input "
                f"spatial size ({image_size_pixels})"
            )

        conv_layers = []

        conv_layers += [
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=conv3d_channels,
                kernel_size=(temporal_kernel_size, spatial_kernel_size, spatial_kernel_size),
                padding=padding,
            ),
            nn.ELU(),
        ]
        for i in range(0, number_of_conv3d_layers - 1):
            conv_layers += [
                nn.Conv3d(
                    in_channels=conv3d_channels,
                    out_channels=conv3d_channels,
                    kernel_size=(temporal_kernel_size, spatial_kernel_size, spatial_kernel_size),
                    padding=padding,
                ),
                nn.ELU(),
            ]

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate the size of the output of the 3D convolutional layers
        cnn_output_size = conv3d_channels * cnn_spatial_output_size**2 * cnn_sequence_length

        self.final_block = nn.Sequential(
            nn.Linear(in_features=cnn_output_size, out_features=fc_features),
            nn.ELU(),
            nn.Linear(in_features=fc_features, out_features=out_features),
            nn.ELU(),
        )

    def forward(self, x):
        """Run model forward"""
        out = self.conv_layers(x)
        out = out.reshape(x.shape[0], -1)

        # Fully connected layers
        out = self.final_block(out)

        return out


class DefaultPVNet2(AbstractNWPSatelliteEncoder):
    """The original encoding module used in PVNet, with a few minor tweaks, and batchnorm."""

    def __init__(
        self,
        sequence_length: int,
        image_size_pixels: int,
        in_channels: int,
        out_features: int,
        number_of_conv3d_layers: int = 4,
        conv3d_channels: int = 32,
        fc_features: int = 128,
        batch_norm=True,
        fc_dropout=0.2,
    ):
        """The original encoding module used in PVNet, with a few minor tweaks, and batchnorm.

        Args:
            sequence_length: The time sequence length of the data.
            image_size_pixels: The spatial size of the image. Assumed square.
            in_channels: Number of input channels.
            out_features: Number of output features.
            number_of_conv3d_layers: Number of convolution 3d layers that are used.
            conv3d_channels: Number of channels used in each conv3d layer.
            fc_features: number of output nodes out of the hidden fully connected layer.
            batch_norm: Whether to include 3D batch normalisation.
            fc_dropout: Probability of an element to be zeroed before the last two fully connected
                layers.
        """
        super().__init__(sequence_length, image_size_pixels, in_channels, out_features)

        # Check that the output shape of the convolutional layers will be at least 1x1
        cnn_spatial_output_size = image_size_pixels - 2 * number_of_conv3d_layers
        if not (cnn_spatial_output_size > 0):
            raise ValueError(
                f"cannot use this many conv3d layers ({number_of_conv3d_layers}) with this input "
                f"spatial size ({image_size_pixels})"
            )

        conv_layers = [
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=conv3d_channels,
                kernel_size=(3, 3, 3),
                padding=(1, 0, 0),
            ),
            nn.LeakyReLU(),
        ]
        if batch_norm:
            # Inserted before activation using position -1
            conv_layers.insert(-1, nn.BatchNorm3d(conv3d_channels))
        for i in range(0, number_of_conv3d_layers - 1):
            conv_layers += [
                nn.Conv3d(
                    in_channels=conv3d_channels,
                    out_channels=conv3d_channels,
                    kernel_size=(3, 3, 3),
                    padding=(1, 0, 0),
                ),
                nn.LeakyReLU(),
            ]
            if batch_norm:
                # Inserted before activation using position -1
                conv_layers.insert(-1, nn.BatchNorm3d(conv3d_channels))

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate the size of the output of the 3D convolutional layers
        cnn_output_size = conv3d_channels * cnn_spatial_output_size**2 * sequence_length

        final_block = [
            nn.Linear(in_features=cnn_output_size, out_features=fc_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=fc_features, out_features=out_features),
            nn.LeakyReLU(),
        ]

        if fc_dropout > 0:
            # Insert after the linear layers
            final_block.insert(1, nn.Dropout(fc_dropout))
            final_block.insert(-1, nn.Dropout(fc_dropout))

        self.final_block = nn.Sequential(*final_block)

    def forward(self, x):
        """Run model forward"""
        out = self.conv_layers(x)
        out = out.reshape(x.shape[0], -1)

        # Fully connected layers
        out = self.final_block(out)

        return out


class EncoderUNET(AbstractNWPSatelliteEncoder):
    """An encoder based on emodifed UNet architecture.

    An encoder for satellite and/or NWP data taking inspiration from the kinds of skip
    connections in UNet. This differs from an actual UNet in that it does not have upsampling
    layers, instead it concats features from different spatial scales, and applies a few extra
    conv3d layers.
    """

    def __init__(
        self,
        sequence_length: int,
        image_size_pixels: int,
        in_channels: int,
        out_features: int,
        n_downscale: int = 3,
        res_block_layers: int = 2,
        conv3d_channels: int = 32,
        dropout_frac: float = 0.1,
    ):
        """An encoder based on emodifed UNet architecture.

        Args:
            sequence_length: The time sequence length of the data.
            image_size_pixels: The spatial size of the image. Assumed square.
            in_channels: Number of input channels.
            out_features: Number of output features.
            n_downscale: Number of conv3d and spatially downscaling layers that are used.
            res_block_layers: Number of residual blocks used after each downscale layer.
            conv3d_channels: Number of channels used in each conv3d layer.
            dropout_frac: Probability of an element to be zeroed in the residual pathways.
        """
        cnn_spatial_output = image_size_pixels // (2**n_downscale)

        if not (cnn_spatial_output > 0):
            raise ValueError(
                f"cannot use this many downscaling layers ({n_downscale}) with this input "
                f"spatial size ({image_size_pixels})"
            )

        super().__init__(sequence_length, image_size_pixels, in_channels, out_features)

        self.first_layer = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=conv3d_channels,
                kernel_size=(1, 1, 1),
                padding=(0, 0, 0),
            ),
            ResidualConv3dBlock(
                in_channels=conv3d_channels,
                n_layers=res_block_layers,
                dropout_frac=dropout_frac,
            ),
        )

        downscale_layers = []
        for _ in range(n_downscale):
            downscale_layers += [
                nn.Sequential(
                    ResidualConv3dBlock(
                        in_channels=conv3d_channels,
                        n_layers=res_block_layers,
                        dropout_frac=dropout_frac,
                    ),
                    nn.ELU(),
                    nn.Conv3d(
                        in_channels=conv3d_channels,
                        out_channels=conv3d_channels,
                        kernel_size=(1, 2, 2),
                        padding=(0, 0, 0),
                        stride=(1, 2, 2),
                    ),
                )
            ]

        self.downscale_layers = nn.ModuleList(downscale_layers)

        self.crop_fn = CenterCrop(cnn_spatial_output)

        cat_channels = conv3d_channels * (1 + n_downscale)
        self.post_cat_conv = nn.Sequential(
            ResidualConv3dBlock(
                in_channels=cat_channels,
                n_layers=res_block_layers,
            ),
            nn.ELU(),
            nn.Conv3d(
                in_channels=cat_channels,
                out_channels=conv3d_channels,
                kernel_size=(1, 1, 1),
            ),
        )

        final_channels = (
            (image_size_pixels // (2**n_downscale)) ** 2 * conv3d_channels * sequence_length
        )
        self.final_layer = nn.Sequential(
            nn.ELU(),
            nn.Linear(
                in_features=final_channels,
                out_features=out_features,
            ),
            nn.ELU(),
        )

    def forward(self, x):
        """Run model forward"""
        out = self.first_layer(x)
        outputs = [self.crop_fn(out)]

        for layer in self.downscale_layers:
            out = layer(out)
            outputs += [self.crop_fn(out)]

        out = torch.cat(outputs, dim=1)
        out = self.post_cat_conv(out)
        out = torch.flatten(out, start_dim=1)
        out = self.final_layer(out)
        return out
