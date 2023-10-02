"""Basic blocks for image sequence encoders"""
from abc import ABCMeta, abstractmethod

import torch
from torch import nn


class AbstractNWPSatelliteEncoder(nn.Module, metaclass=ABCMeta):
    """Abstract class for NWP/satellite encoder.

    The encoder will take an input of shape (batch_size, sequence_length, channels, height, width)
    and return an output of shape (batch_size, out_features).
    """

    def __init__(
        self,
        sequence_length: int,
        image_size_pixels: int,
        in_channels: int,
        out_features: int,
    ):
        """Abstract class for NWP/satellite encoder.

        Args:
            sequence_length: The time sequence length of the data.
            image_size_pixels: The spatial size of the image. Assumed square.
            in_channels: Number of input channels.
            out_features: Number of output features.
        """
        super().__init__()
        self.out_features = out_features
        self.image_size_pixels = image_size_pixels

    @abstractmethod
    def forward(self):
        """Run model forward"""
        pass


class ResidualConv3dBlock(nn.Module):
    """Fully-connected deep network based on ResNet architecture.

    Internally, this network uses ELU activations throughout the residual blocks.
    """

    def __init__(
        self,
        in_channels,
        n_layers: int = 2,
        dropout_frac: float = 0.0,
    ):
        """Fully-connected deep network based on ResNet architecture.

        Args:
            in_channels: Number of input channels.
            n_layers: Number of layers in residual pathway.
            dropout_frac: Probability of an element to be zeroed.
        """
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers += [
                nn.ELU(),
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=(3, 3, 3),
                    padding=(1, 1, 1),
                ),
                nn.Dropout3d(p=dropout_frac),
            ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Run residual connection"""
        return self.model(x) + x


class ImageSequenceEncoder(nn.Module):
    """Simple network which independently encodes each image in a sequence into 1D features"""

    def __init__(
        self,
        image_size_pixels: int,
        in_channels: int,
        number_of_conv2d_layers: int = 4,
        conv2d_channels: int = 32,
        fc_features: int = 128,
    ):
        """Simple network which independently encodes each image in a sequence into 1D features.

        For input image with shape [N, C, L, H, W] the output is of shape [N, L, fc_features] where
        N is number of samples in batch, C is the number of input channels, L is the length of the
        sequence, and H and W are the height and width.

        Args:
            image_size_pixels: The spatial size of the image. Assumed square.
            in_channels: Number of input channels.
            number_of_conv2d_layers: Number of convolution 2D layers that are used.
            conv2d_channels: Number of channels used in each conv2d layer.
            fc_features: Number of output nodes for each image in each sequence.
        """
        super().__init__()

        # Check that the output shape of the convolutional layers will be at least 1x1
        cnn_spatial_output_size = image_size_pixels - 2 * number_of_conv2d_layers
        if not (cnn_spatial_output_size >= 1):
            raise ValueError(
                f"cannot use this many conv2d layers ({number_of_conv2d_layers}) with this input "
                f"spatial size ({image_size_pixels})"
            )

        conv_layers = []

        conv_layers += [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv2d_channels,
                kernel_size=3,
                padding=0,
            ),
            nn.ELU(),
        ]
        for i in range(0, number_of_conv2d_layers - 1):
            conv_layers += [
                nn.Conv2d(
                    in_channels=conv2d_channels,
                    out_channels=conv2d_channels,
                    kernel_size=3,
                    padding=0,
                ),
                nn.ELU(),
            ]

        self.conv_layers = nn.Sequential(*conv_layers)

        self.final_block = nn.Sequential(
            nn.Linear(
                in_features=(cnn_spatial_output_size**2) * conv2d_channels,
                out_features=fc_features,
            ),
            nn.ELU(),
        )

    def forward(self, x):
        """Run model forward"""
        batch_size, channel, seq_len, height, width = x.shape

        x = torch.swapaxes(x, 1, 2)
        x = x.reshape(batch_size * seq_len, channel, height, width)

        out = self.conv_layers(x)
        out = out.reshape(batch_size * seq_len, -1)

        out = self.final_block(out)
        out = out.reshape(batch_size, seq_len, -1)

        return out
