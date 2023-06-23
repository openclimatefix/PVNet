"""Encoder modules for the satellite/NWP data based on recursive and 2D convolutional layers.
"""

import torch
from torch import nn

from pvnet.models.multimodal.encoders.basic_blocks import (
    AbstractNWPSatelliteEncoder,
    ImageSequenceEncoder,
)


class ConvLSTM(AbstractNWPSatelliteEncoder):
    """Convolutional LSTM block from MetNet."""

    def __init__(
        self,
        sequence_length: int,
        image_size_pixels: int,
        in_channels: int,
        out_features: int,
        hidden_channels: int = 32,
        num_layers: int = 2,
        kernel_size: int = 3,
        bias: bool = True,
        activation=torch.tanh,
        batchnorm=False,
    ):
        """Convolutional LSTM block from MetNet.

        Args:
            sequence_length: The time sequence length of the data.
            image_size_pixels: The spatial size of the image. Assumed square.
            in_channels: Number of input channels.
            out_features: Number of output features.
            hidden_channels: Hidden dimension size.
            num_layers: Depth of ConvLSTM cells.
            kernel_size: Kernel size.
            bias: Whether to add bias.
            activation: Activation function for ConvLSTM cells.
            batchnorm: Whether to use batch norm.
        """
        from metnet.layers.ConvLSTM import ConvLSTM as _ConvLSTM

        super().__init__(sequence_length, image_size_pixels, in_channels, out_features)

        self.conv_lstm = _ConvLSTM(
            input_dim=in_channels,
            hidden_dim=hidden_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
            activation=activation,
            batchnorm=batchnorm,
        )

        # Calculate the size of the output of the ConvLSTM network
        convlstm_output_size = hidden_channels * image_size_pixels**2

        self.final_block = nn.Sequential(
            nn.Linear(in_features=convlstm_output_size, out_features=out_features),
            nn.ELU(),
        )

    def forward(self, x):
        """Run model forward"""

        batch_size, channel, seq_len, height, width = x.shape
        x = torch.swapaxes(x, 1, 2)

        res, _ = self.conv_lstm(x)

        # Select last state only
        out = res[:, -1]

        # Flatten and fully connected layer
        out = out.reshape(batch_size, -1)
        out = self.final_block(out)

        return out


class FlattenLSTM(AbstractNWPSatelliteEncoder):
    """Convolutional blocks followed by LSTM."""

    def __init__(
        self,
        sequence_length: int,
        image_size_pixels: int,
        in_channels: int,
        out_features: int,
        num_layers: int = 2,
        number_of_conv2d_layers: int = 4,
        conv2d_channels: int = 32,
    ):
        """Network consisting of 2D spatial convolutional and LSTM sequence encoder.

        Args:
            sequence_length: The time sequence length of the data.
            image_size_pixels: The spatial size of the image. Assumed square.
            in_channels: Number of input channels.
            out_features: Number of output features. Also used for LSTM hidden dimension.
            num_layers: Number of recurrent layers. E.g., setting num_layers=2 would mean stacking
                two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of
                the first LSTM and computing the final results.
            number_of_conv2d_layers: Number of convolution 2D layers that are used.
            conv2d_channels: Number of channels used in each conv2d layer.
        """

        super().__init__(sequence_length, image_size_pixels, in_channels, out_features)

        self.lstm = nn.LSTM(
            input_size=out_features,
            hidden_size=out_features,
            num_layers=num_layers,
            batch_first=True,
        )

        self.encode_image_sequence = ImageSequenceEncoder(
            image_size_pixels=image_size_pixels,
            in_channels=in_channels,
            number_of_conv2d_layers=number_of_conv2d_layers,
            conv2d_channels=conv2d_channels,
            fc_features=out_features,
        )

        self.final_block = nn.Sequential(
            nn.Linear(in_features=out_features, out_features=out_features),
            nn.ELU(),
        )

    def forward(self, x):
        """Run model forward"""
        encoded_images = self.encode_image_sequence(x)

        _, (_, c_n) = self.lstm(encoded_images)

        # Take only the deepest level hidden cell state
        out = self.final_block(c_n[-1])

        return out
