"""Basic blocks for immage sequence encoders"""
from abc import ABCMeta, abstractmethod

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
