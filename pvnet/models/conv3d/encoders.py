import torch
from torch import nn
import torch.nn.functional as F
from abc import ABCMeta
from abc import ABCMeta, abstractmethod

from pvnet.models.base_model import BaseModel
from torchvision.transforms import CenterCrop
from torchvision.transforms.functional import center_crop

from pvnet.models.conv3d.basic_blocks import ResidualConv3dBlock


class AbstractNWPSatelliteEncoder(nn.Module, metaclass=ABCMeta):
    """Abstract class for NWP/satellite encoder. The encoder will take an input of shape
    (batch_size, sequence_length, channels, height, width) and return an output of shape
    (batch_size, out_features).
    
    Args:
        sequence_length: The time sequence length of the data.
        image_size_pixels: The spatial size of the image. Assumed square.
        in_channels: Number of input channels.
        out_features: Number of output features.
    """
    def __init__(
        self,
        sequence_length: int,
        image_size_pixels: int,
        in_channels: int,
        out_features: int,
    ):
        super().__init__()
        
    @abstractmethod
    def forward(self):
        pass
        
    
class DefaultPVNet(AbstractNWPSatelliteEncoder):
    """
    This is the original encoding module used in PVNet, with a few minor tweaks.

    Args:
        sequence_length: The time sequence length of the data.
        image_size_pixels: The spatial size of the image. Assumed square.
        in_channels: Number of input channels.
        out_features: Number of output features.
        number_of_conv3d_layers: Number of convolution 3d layers that are used.
        conv3d_channels: Number of channels used in each conv3d layer.
        fc_features: number of output nodes out of the hidden fully connected layer.
    """

    def __init__(
        self,
        sequence_length: int,
        image_size_pixels: int,
        in_channels: int,
        out_features: int,
        number_of_conv3d_layers: int = 4,
        conv3d_channels: int = 32,
        fc_features: int = 128,
    ):


        super().__init__(sequence_length, image_size_pixels, in_channels, out_features)

        cnn_spatial_output_size = (image_size_pixels - 2 * number_of_conv3d_layers)
        if not (cnn_spatial_output_size>0):
            raise ValueError(
                f"cannot use this many conv3d layers ({number_of_conv3d_layers}) with this input "
                f"spatial size ({image_size_pixels})"
            )
        
        conv_layers = []
        
        conv_layers += [
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=conv3d_channels,
                kernel_size=(3, 3, 3),
                padding=(1, 0, 0),
            ),
            nn.ELU(),
        ]
        for i in range(0, number_of_conv3d_layers - 1):
            conv_layers += [
                nn.Conv3d(
                    in_channels=conv3d_channels,
                    out_channels=conv3d_channels,
                    kernel_size=(3, 3, 3),
                    padding=(1, 0, 0),
                ),
                nn.ELU(),
            ]
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        cnn_output_size = (
            conv3d_channels
            * cnn_spatial_output_size**2
            * sequence_length
        )
        
        self.final_block = nn.Sequential(
            nn.Linear(
                in_features=cnn_output_size, out_features=fc_features
            ),
            nn.ELU(),
            nn.Linear(
                in_features=fc_features, out_features=out_features
            ),
            nn.ELU(),
        )

    def forward(self, x):
                
        out = self.conv_layers(x)
        out = out.reshape(x.shape[0], -1)

        # Fully connected layers
        out = self.final_block(out)
        
        return out
    
    
    
class EncoderUNET(AbstractNWPSatelliteEncoder):
    """
    An encoder for satellite and/or NWP data taking inspiration from the kinds of skip 
    connections in UNet. This differs from an actual UNet in that it does not have upsampling
    layers, instead it concats features from different spatial scales, and applies a few extra
    conv3d layers.

    Args:
        sequence_length: The time sequence length of the data.
        image_size_pixels: The spatial size of the image. Assumed square.
        in_channels: Number of input channels.
        out_features: Number of output features.
        n_downscale: Number of conv3d and spatially downscaling layers that are used.
        conv3d_channels: Number of channels used in each conv3d layer.
        fc_features: number of output nodes out of the hidden fully connected layer.
        dropout_frac: Probability of an element to be zeroed in the residual pathways.
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
        
        cnn_spatial_output = image_size_pixels//(2**n_downscale)
        
        if not (cnn_spatial_output>0):
            raise ValueError(
                f"cannot use this many downscaling layers ({n_downscale}) with this input "
                f"spatial size ({image_size_pixels})"
            )
        self.cnn_spatial_output = cnn_spatial_output
            
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
                        stride=(1,2,2),
                    ),
                )
            ]
            
        self.downscale_layers = nn.ModuleList(downscale_layers)
        
        #self.crop_fn = CenterCrop(image_size_pixels//(2**n_downscale))
        
        cat_channels = conv3d_channels*(1+n_downscale)
        self.post_cat_conv = nn.Sequential(
            ResidualConv3dBlock(
                in_channels=cat_channels, 
                n_layers=res_block_layers,
            ),
            nn.ELU(),
            nn.Conv3d(
                in_channels=cat_channels, 
                out_channels=conv3d_channels, 
                kernel_size=(1,1,1),
            ),
        )
        
        final_channels = (
            (image_size_pixels//(2**n_downscale))**2
            *conv3d_channels
            *sequence_length
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
                
        out = self.first_layer(x)
        outputs = [center_crop(out, self.cnn_spatial_output)]
            
        for layer in self.downscale_layers:
            out = layer(out)
            outputs += [center_crop(out, self.cnn_spatial_output)]
            #outputs += [self.crop_fn(out)]
        
        out = torch.cat(outputs, dim=1)
        out = self.post_cat_conv(out)
        out = torch.flatten(out, start_dim=1)
        out = self.final_layer(out)
        return out
    
    
class EncoderNaiveEfficientNet(AbstractNWPSatelliteEncoder):
    """
    A naive implementation of EfficientNet as an encoder for the satellite/NWP data.
    Stacks the time dimension into extra channels.

    Args:
        sequence_length: The time sequence length of the data.
        image_size_pixels: The spatial size of the image. Assumed square.
        in_channels: Number of input channels.
        out_features: Number of output features.
        model_name: Name for efficientnet.
    """
    
    def __init__(
        self,
        sequence_length: int,
        image_size_pixels: int,
        in_channels: int,
        out_features: int,
        model_name: str = "efficientnet-b0",
    ):
        
        try:
            from efficientnet_pytorch import EfficientNet
        except:
            raise ImportError(
                "The efficientnet_pytorch package must be installed to use the " 
                "EncoderNaiveEfficientNet encoder. See "
                "https://github.com/lukemelas/EfficientNet-PyTorch for install instructions."
            )

        super().__init__(sequence_length, image_size_pixels, in_channels, out_features)
        

        self.model = EfficientNet.from_name(
            model_name, 
            in_channels=in_channels*sequence_length, 
            image_size=image_size_pixels, 
            num_classes=out_features
        )

    def forward(self, x):
        
        bs, s, c, h, w = x.shape
        
        x = x.reshape((bs, s*c, h, w))
        
        return self.model(x)