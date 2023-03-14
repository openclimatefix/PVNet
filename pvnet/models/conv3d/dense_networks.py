import torch
from torch import nn
from abc import ABCMeta, abstractmethod

from pvnet.models.base_model import BaseModel
from torchvision.transforms import CenterCrop


class AbstractTabularNetwork(nn.Module, metaclass=ABCMeta):
    """Abstract class for a network to combine the features from all the inputs.
    
    Args:
        in_features: Number of input features.
        out_features: Number of output features.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        
    @abstractmethod
    def forward(self):
        pass
        
    
class DefaultFCNet(AbstractTabularNetwork):
    """
    Similar to the original FCNet module used in PVNet, with a few minor tweaks.
    A 2-layer 

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        fc_hidden_features: Number of features in middle hidden layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fc_hidden_features: int=128,
    ):


        super().__init__(in_features, out_features)
    
        self.model = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=fc_hidden_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=fc_hidden_features, out_features=out_features),
        )


    def forward(self, x):
        return self.model(x)
    
  